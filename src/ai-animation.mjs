/**
 * Prompt -> Animation generator for StykAnimate.
 * Uses LLM to produce a structured animation plan, then converts to document keyframes.
 *
 * Design goals:
 * - Works with any OpenAI-compatible LLM (no vision model required).
 * - LLM only plans *what* to do; we map to safe, validated poses from a motion library
 *   so we never hallucinate impossible joint angles.
 * - User can custom provider/model/baseUrl via Settings -> AI (ai-provider.mjs)
 *
 * Two modes:
 *  1) Library mode (default, robust): LLM picks sequence of named moves (stand, walk, punch...)
 *     We interpolate those moves into real figure poses.
 *  2) Freeform mode (advanced): LLM outputs joint angles directly — we validate against rig constraints.
 */

import { callChatCompletion } from './ai-provider.mjs';

// Minimal motion library: named poses -> relaxed joint deltas relative to Blue's rest pose.
// These are *deltas* for demo; real pose generation uses editor commands for exact IK.
// For v1 we emit a descriptive plan that the editor can preview as text + simple keyframes.
export const MOTION_LIBRARY = Object.freeze({
  stand: { label: 'Stand', description: 'Upright idle, feet shoulder-width' },
  walk: { label: 'Walk', description: 'Alternating steps, arms swing' },
  run: { label: 'Run', description: 'Faster walk, leaning forward' },
  jump: { label: 'Jump', description: 'Crouch then rise, legs extend' },
  punch: { label: 'Punch', description: 'Forward arm extends, torso twist' },
  kick: { label: 'Kick', description: 'One leg forward kick, balance on other' },
  salto: { label: 'Salto', description: 'Backflip kick, airborne rotation' },
  block: { label: 'Block', description: 'Arms raised defensive' },
  hadouken: { label: 'Hadouken', description: 'Charge then push, both arms forward' },
  dance: { label: 'Dance', description: 'Rhythmic side steps' },
  fall: { label: 'Fall', description: 'Collapse to ground' },
  getup: { label: 'Get up', description: 'Rise from ground to stand' },
});

export const MOTION_IDS = Object.keys(MOTION_LIBRARY);

/**
 * System prompt that teaches LLM the StykAnimate animation schema.
 * We want LLM to output JSON only, with validated motion ids.
 */
export function buildSystemPrompt() {
  return `You are StykAnimate AI Director. You convert a user prompt into a stickman animation plan.

You MUST output ONLY valid JSON (no markdown, no explanation) with this exact schema:
{
  "title": "short title 2-4 words",
  "fps": 24,
  "durationFrames": 48,
  "figures": 1,
  "sequence": [
    {"move": "stand", "frames": 6, "easing": "easeOut"},
    {"move": "punch", "frames": 8, "easing": "easeIn"},
    {"move": "kick", "frames": 12, "easing": "easeInOut"}
  ]
}

Rules:
- move MUST be one of: ${MOTION_IDS.join(', ')}
- frames: integer 3..30 per move
- easing: "linear" | "easeIn" | "easeOut" | "easeInOut"
- durationFrames = sum of frames, 12..120
- figures: 1 or 2
- Keep sequence 2..6 moves long.
- Map user intent to the closest moves. "salto" for flips, "hadouken" for energy attack, etc.
- Respond ONLY with JSON. No comments.
Example for "stickman punch combo":
{"title":"Punch Combo","fps":24,"durationFrames":48,"figures":1,"sequence":[{"move":"stand","frames":8,"easing":"easeOut"},{"move":"punch","frames":6,"easing":"easeIn"},{"move":"punch","frames":6,"easing":"easeIn"},{"move":"kick","frames":14,"easing":"easeInOut"},{"move":"stand","frames":14,"easing":"easeOut"}]}`;
}

export function buildUserPrompt(prompt, opts = {}) {
  const figures = opts.figures ?? 1;
  const style = opts.style ? ` Style: ${opts.style}.` : '';
  return `Prompt: "${prompt}" Figures: ${figures}.${style} Output JSON only.`;
}

/**
 * Parse LLM content -> animation plan JSON.
 * Handles markdown fences, extra whitespace.
 */
export function parsePlan(content) {
  let text = String(content || '').trim();
  // Remove markdown code fences if present
  text = text.replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/i, '').trim();
  // Extract first JSON object
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start === -1 || end === -1) throw new Error('LLM did not return JSON object');
  text = text.slice(start, end + 1);
  const raw = JSON.parse(text);
  return normalizePlan(raw);
}

export function normalizePlan(raw = {}) {
  const fps = Number.isFinite(raw.fps) ? Math.max(1, Math.min(60, Math.round(raw.fps))) : 24;
  const title = typeof raw.title === 'string' ? raw.title.slice(0, 60) : 'AI Animation';
  let figures = Number.isFinite(raw.figures) ? Math.round(raw.figures) : 1;
  figures = Math.max(1, Math.min(2, figures));
  let sequence = Array.isArray(raw.sequence) ? raw.sequence : [];
  sequence = sequence.map(entry => {
    const move = typeof entry.move === 'string' ? entry.move.toLowerCase() : '';
    const frames = Number.isFinite(entry.frames) ? Math.max(3, Math.min(30, Math.round(entry.frames))) : 8;
    const easing = ['linear','easeIn','easeOut','easeInOut'].includes(entry.easing) ? entry.easing : 'easeInOut';
    if (!MOTION_IDS.includes(move)) return null;
    return { move, frames, easing };
  }).filter(Boolean).slice(0, 8);
  if (sequence.length < 2) {
    sequence = [{move:'stand', frames:8, easing:'easeOut'}, {move:'punch', frames:10, easing:'easeIn'}, {move:'stand', frames:8, easing:'easeOut'}];
  }
  let durationFrames = sequence.reduce((a,b)=>a+b.frames, 0);
  if (Number.isFinite(raw.durationFrames)) {
    const claimed = Math.max(12, Math.min(180, Math.round(raw.durationFrames)));
    // If claimed differs a lot, keep sum but warn; trust sum for consistency
    if (Math.abs(claimed - durationFrames) > 6) durationFrames = durationFrames; else durationFrames = claimed;
  }
  return { title, fps, durationFrames, figures, sequence };
}

/**
 * Main: prompt -> plan via LLM.
 * Requires a provider {baseUrl, model, apiKey}
 */
export async function generatePlan(prompt, provider, opts = {}) {
  if (!prompt || !String(prompt).trim()) throw new Error('Prompt is empty');
  if (!provider?.baseUrl) throw new Error('No AI provider configured. Add one in Settings → AI.');
  const messages = [
    { role: 'system', content: buildSystemPrompt() },
    { role: 'user', content: buildUserPrompt(prompt, opts) },
  ];
  const content = await callChatCompletion(provider, messages, {
    temperature: 0.7,
    maxTokens: 1200,
    signal: opts.signal,
  });
  return parsePlan(content);
}

/**
 * Local fallback when no provider is configured or offline: keyword heuristic.
 * Produces a plausible plan without LLM.
 */
export function generatePlanLocal(prompt = '') {
  const p = String(prompt).toLowerCase();
  const seq = [];
  const push = (move, frames=8, easing='easeInOut') => seq.push({move, frames, easing});
  push('stand', 8, 'easeOut');
  if (p.includes('kick') || p.includes('tendang')) { push('kick', 12, 'easeIn'); }
  else if (p.includes('punch') || p.includes('pukul')) { push('punch', 8, 'easeIn'); if (p.includes('combo')) push('punch', 6, 'easeIn'); }
  else if (p.includes('jump') || p.includes('loncat') || p.includes('lompat')) { push('jump', 10, 'easeOut'); }
  else if (p.includes('run') || p.includes('lari')) { push('run', 14, 'linear'); }
  else if (p.includes('walk') || p.includes('jalan')) { push('walk', 12, 'linear'); }
  else if (p.includes('salto') || p.includes('flip')) { push('salto', 16, 'easeInOut'); }
  else if (p.includes('hadouken') || p.includes('energy') || p.includes('kame')) { push('hadouken', 14, 'easeIn'); }
  else if (p.includes('dance') || p.includes('dansa') || p.includes('joget')) { push('dance', 16, 'easeInOut'); }
  else {
    // generic fight
    push('punch', 8, 'easeIn'); push('kick', 12, 'easeInOut');
  }
  if (p.includes('fall') || p.includes('jatuh')) push('fall', 10, 'easeIn');
  push('stand', 8, 'easeOut');
  const title = prompt.slice(0, 30) || 'AI Animation';
  const durationFrames = seq.reduce((a,b)=>a+b.frames,0);
  return { title, fps: 24, durationFrames, figures: p.includes('2') || p.includes('dua') ? 2 : 1, sequence: seq };
}

/**
 * Convert plan -> human readable summary and placeholder document patch.
 * For v1, we don't mutate the actual figure geometry; we return a plan that
 * the UI can display and that future version can commit as keyframes via editor commands.
 *
 * Returns: {plan, summary, documentPatchSuggestion}
 */
export function planToSummary(plan) {
  const lines = plan.sequence.map(s => `${MOTION_LIBRARY[s.move].label} ×${s.frames}f (${s.easing})`);
  return `${plan.title} — ${plan.fps}fps, ${plan.durationFrames}f, ${plan.figures} fig. → ${lines.join(' → ')}`;
}
