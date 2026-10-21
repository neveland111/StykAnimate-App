/**
 * AI Provider management for StykAnimate (Prompt -> Animation).
 * OpenAI-compatible chat API: every provider has {id, name, baseUrl, model, apiKey}.
 * Stored in EditorSettings (stykanimate.editor.settings.v1) so user can add providers.
 * No hardcoded provider: user adds them via Settings -> AI.
 *
 * Supports: OpenAI, OpenRouter, Groq, Ollama (http://localhost:11434/v1), any OpenAI-compatible.
 */

export const AI_SETTINGS_KEY = 'styk-ai-providers';

/**
 * Normalize a single provider entry.
 * baseUrl is trimmed and trailing slash removed. apiKey may be empty for Ollama.
 */
export function normalizeAIProvider(raw = {}) {
  const id = typeof raw.id === 'string' && raw.id.trim() ? raw.id.trim().slice(0, 64) : `ai-${Date.now().toString(36)}-${Math.random().toString(36).slice(2,6)}`;
  const name = typeof raw.name === 'string' && raw.name.trim() ? raw.name.trim().slice(0, 100) : 'Custom Provider';
  let baseUrl = typeof raw.baseUrl === 'string' ? raw.baseUrl.trim() : '';
  // Remove trailing slash
  if (baseUrl.endsWith('/')) baseUrl = baseUrl.slice(0, -1);
  // Basic validation: must start with http(s)://
  if (baseUrl && !/^https?:\/\/.+/.test(baseUrl)) baseUrl = '';
  const model = typeof raw.model === 'string' && raw.model.trim() ? raw.model.trim().slice(0, 120) : 'gpt-4o-mini';
  const apiKey = typeof raw.apiKey === 'string' ? raw.apiKey.trim().slice(0, 500) : '';
  const enabled = raw.enabled !== false; // default true
  return { id, name, baseUrl, model, apiKey, enabled };
}

export function normalizeAIProviders(list) {
  if (!Array.isArray(list)) return [];
  const seen = new Set();
  const out = [];
  for (const item of list) {
    const p = normalizeAIProvider(item);
    if (!p.baseUrl) continue; // require baseUrl
    // deduplicate by id
    let pid = p.id;
    while (seen.has(pid)) pid = pid + '-' + Math.random().toString(36).slice(2,4);
    p.id = pid;
    seen.add(pid);
    out.push(Object.freeze(p));
    if (out.length >= 20) break; // cap 20 providers
  }
  return Object.freeze(out);
}

export const DEFAULT_AI_SETTINGS = Object.freeze({
  aiProviders: Object.freeze([]),
  aiActiveProviderId: null,
  aiAutoGenerate: false,
});

/**
 * Full normalization for EditorSettings part.
 */
export function normalizeAISettings(value = {}) {
  const providers = normalizeAIProviders(value.aiProviders);
  let activeId = typeof value.aiActiveProviderId === 'string' ? value.aiActiveProviderId : null;
  if (activeId && !providers.some(p => p.id === activeId)) activeId = null;
  if (!activeId && providers.length) activeId = providers[0].id;
  return {
    aiProviders: providers,
    aiActiveProviderId: activeId,
  };
}

export function getActiveProvider(settings) {
  if (!settings?.aiProviders?.length) return null;
  const id = settings.aiActiveProviderId;
  return settings.aiProviders.find(p => p.id === id) ?? settings.aiProviders[0] ?? null;
}

/**
 * OpenAI-compatible chat completion.
 * Can go either via backend proxy (/api/ai/chat) or directly to baseUrl.
 * Prefer backend proxy to avoid CORS and to hide key from browser history if desktop.
 *
 * @param {object} provider {baseUrl, model, apiKey}
 * @param {Array} messages [{role, content}]
 * @param {object} opts {temperature, maxTokens, signal, useProxy}
 */
export async function callChatCompletion(provider, messages, opts = {}) {
  if (!provider?.baseUrl) throw new Error('Provider baseUrl is required. Add a provider in Settings → AI.');
  if (!provider?.model) throw new Error('Provider model is required.');
  const body = JSON.stringify({
    model: provider.model,
    messages,
    temperature: opts.temperature ?? 0.7,
    max_tokens: opts.maxTokens ?? 1500,
    stream: false,
  });

  const headers = {
    'Content-Type': 'application/json',
  };
  if (provider.apiKey) headers['Authorization'] = `Bearer ${provider.apiKey}`;

  // Try backend proxy first if available (loopback 47831), fallback to direct
  const useProxy = opts.useProxy !== false; // default try proxy
  let url;
  let fetchHeaders = headers;
  if (useProxy) {
    // Backend proxy expects {baseUrl, model, apiKey, messages, temperature, maxTokens}
    // It will forward server-side to avoid CORS for Ollama/local.
    url = '/api/ai/chat';
    // For proxy, we don't add Authorization here; backend will forward apiKey securely.
    // But we still need to pass baseUrl/model via body.
    const proxyBody = JSON.stringify({
      baseUrl: provider.baseUrl,
      model: provider.model,
      apiKey: provider.apiKey || '',
      messages,
      temperature: opts.temperature ?? 0.7,
      maxTokens: opts.maxTokens ?? 1500,
    });
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...(opts.signal ? {}: {}) },
        body: proxyBody,
        signal: opts.signal,
      });
      if (res.ok) {
        const data = await res.json();
        // Backend may return OpenAI shape {choices:[{message:{content}}]} or {content}
        if (data?.choices?.[0]?.message?.content) return data.choices[0].message.content;
        if (typeof data?.content === 'string') return data.content;
        if (typeof data?.text === 'string') return data.text;
        return JSON.stringify(data);
      }
      // If proxy not found (404) fall through to direct
      if (res.status !== 404 && res.status !== 501) {
        const errText = await res.text().catch(()=> '');
        throw new Error(`AI proxy error ${res.status}: ${errText || res.statusText}`);
      }
    } catch (e) {
      if (e?.name === 'AbortError') throw e;
      // Fall through to direct fetch if proxy failed due to network/404
      if (String(e.message).includes('AI proxy error')) throw e;
    }
  }

  // Direct fetch to provider (needs CORS allowed)
  url = provider.baseUrl.replace(/\/$/, '') + '/chat/completions';
  const res = await fetch(url, {
    method: 'POST',
    headers: fetchHeaders,
    body,
    signal: opts.signal,
  });
  if (!res.ok) {
    const text = await res.text().catch(()=> res.statusText);
    throw new Error(`LLM error ${res.status}: ${text.slice(0,500)}`);
  }
  const data = await res.json();
  const content = data?.choices?.[0]?.message?.content;
  if (!content) throw new Error('LLM returned no content');
  return content;
}

/**
 * Simple test of provider connectivity.
 */
export async function testProvider(provider, signal) {
  const content = await callChatCompletion(provider, [
    { role: 'user', content: 'Reply with exactly: OK' }
  ], { maxTokens: 10, temperature: 0, signal });
  if (!/OK/i.test(content)) throw new Error(`Unexpected reply: ${content.slice(0,100)}`);
  return content;
}
