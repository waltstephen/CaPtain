// Code-mode runner: execute the model's JavaScript with Codex's `exec` helpers.
// Nested tools are proxied to the Python driver over a loopback HTTP endpoint.
// Protocol: env CODE_MODE_ENDPOINT (POST /call {name, args}) ; script source on stdin ;
// result JSON on stdout: {"status": "completed"|"failed", "parts": [...], "error": string|null}
'use strict';
const http = require('http');

const endpoint = new URL(process.env.CODE_MODE_ENDPOINT);
const toolNames = JSON.parse(process.env.CODE_MODE_TOOLS || '[]');
const modern = process.env.CODE_MODE_PROTOCOL === '2';
const descriptions = JSON.parse(process.env.CODE_MODE_TOOL_DESCRIPTIONS || '{}');
const parts = [];
const stored = new Map();

function callTool(name, args) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({ name, args: args === undefined ? {} : args });
    const req = http.request({
      hostname: endpoint.hostname, port: endpoint.port, path: '/call', method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
    }, (res) => {
      let data = '';
      res.setEncoding('utf8');
      res.on('data', (c) => { data += c; });
      res.on('end', () => {
        let payload;
        try { payload = JSON.parse(data); } catch (e) { reject(new Error(`tool ${name} returned invalid JSON`)); return; }
        if (payload && payload.error !== undefined && payload.error !== null) { reject(new Error(String(payload.error))); return; }
        resolve(payload.result);
      });
    });
    req.on('error', (e) => reject(new Error(`tool ${name} failed: ${e.message}`)));
    req.write(body);
    req.end();
  });
}

const tools = {};
for (const name of toolNames) {
  tools[name] = (args) => callTool(name, args);
}
const ALL_TOOLS = toolNames.map((name) => ({ name, description: descriptions[name] || '' }));

class ExitSignal extends Error {}
function exit() { throw new ExitSignal('exit'); }
function text(value) {
  if (typeof value === 'string') { parts.push({ type: 'input_text', text: value }); return; }
  let rendered;
  try { rendered = JSON.stringify(value); } catch (e) { rendered = String(value); }
  parts.push({ type: 'input_text', text: rendered === undefined ? String(value) : rendered });
}
function image(item, detail) {
  let url = null; let d = detail === undefined ? null : detail;
  if (typeof item === 'string') { url = item; }
  else if (item && typeof item === 'object') {
    url = item.image_url || (item.data && item.mimeType ? `data:${item.mimeType};base64,${item.data}` : null);
    if (d === null && item.detail) d = item.detail;
  }
  if (!url) throw new Error('image() requires a data: URL or an image item');
  if (modern && (typeof url !== 'string' || !url.startsWith('data:image/'))) {
    throw new Error('image() requires a valid data-image URL, for example from tools.view_image; external images are unavailable');
  }
  if (typeof url !== 'string' || !/^(data:image\/[A-Za-z0-9.+-]+;base64,|https?:\/\/)/.test(url)) {
    throw new Error('image() requires a data:image/...;base64 URL (the image_url returned by tools.view_image on an image file) or an http(s) URL; got: ' + String(url).slice(0, 80));
  }
  if (url.startsWith('data:')) {
    const payload = url.slice(url.indexOf(',') + 1);
    if (!/^[A-Za-z0-9+/]+={0,2}$/.test(payload) || payload.length % 4 !== 0) {
      throw new Error('image() got a data: URL whose base64 payload is invalid; pass the image_url returned by tools.view_image unchanged');
    }
  }
  parts.push({ type: 'input_image', image_url: url, detail: d || 'high' });
}
function audio(item) { throw new Error('audio() is not supported in this session'); }
function generatedImage(item) { throw new Error('generatedImage() is not supported in this session'); }
function jsonCopy(value) {
  const encoded = JSON.stringify(value);
  if (encoded === undefined) throw new Error('store() requires a JSON-serializable value');
  return JSON.parse(encoded);
}
function store(key, value) { stored.set(String(key), modern ? jsonCopy(value) : value); }
function load(key) {
  const value = stored.get(String(key));
  return modern && value !== undefined ? jsonCopy(value) : value;
}
function notify(value) { text(value); }
function yield_control() { /* Compatibility: scripts run to completion. */ }

async function main() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  let source = Buffer.concat(chunks).toString('utf8');
  if (modern) {
    const envelope = JSON.parse(source);
    source = envelope.source;
    for (const [key, value] of Object.entries(envelope.store || {})) stored.set(key, value);
  }
  const lines = source.split('\n');
  if (lines.length && /^[ \t]*\/\/ @exec:/.test(lines[0])) { source = lines.slice(1).join('\n'); }
  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
  let status = 'completed'; let error = null;
  try {
    const fn = new AsyncFunction('tools', 'text', 'image', 'audio', 'generatedImage', 'exit', 'store', 'load', 'notify', 'yield_control', 'ALL_TOOLS', 'setTimeout', 'clearTimeout', source);
    await fn(tools, text, image, audio, generatedImage, exit, store, load, notify, yield_control, ALL_TOOLS, setTimeout, clearTimeout);
  } catch (e) {
    if (!(e instanceof ExitSignal)) { status = 'failed'; error = (e && e.message) ? e.message : String(e); }
  }
  process.stdout.write(JSON.stringify({ status, parts, error, ...(modern ? { store: Object.fromEntries(stored) } : {}) }));
}

main().catch((e) => { process.stdout.write(JSON.stringify({ status: 'failed', parts, error: String(e && e.message || e) })); });
