/* ═══ TurboQuant Web UI v0.3 ═══ */

const API = '';
let workspaces = [];
let currentWorkspaceId = null;
let conversations = [];
let currentConvId = null;
let abortController = null;
let isGenerating = false;
let downloadPollInterval = null;

// ─── DOM ───────────────────────────────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ─── Toast ─────────────────────────────────────────────────────────────────────

function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  $('#toast-container').appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }, 3500);
}

// ─── Tab Navigation ────────────────────────────────────────────────────────────

$$('.nav-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.nav-tab').forEach(b => b.classList.remove('active'));
    $$('.tab-content').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    $(`#tab-${btn.dataset.tab}`).classList.add('active');
    if (btn.dataset.tab === 'models') refreshModelsPage();
    if (btn.dataset.tab === 'bench') refreshBenchPage();
  });
});

// ─── Init ──────────────────────────────────────────────────────────────────────

async function init() {
  loadWorkspaces();
  loadConversations();
  await refreshChatModels();
  await refreshStatus();
  renderWorkspaceSelector();
  if (conversations.length === 0) newChat();
  setInterval(refreshStatus, 5000);
  setInterval(pollDownloads, 2000);
}

// ─── Status Bar ────────────────────────────────────────────────────────────────

async function refreshStatus() {
  try {
    const res = await fetch(`${API}/api/status`);
    const d = await res.json();
    const dot = $('#nav-engine');
    dot.className = d.engine_running ? 'status-dot online' : 'status-dot offline';
    dot.title = d.engine_running ? `Running: ${d.current_model}` : 'Idle';
    if (d.gpu && d.gpu.vram_total_mb) {
      const pct = Math.round(d.gpu.vram_used_mb / d.gpu.vram_total_mb * 100);
      $('#nav-gpu-label').textContent = `GPU ${pct}%`;
      $('#nav-gpu-fill').style.width = `${pct}%`;
      const c = pct > 90 ? 'var(--danger)' : pct > 70 ? 'var(--warning)' : 'var(--accent)';
      $('#nav-gpu-fill').style.background = c;
    }
    if (d.config) {
      const gl = $('#setting-gpu-layers');
      const cs = $('#setting-ctx-size');
      if (gl && !gl.matches(':focus')) gl.value = d.config.gpu_layers || 99;
      if (cs && !cs.matches(':focus')) cs.value = d.config.ctx_size || 4096;
    }
  } catch (e) {
    $('#nav-engine').className = 'status-dot offline';
    $('#nav-engine').title = 'Offline';
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHAT TAB
// ═══════════════════════════════════════════════════════════════════════════════

async function refreshChatModels() {
  try {
    const res = await fetch(`${API}/api/tags`);
    const data = await res.json();
    const sel = $('#chat-model-select');
    const benchSel = $('#bench-model-select');
    sel.innerHTML = '';
    if (benchSel) benchSel.innerHTML = '';
    if (data.models && data.models.length > 0) {
      data.models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = `${m.name} (${fmtSize(m.size)})`;
        sel.appendChild(opt);
        if (benchSel) benchSel.appendChild(opt.cloneNode(true));
      });
    } else {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No models — go to Models tab';
      sel.appendChild(opt);
    }
  } catch (e) { /* ignore */ }
}

function loadWorkspaces() {
  try {
    workspaces = JSON.parse(localStorage.getItem('tq_workspaces') || '[]');
    if (workspaces.length === 0) {
      workspaces = [{ id: 'default', name: 'Default Workspace', created: Date.now() }];
      localStorage.setItem('tq_workspaces', JSON.stringify(workspaces));
    }
    currentWorkspaceId = localStorage.getItem('tq_current_workspace') || workspaces[0].id;
  } catch {
    workspaces = [{ id: 'default', name: 'Default Workspace', created: Date.now() }];
    currentWorkspaceId = 'default';
  }
}

function loadConversations() {
  const key = `tq_convs_${currentWorkspaceId}`;
  try { conversations = JSON.parse(localStorage.getItem(key) || '[]'); }
  catch { conversations = []; }
  renderConvList();
}

function saveConversations() {
  const key = `tq_convs_${currentWorkspaceId}`;
  localStorage.setItem(key, JSON.stringify(conversations));
  renderConvList();
}

function saveWorkspaces() {
  localStorage.setItem('tq_workspaces', JSON.stringify(workspaces));
  localStorage.setItem('tq_current_workspace', currentWorkspaceId);
}

function renderConvList() {
  const el = $('#conversations-list');
  el.innerHTML = '';
  conversations.forEach(c => {
    const d = document.createElement('div');
    d.className = `conv-item${c.id === currentConvId ? ' active' : ''}`;
    d.textContent = c.title || 'New Chat';
    d.onclick = () => switchConv(c.id);
    el.appendChild(d);
  });
}

function newChat() {
  const c = { 
    id: Date.now().toString(), 
    title: 'New Chat', 
    messages: [], 
    model: $('#chat-model-select').value,
    workspace: currentWorkspaceId,
    created: Date.now()
  };
  conversations.unshift(c);
  currentConvId = c.id;
  saveConversations();
  renderMessages();
}

function createWorkspace() {
  const name = prompt('Workspace name:', `Project ${workspaces.length + 1}`);
  if (!name) return;
  const ws = { id: Date.now().toString(), name, created: Date.now() };
  workspaces.push(ws);
  saveWorkspaces();
  switchWorkspace(ws.id);
  renderWorkspaceSelector();
  toast(`Workspace "${name}" created`, 'success');
}

function switchWorkspace(id) {
  if (currentWorkspaceId === id) return;
  currentWorkspaceId = id;
  saveWorkspaces();
  loadConversations();
  renderWorkspaceSelector();
  if (conversations.length > 0) {
    switchConv(conversations[0].id);
  } else {
    currentConvId = null;
    renderMessages();
  }
  const ws = workspaces.find(w => w.id === id);
  toast(`Switched to "${ws?.name || 'Unknown'}"`, 'info');
}

function deleteWorkspace(id) {
  if (workspaces.length === 1) {
    toast('Cannot delete the last workspace', 'error');
    return;
  }
  const ws = workspaces.find(w => w.id === id);
  if (!confirm(`Delete workspace "${ws?.name}"? All conversations will be lost.`)) return;
  workspaces = workspaces.filter(w => w.id !== id);
  localStorage.removeItem(`tq_convs_${id}`);
  if (currentWorkspaceId === id) {
    switchWorkspace(workspaces[0].id);
  }
  saveWorkspaces();
  renderWorkspaceSelector();
  toast(`Workspace "${ws?.name}" deleted`, 'info');
}

function renameWorkspace(id) {
  const ws = workspaces.find(w => w.id === id);
  if (!ws) return;
  const newName = prompt('New workspace name:', ws.name);
  if (!newName || newName === ws.name) return;
  ws.name = newName;
  saveWorkspaces();
  renderWorkspaceSelector();
  toast(`Workspace renamed to "${newName}"`, 'success');
}

function renderWorkspaceSelector() {
  const container = $('#workspace-selector');
  if (!container) return;
  const current = workspaces.find(w => w.id === currentWorkspaceId);
  container.innerHTML = `
    <div class="workspace-current" onclick="toggleWorkspaceMenu()">
      <span class="workspace-icon">📁</span>
      <span class="workspace-name">${escHtml(current?.name || 'Workspace')}</span>
      <span class="workspace-arrow">▼</span>
    </div>
    <div id="workspace-menu" class="workspace-menu hidden">
      ${workspaces.map(ws => `
        <div class="workspace-item ${ws.id === currentWorkspaceId ? 'active' : ''}">
          <span class="ws-name" onclick="switchWorkspace('${ws.id}')">${escHtml(ws.name)}</span>
          <div class="ws-actions">
            <button onclick="event.stopPropagation(); renameWorkspace('${ws.id}')" title="Rename">✏️</button>
            ${workspaces.length > 1 ? `<button onclick="event.stopPropagation(); deleteWorkspace('${ws.id}')" title="Delete">🗑️</button>` : ''}
          </div>
        </div>
      `).join('')}
      <div class="workspace-item new" onclick="createWorkspace()">
        <span class="ws-name">+ New Workspace</span>
      </div>
    </div>
  `;
}

function toggleWorkspaceMenu() {
  const menu = $('#workspace-menu');
  if (!menu) return;
  menu.classList.toggle('hidden');
}

document.addEventListener('click', (e) => {
  const menu = $('#workspace-menu');
  const selector = $('#workspace-selector');
  if (menu && selector && !selector.contains(e.target)) {
    menu.classList.add('hidden');
  }
});

function switchConv(id) {
  currentConvId = id;
  renderConvList();
  renderMessages();
}

function getConv() { return conversations.find(c => c.id === currentConvId); }

function renderMessages() {
  const c = getConv();
  $('#messages').innerHTML = '';
  if (!c || c.messages.length === 0) { $('#welcome').classList.remove('hidden'); return; }
  $('#welcome').classList.add('hidden');
  c.messages.forEach(m => appendMsg(m.role, m.content, m.meta));
  scrollBottom();
}

function appendMsg(role, content, meta) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  const av = document.createElement('div');
  av.className = 'msg-avatar';
  av.textContent = role === 'user' ? 'U' : '⚡';
  const body = document.createElement('div');
  body.className = 'msg-body';
  body.innerHTML = mdRender(content);
  div.appendChild(av);
  div.appendChild(body);
  if (meta) {
    const m = document.createElement('div');
    m.className = 'msg-meta';
    m.textContent = meta;
    body.appendChild(m);
  }
  $('#messages').appendChild(div);
  return body;
}

function scrollBottom() { requestAnimationFrame(() => { $('#messages').scrollTop = $('#messages').scrollHeight; }); }

function mdRender(text) {
  if (!text) return '';
  let h = escHtml(text);
  h = h.replace(/```(\w*)\n([\s\S]*?)```/g, (_, l, c) => `<pre><code>${c.trim()}</code></pre>`);
  h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
  h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  h = h.replace(/\*(.+?)\*/g, '<em>$1</em>');
  h = h.split('\n\n').map(p => `<p>${p}</p>`).join('');
  h = h.replace(/\n/g, '<br>');
  return h;
}

function escHtml(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }

async function sendMessage() {
  const text = $('#msg-input').value.trim();
  if (!text || isGenerating) return;
  const model = $('#chat-model-select').value;
  if (!model) { toast('No model selected. Go to Models tab to download one.', 'error'); return; }
  const conv = getConv();
  if (!conv) return;
  if (conv.messages.length === 0) conv.title = text.slice(0, 60) + (text.length > 60 ? '...' : '');
  conv.messages.push({ role: 'user', content: text });
  conv.model = model;
  saveConversations();
  $('#msg-input').value = '';
  autoResize();
  $('#welcome').classList.add('hidden');
  appendMsg('user', text);
  scrollBottom();

  isGenerating = true;
  $('#send-btn').classList.add('hidden');
  $('#stop-btn').classList.remove('hidden');
  const aiBody = appendMsg('assistant', '');
  const cursor = document.createElement('span');
  cursor.className = 'cursor-blink';
  aiBody.appendChild(cursor);
  let full = '', promptTk = 0, evalTk = 0;
  const t0 = Date.now();
  abortController = new AbortController();

  try {
    const res = await fetch(`${API}/api/chat`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, messages: conv.messages.map(m => ({ role: m.role, content: m.content })), stream: true }),
      signal: abortController.signal,
    });
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop() || '';
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const chunk = JSON.parse(line);
          if (chunk.message && chunk.message.content) {
            full += chunk.message.content;
            aiBody.innerHTML = mdRender(full);
            aiBody.appendChild(cursor);
            scrollBottom();
          }
          if (chunk.done) { promptTk = chunk.prompt_eval_count || 0; evalTk = chunk.eval_count || 0; }
        } catch { /* skip */ }
      }
    }
  } catch (e) {
    if (e.name === 'AbortError') full += '\n\n*[Stopped]*';
    else { full = `Error: ${e.message}`; toast(e.message, 'error'); }
  }

  cursor.remove();
  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  const tps = evalTk > 0 ? (evalTk / parseFloat(elapsed)).toFixed(1) : '?';
  const meta = `${evalTk} tokens · ${elapsed}s · ${tps} t/s`;
  aiBody.innerHTML = mdRender(full);
  const metaEl = document.createElement('div');
  metaEl.className = 'msg-meta';
  metaEl.textContent = meta;
  aiBody.appendChild(metaEl);
  conv.messages.push({ role: 'assistant', content: full, meta });
  saveConversations();
  $('#token-info').textContent = meta;
  isGenerating = false;
  $('#send-btn').classList.remove('hidden');
  $('#stop-btn').classList.add('hidden');
  abortController = null;
  scrollBottom();
}

function stopGen() { if (abortController) abortController.abort(); }

function autoResize() {
  const el = $('#msg-input');
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 180) + 'px';
}

$('#msg-input').addEventListener('input', autoResize);
$('#msg-input').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
$('#send-btn').addEventListener('click', sendMessage);
$('#stop-btn').addEventListener('click', stopGen);
$('#new-chat-btn').addEventListener('click', newChat);
$('#chat-model-select').addEventListener('change', async (e) => {
  const newModel = e.target.value;
  if (!newModel) return;
  const conv = getConv();
  if (conv) conv.model = newModel;
  try {
    await fetch(`${API}/api/models/unload`, { method: 'POST' });
    await fetch(`${API}/api/models/load`, { 
      method: 'POST', 
      headers: { 'Content-Type': 'application/json' }, 
      body: JSON.stringify({ name: newModel }) 
    });
    toast(`Switched to ${newModel}`, 'success');
    refreshStatus();
  } catch (e) {
    toast(`Failed to switch model: ${e.message}`, 'error');
  }
});

// ═══════════════════════════════════════════════════════════════════════════════
// MODELS TAB
// ═══════════════════════════════════════════════════════════════════════════════

// ─── Hardware Compatibility ─────────────────────────────────────────────────

let hwInfo = null;

async function loadHardwareInfo() {
  try {
    const res = await fetch(`${API}/api/hardware-info`);
    hwInfo = await res.json();
    renderHardwareSummary();
    renderCompatTable();
  } catch (e) {
    $('#hw-compat-summary').innerHTML = '<div class="empty-state">Could not detect hardware</div>';
  }
}

function fitBadge(compat) {
  if (!compat) return '';
  const map = {
    gpu:   '<span class="compat-badge good">Fits in GPU</span>',
    mixed: '<span class="compat-badge warn">GPU+CPU mix</span>',
    cpu:   '<span class="compat-badge bad">CPU only</span>',
    no:    '<span class="compat-badge bad">Won\'t fit</span>',
  };
  return map[compat.fits] || '';
}

function fitIcon(fits) {
  return { gpu: '\u2705', mixed: '\u26a0\ufe0f', cpu: '\u26a0\ufe0f', no: '\u274c' }[fits] || '';
}

function renderHardwareSummary() {
  if (!hwInfo) return;
  const gpu = hwInfo.gpu || {};
  const cpu = hwInfo.cpu || {};
  const ram = hwInfo.ram || {};
  const tier = hwInfo.tier || {};
  const vram = (gpu.vram_total_mb / 1024).toFixed(1);
  const tbl = hwInfo.compatibility_table || [];

  // Build quick summary lines
  const lines = [];
  const sizes = [{ b: 3, label: '1-3B' }, { b: 8, label: '7-8B' }, { b: 27, label: '27B' }, { b: 70, label: '70B' }];
  for (const s of sizes) {
    const row = tbl.find(r => r.params_b >= s.b) || tbl[tbl.length - 1];
    if (!row) continue;
    const q4 = row.quants['Q4_K_M'];
    const q2 = row.quants['Q2_K'];
    const best = q4 || q2;
    if (!best) continue;
    const icon = fitIcon(q4 ? q4.fits : q2.fits);
    const quants = [];
    for (const [qn, qv] of Object.entries(row.quants)) {
      if (qv.fits === 'gpu') quants.push(qn);
    }
    const qStr = quants.length > 0 ? `Full GPU: ${quants.join(', ')}` :
                 (q4 && q4.fits === 'mixed' ? 'GPU+CPU (Q4)' :
                 (q2 && q2.fits !== 'no' ? `${q2.fits === 'cpu' ? 'CPU only' : 'GPU+CPU'} (Q2)` : 'Won\'t fit'));
    lines.push(`<div class="hw-line">${icon} <strong>${s.label} models:</strong> ${qStr}</div>`);
  }

  const vendorBadge = gpu.vendor ? `<span class="mc-badge ${gpu.vendor}">${gpu.vendor.toUpperCase()}</span>` : '';
  const unifiedBadge = gpu.unified_memory ? '<span class="mc-badge moe">Unified Memory</span>' : '';
  const tierBadge = tier.tier ? `<span class="mc-badge tier-${tier.tier}">${tier.tier.toUpperCase()}</span>` : '';
  const cpuModel = cpu.model ? escHtml(cpu.model) : 'Unknown CPU';
  const cpuFeats = [];
  if (cpu.avx512) cpuFeats.push('AVX-512');
  else if (cpu.avx2) cpuFeats.push('AVX2');
  else if (cpu.avx) cpuFeats.push('AVX');
  const cpuFeatStr = cpuFeats.length ? ` (${cpuFeats.join(', ')})` : '';

  $('#hw-compat-summary').innerHTML = `
    <div class="hw-info-grid">
      <div class="hw-specs">
        <div class="hw-spec"><strong>GPU</strong> ${escHtml(gpu.name || 'None')} (${vram} GB${gpu.unified_memory ? ' Unified' : ' VRAM'}) ${vendorBadge}${unifiedBadge}</div>
        <div class="hw-spec"><strong>CPU</strong> ${cpuModel}${cpuFeatStr} (${cpu.cores_physical || '?'}c/${cpu.cores_logical || '?'}t)</div>
        <div class="hw-spec"><strong>RAM</strong> ${ram.total_gb || '?'} GB ${tierBadge}</div>
        ${tier.description ? `<div class="hw-spec hw-tier-desc">${escHtml(tier.description)}</div>` : ''}
      </div>
      <div class="hw-fits">${lines.join('')}</div>
    </div>`;

  loadRecommendations();
}

function renderCompatTable() {
  if (!hwInfo || !hwInfo.compatibility_table) return;
  const quants = ['Q8_0', 'Q6_K', 'Q4_K_M', 'Q3_K_M', 'Q2_K', 'IQ2_XXS'];
  let html = '<table><thead><tr><th>Model</th>';
  quants.forEach(q => { html += `<th>${q}</th>`; });
  html += '</tr></thead><tbody>';
  for (const row of hwInfo.compatibility_table) {
    html += `<tr><td><strong>${row.params_b}B</strong></td>`;
    for (const q of quants) {
      const c = row.quants[q];
      if (!c) { html += '<td>-</td>'; continue; }
      const cls = { gpu: 'ct-gpu', mixed: 'ct-mix', cpu: 'ct-cpu', no: 'ct-no' }[c.fits] || '';
      const label = { gpu: `\u2705 ${c.model_size_gb}G`, mixed: `\u26a0\ufe0f ${c.model_size_gb}G`,
                       cpu: `\u26a0\ufe0f CPU`, no: '\u274c' }[c.fits] || '-';
      html += `<td class="${cls}">${label}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  $('#hw-compat-table').innerHTML = html;
}

// ─── Model Recommendations ──────────────────────────────────────────────────

async function loadRecommendations() {
  const el = $('#recommendations-grid');
  if (!el) return;
  try {
    const res = await fetch(`${API}/api/recommendations`);
    const data = await res.json();
    el.innerHTML = '';
    const hw = data.hardware || {};
    const tip = data.tip || '';

    // Tip banner
    if (tip) {
      const tipEl = document.createElement('div');
      tipEl.className = 'rec-tip';
      tipEl.innerHTML = `<strong>Tip:</strong> ${escHtml(tip)}`;
      el.appendChild(tipEl);
    }

    // MoE filter toggle
    const filterEl = document.createElement('div');
    filterEl.className = 'rec-filter';
    filterEl.innerHTML = `<label><input type="checkbox" id="rec-moe-filter" onchange="filterRecommendations()"> Show only MoE models</label>`;
    el.appendChild(filterEl);

    (data.recommendations || []).forEach(r => {
      const card = document.createElement('div');
      card.className = `rec-card${r.is_moe ? ' is-moe' : ' is-dense'}`;
      const badges = [];
      if (r.is_moe) badges.push('<span class="mc-badge moe">MoE</span>');
      else badges.push('<span class="mc-badge dense">Dense</span>');
      const fitBadgeHtml = r.fit ? fitBadge(r.fit) : '';
      const qualityInfo = r.quality && r.quality.label ? `<span class="rec-quality">${r.quality.label}</span>` : '';
      const tpsStr = r.estimated_tps ? `~${r.estimated_tps} tok/s` : '';
      const moeNote = r.moe_advantage ? `<div class="rec-moe-note">${escHtml(r.moe_advantage)}</div>` : '';

      card.innerHTML = `
        <div class="rec-header">
          <div class="rec-name">${escHtml(r.name)}</div>
          <div class="rec-badges">${badges.join('')}${fitBadgeHtml}</div>
        </div>
        <div class="rec-desc">${escHtml(r.description)}</div>
        ${moeNote}
        <div class="rec-details">
          <span><strong>${r.params_b}B</strong> total${r.is_moe ? ` / <strong>${r.active_params_b}B</strong> active` : ''}</span>
          <span>Quant: <strong>${r.recommended_quant}</strong></span>
          ${tpsStr ? `<span>${tpsStr}</span>` : ''}
          ${qualityInfo}
        </div>
        <div class="rec-actions">
          <button class="btn-sm accent" onclick="browseHFFiles('${escHtml(r.repo)}')">Browse Files</button>
        </div>`;
      el.appendChild(card);
    });

    if (!data.recommendations || data.recommendations.length === 0) {
      el.innerHTML += '<div class="empty-state">No recommendations available for your hardware.</div>';
    }
  } catch (e) {
    el.innerHTML = '<div class="empty-state">Could not load recommendations</div>';
  }
}

function filterRecommendations() {
  const moeOnly = $('#rec-moe-filter') && $('#rec-moe-filter').checked;
  $$('.rec-card').forEach(card => {
    if (moeOnly && card.classList.contains('is-dense')) {
      card.style.display = 'none';
    } else {
      card.style.display = '';
    }
  });
}

$('#toggle-compat-table').addEventListener('click', () => {
  const wrap = $('#hw-compat-table-wrap');
  const btn = $('#toggle-compat-table');
  wrap.classList.toggle('hidden');
  btn.textContent = wrap.classList.contains('hidden') ? 'Show Detailed Table' : 'Hide Table';
});

async function refreshModelsPage() {
  loadHardwareInfo();
  loadQuantTypes();
  try {
    const res = await fetch(`${API}/api/models/list`);
    const data = await res.json();
    populateRequantSource(data.models || []);
    const grid = $('#local-models-grid');
    grid.innerHTML = '';
    if (!data.models || data.models.length === 0) {
      grid.innerHTML = '<div class="empty-state">No models found. Search and download below.</div>';
      return;
    }
    data.models.forEach(m => {
      const card = document.createElement('div');
      card.className = `model-card${m.is_loaded ? ' loaded' : ''}${m.is_default ? ' default' : ''}`;
      const badges = [];
      if (m.is_loaded) badges.push('<span class="mc-badge loaded">Loaded</span>');
      if (m.is_default) badges.push('<span class="mc-badge default">Default</span>');
      if (m.is_moe) badges.push('<span class="mc-badge moe">MoE</span>');
      const paramInfo = m.is_moe && m.active_params_b
        ? `${m.details.parameter_size} (${m.active_params_b}B active)`
        : m.details.parameter_size;
      card.innerHTML = `
        <div class="mc-header">
          <div class="mc-name">${escHtml(m.name)}</div>
          <div class="mc-badges">${badges.join('')}</div>
        </div>
        <div class="mc-details">
          <span>${m.details.quantization_level}</span>
          <span>${paramInfo}</span>
          <span>${fmtSize(m.size)}</span>
          <span>${m.details.family}</span>
        </div>
        <div class="mc-actions">
          ${!m.is_loaded ? `<button onclick="loadModel('${escHtml(m.name)}')">Load</button>` : `<button onclick="unloadModel()">Unload</button>`}
          ${!m.is_default ? `<button onclick="setDefault('${escHtml(m.name)}')">Set Default</button>` : ''}
          <button class="danger-text" onclick="deleteModel('${escHtml(m.name)}')">Delete</button>
        </div>`;
      grid.appendChild(card);
    });
  } catch (e) { toast('Failed to load models', 'error'); }
}

async function loadModel(name) {
  toast(`Loading ${name}...`, 'info');
  try {
    await fetch(`${API}/api/models/load`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) });
    toast(`${name} loaded!`, 'success');
    refreshModelsPage();
    refreshChatModels();
    refreshStatus();
  } catch (e) { toast(`Failed to load: ${e.message}`, 'error'); }
}

async function unloadModel() {
  try {
    await fetch(`${API}/api/models/unload`, { method: 'POST' });
    toast('Model unloaded', 'info');
    refreshModelsPage();
    refreshStatus();
  } catch (e) { toast('Failed to unload', 'error'); }
}

async function setDefault(name) {
  try {
    await fetch(`${API}/api/models/set-default`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) });
    toast(`${name} set as default`, 'success');
    refreshModelsPage();
  } catch (e) { toast('Failed', 'error'); }
}

async function deleteModel(name) {
  if (!confirm(`Delete ${name}? This cannot be undone.`)) return;
  try {
    await fetch(`${API}/api/models/delete`, { method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) });
    toast(`${name} deleted`, 'success');
    refreshModelsPage();
    refreshChatModels();
  } catch (e) { toast('Failed to delete', 'error'); }
}

$('#save-settings-btn').addEventListener('click', async () => {
  const gpu_layers = parseInt($('#setting-gpu-layers').value);
  const ctx_size = parseInt($('#setting-ctx-size').value);
  try {
    await fetch(`${API}/api/models/update-settings`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ gpu_layers, ctx_size }) });
    toast('Settings saved', 'success');
  } catch (e) { toast('Failed to save', 'error'); }
});

$('#refresh-models-btn').addEventListener('click', () => { refreshModelsPage(); refreshChatModels(); });

// ─── HuggingFace Search ───────────────────────────────────────────────────────

$('#hf-search-btn').addEventListener('click', searchHF);
$('#hf-search').addEventListener('keydown', e => { if (e.key === 'Enter') searchHF(); });

async function searchHF() {
  const q = $('#hf-search').value.trim();
  if (!q) return;
  const grid = $('#hf-results');
  grid.innerHTML = '<div class="empty-state">Searching HuggingFace...</div>';
  $('#hf-files-panel').classList.add('hidden');
  try {
    const res = await fetch(`${API}/api/models/search-hf?q=${encodeURIComponent(q)}&limit=20`);
    const data = await res.json();
    grid.innerHTML = '';
    if (!data.results || data.results.length === 0) {
      grid.innerHTML = '<div class="empty-state">No GGUF models found. Try different search terms.</div>';
      return;
    }
    data.results.forEach(r => {
      const card = document.createElement('div');
      card.className = 'hf-card';
      card.onclick = () => browseHFFiles(r.id);
      const tags = (r.tags || []).map(t => `<span class="hf-tag">${escHtml(t)}</span>`).join('');
      const paramsLine = r.params_b ? `<span class="hf-params-badge">${r.params_b}B${r.is_moe && r.active_params_b ? ` (${r.active_params_b}B active)` : ''}</span>` : '';
      const moeBadge = r.is_moe ? '<span class="mc-badge moe">MoE</span>' : '';
      const sizeLine = r.estimated_size_gb ? `<span class="hf-size-est">~${r.estimated_size_gb} GB (Q4)</span>` : '';
      const compatLine = r.compatibility ? fitBadge(r.compatibility) : '';
      card.innerHTML = `
        <div class="hf-name">${escHtml(r.id)}</div>
        ${(paramsLine || sizeLine || compatLine || moeBadge) ? `<div class="hf-compat-row">${moeBadge}${paramsLine}${sizeLine}${compatLine}</div>` : ''}
        <div class="hf-meta">
          <span>Downloads: ${fmtNum(r.downloads)}</span>
          <span>Likes: ${r.likes}</span>
        </div>
        ${tags ? `<div class="hf-tags">${tags}</div>` : ''}`;
      grid.appendChild(card);
    });
  } catch (e) { grid.innerHTML = `<div class="empty-state">Error: ${e.message}</div>`; }
}

async function browseHFFiles(repo) {
  const panel = $('#hf-files-panel');
  panel.classList.remove('hidden');
  panel.innerHTML = `<h4>Loading files from ${escHtml(repo)}...</h4>`;
  try {
    const res = await fetch(`${API}/api/models/hf-files?repo=${encodeURIComponent(repo)}`);
    const data = await res.json();
    if (!data.files || data.files.length === 0) {
      panel.innerHTML = '<h4>No GGUF files found in this repo.</h4>';
      return;
    }
    const paramsStr = data.params_b ? ` — ${data.params_b}B params` : '';
    let html = `<h4>${escHtml(repo)}${paramsStr}</h4>`;
    data.files.forEach(f => {
      const badge = f.compatibility ? fitBadge(f.compatibility) : '';
      const rec = f.recommended ? '<span class="compat-rec">Recommended</span>' : '';
      const quantTag = f.quant ? `<span class="hf-quant-tag">${f.quant}</span>` : '';
      html += `<div class="hf-file-row${f.recommended ? ' recommended' : ''}">
        <div class="hf-file-info">
          <span class="hf-file-name">${escHtml(f.filename)}</span>
          <span class="hf-file-details">${fmtSize(f.size)} ${quantTag} ${badge} ${rec}</span>
        </div>
        <button class="btn-sm accent" onclick="startDownload('${escHtml(f.url)}','${escHtml(f.filename)}')">Download</button>
      </div>`;
    });
    panel.innerHTML = html;
  } catch (e) { panel.innerHTML = `<h4>Error: ${e.message}</h4>`; }
}

async function startDownload(url, filename) {
  try {
    const res = await fetch(`${API}/api/models/download`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, filename }),
    });
    const data = await res.json();
    if (data.status === 'exists') { toast('Model already downloaded!', 'info'); refreshModelsPage(); return; }
    toast(`Download started: ${filename}`, 'success');
    $('#downloads-section').classList.remove('hidden');
  } catch (e) { toast(`Download failed: ${e.message}`, 'error'); }
}

async function pollDownloads() {
  try {
    const res = await fetch(`${API}/api/models/downloads`);
    const data = await res.json();
    const dls = data.downloads || {};
    const keys = Object.keys(dls);
    if (keys.length === 0) { $('#downloads-section').classList.add('hidden'); return; }
    $('#downloads-section').classList.remove('hidden');
    const list = $('#downloads-list');
    list.innerHTML = '';
    keys.forEach(id => {
      const d = dls[id];
      const pct = d.total > 0 ? Math.round(d.progress / d.total * 100) : 0;
      const el = document.createElement('div');
      el.className = 'dl-item';
      el.innerHTML = `
        <span class="dl-name">${escHtml(d.filename)}</span>
        <div class="dl-bar"><div class="dl-fill" style="width:${pct}%"></div></div>
        <span class="dl-pct">${d.status === 'complete' ? 'Done' : d.status === 'error' ? 'Error' : `${pct}%`}</span>
        <span class="dl-status ${d.status}">${d.status}</span>`;
      list.appendChild(el);
    });
    // Refresh models list if any completed
    const anyComplete = keys.some(k => dls[k].status === 'complete');
    if (anyComplete) { refreshChatModels(); refreshModelsPage(); }
  } catch { /* ignore */ }
}

// ─── Requantization ──────────────────────────────────────────────────────────

let quantTypes = [];

async function loadQuantTypes() {
  try {
    const res = await fetch(`${API}/api/models/quant-types`);
    const data = await res.json();
    quantTypes = data.types || [];
    const sel = $('#requant-target');
    sel.innerHTML = '';
    quantTypes.filter(t => t.safe_requant).forEach(t => {
      const opt = document.createElement('option');
      opt.value = t.type;
      opt.textContent = `${t.type} — ${t.label} (${t.quality_pct}% quality, ${t.bpw} bpw)`;
      sel.appendChild(opt);
    });
    sel.value = 'Q2_K';
    updateRequantInfo();
  } catch (e) { console.error('Failed to load quant types', e); }
}

function populateRequantSource(models) {
  const sel = $('#requant-source');
  sel.innerHTML = '';
  (models || []).forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.name;
    opt.textContent = `${m.name} (${fmtSize(m.size)})`;
    sel.appendChild(opt);
  });
}

function updateRequantInfo() {
  const target = $('#requant-target').value;
  const info = quantTypes.find(t => t.type === target);
  const el = $('#requant-quality-info');
  if (!info) { el.innerHTML = ''; return; }
  const tierColors = { excellent: '#22c55e', great: '#3b82f6', good: '#eab308', fair: '#f97316', poor: '#ef4444' };
  const color = tierColors[info.tier] || '#888';
  const sourceSel = $('#requant-source');
  const sourceOpt = sourceSel.options[sourceSel.selectedIndex];
  let sizeEst = '';
  if (sourceOpt) {
    const sizeMatch = sourceOpt.textContent.match(/\(([\d.]+)\s*(GB|MB)/i);
    if (sizeMatch) {
      const srcGB = sizeMatch[2] === 'MB' ? parseFloat(sizeMatch[1]) / 1024 : parseFloat(sizeMatch[1]);
      const ratio = info.bpp / 0.56; // Ratio vs Q4_K_M baseline
      sizeEst = ` — Est. output: ~${(srcGB * ratio).toFixed(1)} GB`;
    }
  }
  el.innerHTML = `
    <div class="rq-info-row">
      <span class="rq-tier" style="color:${color}">${info.tier.toUpperCase()}</span>
      <span class="rq-quality">Quality: ${info.quality_pct}%</span>
      <span class="rq-bpw">${info.bpw} bits/weight</span>
      <span class="rq-speed">${info.speed_mult}x speed vs Q8</span>
      <span class="rq-warn">${info.quality_pct < 80 ? 'Warning: noticeable quality loss' : ''}</span>
    </div>
    <div class="rq-desc">${info.label}${sizeEst}</div>`;
}

$('#requant-target').addEventListener('change', updateRequantInfo);
$('#requant-source').addEventListener('change', updateRequantInfo);

let activeQuantId = null;

$('#requant-btn').addEventListener('click', async () => {
  const source = $('#requant-source').value;
  const targetType = $('#requant-target').value;
  if (!source) { toast('Select a source model', 'error'); return; }
  if (!confirm(`Requantize ${source} to ${targetType}?\nThis may take several minutes.`)) return;
  try {
    const res = await fetch(`${API}/api/models/quantize`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source, target_type: targetType }),
    });
    const data = await res.json();
    if (data.status === 'exists') { toast('Model already exists!', 'info'); refreshModelsPage(); return; }
    activeQuantId = data.quant_id;
    toast(`Requantization started: ${data.output}`, 'success');
    $('#requant-progress-wrap').classList.remove('hidden');
    $('#requant-btn').disabled = true;
    pollQuantProgress();
  } catch (e) { toast(`Requantization failed: ${e.message}`, 'error'); }
});

async function pollQuantProgress() {
  if (!activeQuantId) return;
  try {
    const res = await fetch(`${API}/api/models/quantize-status/${activeQuantId}`);
    const data = await res.json();
    const pct = data.progress || 0;
    $('#requant-fill').style.width = `${pct}%`;
    $('#requant-pct').textContent = `${pct}%`;
    $('#requant-status').textContent = data.status === 'quantizing' ? 'Converting...' :
      data.status === 'complete' ? 'Done!' : data.status === 'error' ? `Error: ${data.error}` : data.status;

    if (data.status === 'complete') {
      toast('Requantization complete!', 'success');
      $('#requant-btn').disabled = false;
      activeQuantId = null;
      refreshModelsPage();
      return;
    }
    if (data.status === 'error') {
      toast(`Requantization error: ${data.error}`, 'error');
      $('#requant-btn').disabled = false;
      activeQuantId = null;
      return;
    }
    setTimeout(pollQuantProgress, 2000);
  } catch { setTimeout(pollQuantProgress, 3000); }
}

// ─── Smart Search ──────────────────────────────────────────────────────────

$('#hf-smart-search-btn').addEventListener('click', smartSearchHF);

async function smartSearchHF() {
  const q = $('#hf-search').value.trim();
  const grid = $('#hf-results');
  grid.innerHTML = '<div class="empty-state">Finding models optimized for your hardware...</div>';
  $('#hf-files-panel').classList.add('hidden');
  try {
    const res = await fetch(`${API}/api/models/smart-search?q=${encodeURIComponent(q)}&size=auto`);
    const data = await res.json();
    grid.innerHTML = '';
    if (!data.results || data.results.length === 0) {
      grid.innerHTML = '<div class="empty-state">No compatible models found.</div>';
      return;
    }
    const hw = data.hardware || {};
    grid.innerHTML = `<div class="smart-search-header">
      <span>Hardware: ${hw.vram_mb ? Math.round(hw.vram_mb/1024)+'GB VRAM' : '?'}, ${hw.ram_mb ? Math.round(hw.ram_mb/1024)+'GB RAM' : '?'}</span>
      <span>Max GPU: ${hw.max_gpu_params || '?'}B | Max Mixed: ${hw.max_mixed_params || '?'}B</span>
    </div>`;
    data.results.forEach(r => {
      const card = document.createElement('div');
      const recCls = { perfect: 'rec-perfect', good: 'rec-good', usable: 'rec-usable', none: 'rec-none' }[r.recommendation] || '';
      card.className = `hf-card ${recCls}`;
      card.onclick = () => browseHFFiles(r.id);
      const recBadge = { perfect: '<span class="rec-badge perfect">Perfect fit</span>',
        good: '<span class="rec-badge good">Good fit</span>',
        usable: '<span class="rec-badge usable">Usable (mixed)</span>',
        none: '<span class="rec-badge none">Too large</span>' }[r.recommendation] || '';
      const paramsLine = r.params_b ? `<span class="hf-params-badge">${r.params_b}B${r.is_moe && r.active_params_b ? ` (${r.active_params_b}B active)` : ''}</span>` : '';
      const moeBadge = r.is_moe ? '<span class="mc-badge moe">MoE</span>' : '';
      const bestQ = r.best_quant ? `<span class="hf-quant-tag">${r.best_quant}</span>` : '';
      const fitInfo = r.best_fit ? `<span class="hf-size-est">~${r.best_fit.model_size_gb}GB</span>` : '';
      card.innerHTML = `
        <div class="hf-name">${escHtml(r.id)}</div>
        <div class="hf-compat-row">${moeBadge}${recBadge}${paramsLine}${bestQ}${fitInfo}</div>
        <div class="hf-meta">
          <span>Downloads: ${fmtNum(r.downloads)}</span>
          <span>Likes: ${r.likes}</span>
        </div>`;
      grid.appendChild(card);
    });
  } catch (e) { grid.innerHTML = `<div class="empty-state">Error: ${e.message}</div>`; }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BENCHMARK TAB
// ═══════════════════════════════════════════════════════════════════════════════

async function refreshBenchPage() {
  await refreshChatModels();
  await loadBenchHistory();
}

$('#run-bench-btn').addEventListener('click', runBenchmark);
$('#clear-bench-btn').addEventListener('click', async () => {
  if (!confirm('Clear all benchmark history?')) return;
  await fetch(`${API}/api/bench/clear`, { method: 'DELETE' });
  loadBenchHistory();
  toast('Benchmarks cleared', 'info');
});

async function runBenchmark() {
  const model = $('#bench-model-select').value;
  if (!model) { toast('Select a model first', 'error'); return; }
  const prompt = $('#bench-prompt').value.trim() || 'Hello';
  const max_tokens = parseInt($('#bench-max-tokens').value) || 128;
  const runs = parseInt($('#bench-runs').value) || 3;

  const btn = $('#run-bench-btn');
  btn.textContent = 'Running...';
  btn.disabled = true;
  toast(`Benchmarking ${model} (${runs} runs)...`, 'info');

  try {
    const res = await fetch(`${API}/api/bench/run`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, prompt, max_tokens, runs }),
    });
    const data = await res.json();
    renderBenchResult(data, true);
    await loadBenchHistory();
    toast(`Benchmark complete: ${data.average.tokens_per_second} t/s avg`, 'success');
  } catch (e) { toast(`Benchmark failed: ${e.message}`, 'error'); }
  btn.textContent = 'Run Benchmark';
  btn.disabled = false;
}

function renderBenchResult(d, live = false) {
  const container = live ? '#bench-live-content' : null;
  const html = `
    <div class="bench-result-card">
      <div class="bench-header">
        <span class="bench-model-name">${escHtml(d.model)} (${d.quant}, ${d.params})</span>
        <span class="bench-date">${d.timestamp || ''}</span>
      </div>
      <div class="bench-metrics">
        <div class="bench-metric"><div class="value">${d.average.tokens_per_second}</div><div class="label">tokens/sec avg</div></div>
        <div class="bench-metric"><div class="value">${d.average.elapsed_s}s</div><div class="label">avg latency</div></div>
        <div class="bench-metric"><div class="value">${d.runs}</div><div class="label">runs</div></div>
        ${d.gpu && d.gpu.name ? `<div class="bench-metric"><div class="value" style="font-size:14px">${d.gpu.name}</div><div class="label">GPU</div></div>` : ''}
      </div>
      <table class="bench-runs-table">
        <thead><tr><th>Run</th><th>Time</th><th>Prompt Tk</th><th>Gen Tk</th><th>t/s</th></tr></thead>
        <tbody>${d.results.map(r => `<tr><td>#${r.run}</td><td>${r.elapsed_s}s</td><td>${r.prompt_tokens}</td><td>${r.completion_tokens}</td><td>${r.tokens_per_second}</td></tr>`).join('')}</tbody>
      </table>
    </div>`;

  if (live) {
    $('#bench-live').classList.remove('hidden');
    $('#bench-live-content').innerHTML = html;
  }
  return html;
}

async function loadBenchHistory() {
  try {
    const res = await fetch(`${API}/api/bench/history`);
    const data = await res.json();
    const el = $('#bench-history');
    if (!data.benchmarks || data.benchmarks.length === 0) {
      el.innerHTML = '<div class="empty-state">No benchmarks yet. Run one above.</div>';
      return;
    }
    el.innerHTML = data.benchmarks.map(b => renderBenchResult(b)).join('');
  } catch { /* ignore */ }
}

// ─── Helpers ───────────────────────────────────────────────────────────────────

function fmtSize(b) {
  if (!b) return '?';
  if (b >= 1073741824) return `${(b / 1073741824).toFixed(1)} GB`;
  if (b >= 1048576) return `${(b / 1048576).toFixed(0)} MB`;
  return `${(b / 1024).toFixed(0)} KB`;
}

function fmtNum(n) {
  if (!n) return '0';
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
  return String(n);
}

// ─── Start ─────────────────────────────────────────────────────────────────────

init();
