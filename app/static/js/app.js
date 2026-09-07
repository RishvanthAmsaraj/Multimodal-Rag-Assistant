/* ============================================================
 * app.js — Multi-Modal RAG Assistant Web UI
 * ============================================================ */

// ===== Theme =====
(function initTheme() {
  const html = document.documentElement;
  const btn = document.getElementById('theme-toggle');
  btn.textContent = html.getAttribute('data-theme') === 'dark' ? 'Light' : 'Dark';
  btn.addEventListener('click', () => {
    const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    btn.textContent = next === 'dark' ? 'Light' : 'Dark';
  });
})();

// ===== Upload zone =====
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');

uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('drag-over');
});
uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) {
    handleFiles(e.dataTransfer.files);
  }
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) {
    handleFiles(fileInput.files);
    fileInput.value = '';
  }
});

async function handleFiles(files) {
  const statusEl = document.getElementById('ingest-status');
  for (const file of files) {
    statusEl.style.display = 'block';
    statusEl.innerHTML = `<div class="progress-bar"><div class="fill" style="width:50%;"></div></div><p style="font-size:12px;color:var(--color-text-muted);margin-top:4px;">Ingesting ${escapeHtml(file.name)}...</p>`;
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch('/api/ingest', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.success) {
        statusEl.innerHTML = `<div class="alert-success">Ingested ${escapeHtml(file.name)} — ${data.chunks_ingested} chunk(s)</div>`;
      } else {
        statusEl.innerHTML = `<div class="alert-error">${escapeHtml(data.error || 'Ingest failed')}</div>`;
      }
    } catch (err) {
      statusEl.innerHTML = `<div class="alert-error">Network error: ${escapeHtml(err.message)}</div>`;
    }
    setTimeout(() => { statusEl.style.display = 'none'; }, 3000);
  }
  await Promise.all([loadSources(), loadStats()]);
}

// ===== Sources =====
async function loadSources() {
  try {
    const res = await fetch('/api/sources');
    const data = await res.json();
    const list = document.getElementById('source-list');
    if (!data.success || !data.sources.length) {
      list.innerHTML = '<div class="source-empty">No documents indexed yet.</div>';
      return;
    }
    list.innerHTML = data.sources.map(s => {
      const icon = pdfIcons[s.type] || 'DOC';
      return `<div class="source-item">
        <div class="source-icon">${icon}</div>
        <span class="source-name">${escapeHtml(s.name)}</span>
        <span class="source-size">${s.size_kb} KB</span>
      </div>`;
    }).join('');
    // Enable input if we have sources
    document.getElementById('question-input').disabled = false;
    document.getElementById('send-btn').disabled = false;
  } catch (err) {
    console.error('Failed to load sources', err);
  }
}

const pdfIcons = {
  '.pdf': 'PDF',
  '.png': 'IMG',
  '.jpg': 'IMG',
  '.jpeg': 'IMG',
  '.txt': 'TXT',
  '.md': 'MD',
};

const EMPTY_STATE_HTML =
  '<div class="empty-state" id="chat-empty">' +
  '<div class="icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none">' +
  '<path d="M7 3.5h7.5L19 8v9.5a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1v-13a1 1 0 0 1 1-1Z" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/>' +
  '<path d="M14.5 3.5V8H19" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/>' +
  '<path d="M9 14.5h6M9 17h4" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>' +
  '</svg></div>' +
  '<p class="empty-title">Your documents, ready for questions</p>' +
  '<p>Add files to the library, then ask anything about their content.</p>' +
  '</div>';

// ===== Stats =====
async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    if (data.success) {
      document.getElementById('stat-chunks').textContent = data.num_chunks;
      document.getElementById('stat-model').textContent = data.embedder_model.split('/').pop() || 'N/A';
    }
  } catch (err) {
    console.error('Failed to load stats', err);
  }
}

async function loadConfig() {
  try {
    const res = await fetch('/api/config');
    const data = await res.json();
    if (data.top_k) {
      document.getElementById('stat-topk').textContent = data.top_k;
    }
  } catch (err) {
    /* ok */
  }
}

// ===== Chat =====
const chatContainer = document.getElementById('chat-container');
const questionInput = document.getElementById('question-input');
const sendBtn = document.getElementById('send-btn');
const emptyState = document.getElementById('chat-empty');

const AVATARS = {
  user: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><circle cx="12" cy="8" r="3.6"/><path d="M5.6 19.4a6.5 6.5 0 0 1 12.8 0"/></svg>',
  assistant: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><path d="M12 2.6l2.1 6.3 6.3 2.1-6.3 2.1L12 19.4l-2.1-6.3-6.3-2.1 6.3-2.1L12 2.6z"/></svg>',
};

async function sendQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage('user', escapeHtml(question));
  questionInput.value = '';
  sendBtn.disabled = true;
  questionInput.disabled = true;

  showLoadingIndicator();

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();

    removeLoadingIndicator();

    if (!data.success) {
      addMessage('assistant', `Error: ${escapeHtml(data.error || 'Query failed')}`);
      return;
    }

    let answer = renderMarkdown(data.answer || 'No answer generated.');
    if (data.sources && data.sources.length > 0) {
      answer += '<div class="source-citation">';
      answer += `<details><summary>${data.num_sources} cited source(s)</summary>`;
      data.sources.forEach(s => {
        const score = typeof s.score === 'number' ? s.score.toFixed(3) : 'N/A';
        const pct = typeof s.score === 'number' ? Math.min(100, Math.max(0, Math.round(s.score * 100))) : 0;
        answer += `<div class="source-block">
          <div class="source-text">${escapeHtml(s.text)}</div>
          <div class="source-meta">
            <span class="score-chip"><span class="score-meter"><span class="score-fill" style="width:${pct}%;"></span></span>${score}</span>
            <span>chunk ${escapeHtml(s.chunk_id || '—')}</span>
          </div>
        </div>`;
      });
      answer += '</details></div>';
    }
    addMessage('assistant', answer);
  } catch (err) {
    removeLoadingIndicator();
    addMessage('assistant', `Network error: ${escapeHtml(err.message)}`);
  }

  sendBtn.disabled = false;
  questionInput.disabled = false;
  questionInput.focus();
}

function addMessage(role, html) {
  emptyState.style.display = 'none';
  const div = document.createElement('div');
  div.className = `chat-message ${role}`;
  div.innerHTML = `<div class="avatar">${AVATARS[role] || ''}</div>
    <div class="bubble">${html}</div>`;
  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function showLoadingIndicator() {
  const div = document.createElement('div');
  div.className = 'chat-message assistant';
  div.id = 'loading-indicator';
  div.innerHTML = `<div class="avatar">${AVATARS.assistant}</div><div class="bubble"><div class="loading-container"><div class="spinner"></div></div></div>`;
  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function removeLoadingIndicator() {
  const el = document.getElementById('loading-indicator');
  if (el) el.remove();
}

/**
 * Tiny markdown renderer (safe: escapes first, then transforms).
 * Supports: **bold**, *italic*, `code`, fenced code blocks, # headings,
 * - / * bullet lists, 1. numbered lists, paragraphs.
 */
function renderMarkdown(text) {
  let h = escapeHtml(text);

  // Fenced code blocks first (before inline rules touch them)
  h = h.replace(/```([\s\S]*?)```/g, (_m, c) => `<pre><code>${c.trim()}</code></pre>`);

  // Headings
  h = h.replace(/^#{1,3}\s+(.*)$/gm, (_m, t) => `<h3>${t}</h3>`);

  // Block-level: walk lines, group bullet/numbered lists
  const lines = h.split('\n');
  const out = [];
  let list = null;
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) { closeList(); out.push(''); continue; }
    const ul = line.match(/^[-*•]\s+(.*)$/);
    const ol = line.match(/^\d+[.)]\s+(.*)$/);
    if (ul) {
      if (list !== 'ul') { closeList(); out.push('<ul>'); list = 'ul'; }
      out.push(`<li>${ul[1]}</li>`);
      continue;
    }
    if (ol) {
      if (list !== 'ol') { closeList(); out.push('<ol>'); list = 'ol'; }
      out.push(`<li>${ol[1]}</li>`);
      continue;
    }
    closeList();
    out.push(line);
  }
  closeList();
  h = out.join('\n');

  // Inline: bold, italic, inline code
  h = h.replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>');
  h = h.replace(/(^|[\s(])\*([^*\n]+)\*(?!\*)/g, '$1<i>$2</i>');
  h = h.replace(/`([^`\n]+)`/g, '<code>$1</code>');

  // Wrap remaining text runs in paragraphs, keep block tags intact
  const parts = h.split('\n');
  const wrapped = [];
  let para = [];
  const flush = () => {
    if (para.length) { wrapped.push(`<p>${para.join('<br>')}</p>`); para = []; }
  };
  for (const p of parts) {
    if (/^<(ul|ol|pre|h\d|p|li)(\s|>)/.test(p)) { flush(); wrapped.push(p); }
    else if (p === '') { flush(); }
    else para.push(p);
  }
  flush();
  return wrapped.join('\n');
}

sendBtn.addEventListener('click', sendQuestion);
questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendQuestion();
});

// ===== Clear =====
document.getElementById('clear-btn').addEventListener('click', async () => {
  if (!confirm('Clear all indexed documents and reset the vector store?')) return;
  try {
    const res = await fetch('/api/clear', { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      chatContainer.innerHTML = EMPTY_STATE_HTML;
      document.getElementById('question-input').disabled = true;
      document.getElementById('send-btn').disabled = true;
      await Promise.all([loadSources(), loadStats()]);
    }
  } catch (err) {
    console.error('Clear failed', err);
  }
});

// ===== Helpers =====
function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// ===== Init =====
Promise.all([loadSources(), loadStats(), loadConfig()]);
