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
      list.innerHTML = '<div style="font-size:13px;color:var(--color-text-muted);">No documents indexed yet.</div>';
      return;
    }
    list.innerHTML = data.sources.map(s => {
      const icon = pdfIcons[s.type] || '📄';
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

async function sendQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage('user', question);
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

    let answer = data.answer || 'No answer generated.';
    if (data.sources && data.sources.length > 0) {
      answer += '<div class="source-citation">';
      answer += `<details><summary>${data.num_sources} source(s)</summary>`;
      data.sources.forEach(s => {
        const score = typeof s.score === 'number' ? s.score.toFixed(3) : 'N/A';
        answer += `<div style="margin-top:8px;">
          <div class="source-text">${escapeHtml(s.text)}</div>
          <div class="source-meta">
            <span>Score: ${score}</span>
            <span>Chunk: ${escapeHtml(s.chunk_id || '—')}</span>
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
  div.innerHTML = `<div class="avatar">${role === 'user' ? 'U' : 'R'}</div>
    <div class="bubble">${html}</div>`;
  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function showLoadingIndicator() {
  const div = document.createElement('div');
  div.className = 'chat-message assistant';
  div.id = 'loading-indicator';
  div.innerHTML = `<div class="avatar">R</div><div class="bubble"><div class="loading-container"><div class="spinner"></div></div></div>`;
  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function removeLoadingIndicator() {
  const el = document.getElementById('loading-indicator');
  if (el) el.remove();
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
      chatContainer.innerHTML = '<div class="empty-state" id="chat-empty"><div class="icon">💬</div><p>Upload documents, then ask questions about their content.</p></div>';
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
