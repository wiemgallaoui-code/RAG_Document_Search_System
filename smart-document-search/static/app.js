/**
 * RAG Document Assistant — frontend chat UI
 * Handles message rendering, markdown-lite formatting, and API calls.
 */

const chat = document.getElementById("chat");
const welcome = document.getElementById("welcome");
const form = document.getElementById("search-form");
const queryInput = document.getElementById("query-input");
const submitBtn = document.getElementById("submit-btn");
const sendIcon = submitBtn.querySelector(".send-icon");
const spinner = submitBtn.querySelector(".spinner");

const docCountEl = document.getElementById("doc-count");
const retrievalEl = document.getElementById("retrieval-method");
const providerEl = document.getElementById("llm-provider");

/** Load header stats from the backend. */
async function loadStats() {
  try {
    const res = await fetch("/api/stats");
    if (!res.ok) return;
    const data = await res.json();
    docCountEl.textContent = `${data.document_count} Documents Indexed`;
    retrievalEl.textContent = `${data.retrieval_method} Retrieval`;
    providerEl.textContent = `Provider: ${data.llm_provider}`;
  } catch {
    /* keep defaults */
  }
}

function hideWelcome() {
  welcome?.remove();
}

function scrollToBottom() {
  chat.scrollTop = chat.scrollHeight;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Convert plain-text LLM output into HTML with paragraphs and lists.
 * Supports: blank-line paragraphs, "- item" bullets, "1. item" numbered lists.
 */
function formatAnswer(text) {
  const blocks = text.trim().split(/\n{2,}/);
  const html = blocks
    .map((block) => {
      const lines = block.split("\n");
      const isBullet = lines.every((l) => /^\s*[-*•]\s/.test(l) || l.trim() === "");
      const isNumbered = lines.every((l) => /^\s*\d+[.)]\s/.test(l) || l.trim() === "");

      if (isBullet && lines.some((l) => l.trim())) {
        const items = lines
          .filter((l) => l.trim())
          .map((l) => `<li>${escapeHtml(l.replace(/^\s*[-*•]\s+/, ""))}</li>`)
          .join("");
        return `<ul>${items}</ul>`;
      }

      if (isNumbered && lines.some((l) => l.trim())) {
        const items = lines
          .filter((l) => l.trim())
          .map((l) => `<li>${escapeHtml(l.replace(/^\s*\d+[.)]\s+/, ""))}</li>`)
          .join("");
        return `<ol>${items}</ol>`;
      }

      return `<p>${lines.map(escapeHtml).join("<br>")}</p>`;
    })
    .join("");

  return `<div class="answer-content">${html}</div>`;
}

function addUserMessage(text) {
  hideWelcome();
  const el = document.createElement("div");
  el.className = "message user";
  el.innerHTML = `
    <div class="msg-avatar">You</div>
    <div class="message-body">
      <div class="bubble">${escapeHtml(text)}</div>
    </div>
  `;
  chat.appendChild(el);
  scrollToBottom();
}

function addTypingIndicator() {
  const el = document.createElement("div");
  el.className = "message assistant typing";
  el.id = "typing";
  el.innerHTML = `
    <div class="msg-avatar">AI</div>
    <div class="message-body">
      <div class="bubble">
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="dot"></span>
      </div>
    </div>
  `;
  chat.appendChild(el);
  scrollToBottom();
}

function removeTypingIndicator() {
  document.getElementById("typing")?.remove();
}

function renderSources(sources) {
  if (!sources?.length) return "";

  const cards = sources
    .map(
      (s) => `
      <div class="source-card" title="${escapeHtml(s.chunk_id || s.document)} — score ${s.similarity_score}">
        <div class="source-info">
          <span class="source-icon">📄</span>
          <div class="source-meta">
            <span class="source-name">${escapeHtml(s.document)}</span>
            ${s.chunk_id ? `<span class="source-chunk">${escapeHtml(s.chunk_id)}</span>` : ""}
          </div>
        </div>
        <div class="source-score-wrap">
          <span class="source-score">${s.similarity_score.toFixed(4)}</span>
          <span class="source-score-label">Score</span>
        </div>
      </div>`
    )
    .join("");

  return `
    <div class="sources-block">
      <div class="sources-label">Sources used</div>
      <div class="sources-grid">${cards}</div>
    </div>`;
}

function addAssistantMessage(answer, sources) {
  const el = document.createElement("div");
  el.className = "message assistant";
  el.innerHTML = `
    <div class="msg-avatar">AI</div>
    <div class="message-body">
      <div class="bubble">${formatAnswer(answer)}</div>
      ${renderSources(sources)}
    </div>
  `;
  chat.appendChild(el);
  scrollToBottom();
}

function addErrorMessage(text) {
  const el = document.createElement("div");
  el.className = "message assistant error";
  el.innerHTML = `
    <div class="msg-avatar">!</div>
    <div class="message-body">
      <div class="bubble">${escapeHtml(text)}</div>
    </div>
  `;
  chat.appendChild(el);
  scrollToBottom();
}

function setLoading(loading) {
  submitBtn.disabled = loading;
  queryInput.disabled = loading;
  sendIcon.classList.toggle("hidden", loading);
  spinner.classList.toggle("hidden", !loading);
}

async function ask(query) {
  setLoading(true);
  addUserMessage(query);
  queryInput.value = "";
  addTypingIndicator();

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 3 }),
    });

    removeTypingIndicator();

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();
    addAssistantMessage(data.answer, data.sources);

    // Refresh provider label in header after each response
    if (data.provider) {
      const labels = { groq: "Groq", openai: "OpenAI", ollama: "Ollama", fallback: "Fallback" };
      providerEl.textContent = `Provider: ${labels[data.provider] || data.provider}`;
    }
  } catch (error) {
    removeTypingIndicator();
    addErrorMessage(`Something went wrong: ${error.message}`);
  } finally {
    setLoading(false);
    queryInput.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (query) ask(query);
});

document.querySelectorAll(".suggestion").forEach((btn) => {
  btn.addEventListener("click", () => ask(btn.dataset.query));
});

loadStats();
