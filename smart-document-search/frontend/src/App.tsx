import { useCallback, useEffect, useRef, useState } from "react";
import { askQuestion, fetchStats } from "./api";
import type { ChatMessage, SourceHit } from "./types";
import { formatAnswer, formatScore } from "./utils/formatAnswer";

const PROVIDER_LABELS: Record<string, string> = {
  groq: "Groq",
  openai: "OpenAI",
  ollama: "Ollama",
  fallback: "Fallback",
};

const SUGGESTIONS = [
  { icon: "💡", query: "What is RAG and how does it work?", label: "What is RAG?" },
  { icon: "📊", query: "Explain gradient descent in machine learning", label: "Gradient descent" },
  { icon: "🐳", query: "How do Docker containers work?", label: "Docker containers" },
  { icon: "🔗", query: "What are REST API best practices?", label: "REST API design" },
];

let messageId = 0;
function nextId(): string {
  messageId += 1;
  return String(messageId);
}

function SourceCards({ sources }: { sources: SourceHit[] }) {
  if (!sources.length) return null;

  return (
    <div className="sources-block">
      <div className="sources-label">Sources used</div>
      <div className="sources-grid">
        {sources.map((source, index) => {
          const score = formatScore(source);
          const chunkId = source.chunk_id ? String(source.chunk_id) : "";
          return (
            <div
              key={`${source.document}-${chunkId}-${index}`}
              className="source-card"
              title={`${chunkId || source.document} — score ${score}`}
            >
              <div className="source-info">
                <span className="source-icon" aria-hidden="true">
                  📄
                </span>
                <div className="source-meta">
                  <span className="source-name">{source.document}</span>
                  {chunkId ? <span className="source-chunk">{chunkId}</span> : null}
                </div>
              </div>
              <div className="source-score-wrap" aria-label={`Similarity score ${score}`}>
                <span className="source-score">{score}</span>
                <span className="source-score-label">Score</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ChatMessages({ messages }: { messages: ChatMessage[] }) {
  return (
    <>
      {messages.map((msg) => {
        if (msg.role === "typing") {
          return (
            <div key={msg.id} className="message assistant typing">
              <div className="msg-avatar">AI</div>
              <div className="message-body">
                <div className="bubble">
                  <span className="dot" />
                  <span className="dot" />
                  <span className="dot" />
                </div>
              </div>
            </div>
          );
        }

        if (msg.role === "user") {
          return (
            <div key={msg.id} className="message user">
              <div className="msg-avatar">You</div>
              <div className="message-body">
                <div className="bubble">{msg.text}</div>
              </div>
            </div>
          );
        }

        if (msg.role === "error") {
          return (
            <div key={msg.id} className="message assistant error">
              <div className="msg-avatar">!</div>
              <div className="message-body">
                <div className="bubble">{msg.text}</div>
              </div>
            </div>
          );
        }

        return (
          <div key={msg.id} className="message assistant">
            <div className="msg-avatar">AI</div>
            <div className="message-body">
              <div className="bubble">
                <div dangerouslySetInnerHTML={{ __html: formatAnswer(msg.answer) }} />
                <SourceCards sources={msg.sources} />
              </div>
            </div>
          </div>
        );
      })}
    </>
  );
}

export default function App() {
  const chatRef = useRef<HTMLElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState("");
  const [docCount, setDocCount] = useState("—");
  const [retrievalMethod, setRetrievalMethod] = useState("TF-IDF");
  const [providerLabel, setProviderLabel] = useState("—");

  const showWelcome = messages.length === 0;

  useEffect(() => {
    fetchStats().then((data) => {
      if (!data) return;
      setDocCount(`${data.document_count} Documents Indexed`);
      setRetrievalMethod(`${data.retrieval_method} Retrieval`);
      setProviderLabel(`Provider: ${data.llm_provider}`);
    });
  }, []);

  useEffect(() => {
    const chat = chatRef.current;
    if (chat) {
      chat.scrollTop = chat.scrollHeight;
    }
  }, [messages, loading]);

  const handleAsk = useCallback(async (question: string) => {
    const trimmed = question.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setQuery("");
    setMessages((prev) => [...prev, { id: nextId(), role: "user", text: trimmed }]);

    const typingId = nextId();
    setMessages((prev) => [...prev, { id: typingId, role: "typing" }]);

    try {
      const data = await askQuestion(trimmed, 3);
      setMessages((prev) => {
        const withoutTyping = prev.filter((m) => m.id !== typingId);
        return [
          ...withoutTyping,
          {
            id: nextId(),
            role: "assistant",
            answer: data.answer,
            sources: data.sources,
          },
        ];
      });

      if (data.provider) {
        const label = PROVIDER_LABELS[data.provider] || data.provider;
        setProviderLabel(`Provider: ${label}`);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setMessages((prev) => {
        const withoutTyping = prev.filter((m) => m.id !== typingId);
        return [
          ...withoutTyping,
          {
            id: nextId(),
            role: "error",
            text: `Something went wrong: ${message}`,
          },
        ];
      });
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }, [loading]);

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    handleAsk(query);
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-brand">
          <div className="logo" aria-hidden="true">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          <div>
            <h1>RAG Document Assistant</h1>
            <p className="tagline">AI-powered question answering over your document knowledge base</p>
          </div>
        </div>
        <div className="topbar-meta">
          <div className="meta-item">
            <span className="meta-value">{docCount}</span>
            <span className="meta-label">Documents Indexed</span>
          </div>
          <div className="meta-divider" />
          <div className="meta-item">
            <span className="meta-value">{retrievalMethod}</span>
            <span className="meta-label">Retrieval</span>
          </div>
          <div className="meta-divider" />
          <div className="meta-item">
            <span className="meta-value">{providerLabel}</span>
            <span className="meta-label">Provider</span>
          </div>
        </div>
      </header>

      <main className="chat" ref={chatRef} role="log" aria-live="polite">
        {showWelcome ? (
          <div className="welcome">
            <div className="welcome-icon" aria-hidden="true">
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <circle cx="12" cy="12" r="10" />
                <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                <line x1="12" y1="17" x2="12.01" y2="17" />
              </svg>
            </div>
            <h2>Ask a question about your indexed documents</h2>
            <p>Get instant answers grounded in your knowledge base, with source citations.</p>
            <div className="suggestions">
              {SUGGESTIONS.map((item) => (
                <button
                  key={item.query}
                  type="button"
                  className="suggestion"
                  onClick={() => handleAsk(item.query)}
                  disabled={loading}
                >
                  <span className="suggestion-icon">{item.icon}</span>
                  {item.label}
                </button>
              ))}
            </div>
          </div>
        ) : null}

        <ChatMessages messages={messages} />
      </main>

      <footer className="input-bar">
        <form className="input-form" onSubmit={handleSubmit}>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about your indexed documents…"
            autoComplete="off"
            required
            aria-label="Your question"
            disabled={loading}
          />
          <button type="submit" aria-label="Send question" disabled={loading}>
            <svg
              className={`send-icon${loading ? " hidden" : ""}`}
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
            >
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
            <span className={`spinner${loading ? "" : " hidden"}`} aria-hidden="true" />
          </button>
        </form>
        <p className="footer-note">Powered by FastAPI, TF-IDF Retrieval, and Groq LLM</p>
      </footer>
    </div>
  );
}
