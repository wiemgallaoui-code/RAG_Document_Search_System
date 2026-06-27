function escapeHtml(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Convert plain-text LLM output into HTML with paragraphs and lists.
 */
export function formatAnswer(text: string): string {
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

export function formatScore(source: { similarity_score?: number; score?: number }): string {
  const raw = source.similarity_score ?? source.score;
  const value = Number(raw);
  return Number.isFinite(value) ? value.toFixed(4) : "—";
}

export function escapeHtmlText(str: string): string {
  return escapeHtml(str);
}
