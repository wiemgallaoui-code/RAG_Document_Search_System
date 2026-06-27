export interface SourceHit {
  document: string;
  chunk_id?: string;
  similarity_score?: number;
  score?: number;
}

export interface AskResponse {
  query: string;
  answer: string;
  sources: SourceHit[];
  provider: string;
}

export interface StatsResponse {
  document_count: number;
  chunk_count: number;
  retrieval_method: string;
  llm_provider: string;
}

export type ChatMessage =
  | { id: string; role: "user"; text: string }
  | { id: string; role: "assistant"; answer: string; sources: SourceHit[] }
  | { id: string; role: "error"; text: string }
  | { id: string; role: "typing" };
