// Minimal SSE frame parser for POST /research/stream's fetch-based stream (the
// browser's built-in EventSource can't POST a body, so this reads the raw
// text stream the same way eval/harness.py's _parse_sse_stream does).
export interface SseFrame {
  eventType: string;
  data: unknown;
}

export async function* parseSseStream(
  response: Response,
  signal?: AbortSignal
): AsyncGenerator<SseFrame> {
  if (!response.body) return;
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      if (signal?.aborted) return;
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const rawFrame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const frame = parseFrame(rawFrame);
        if (frame) yield frame;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function parseFrame(rawFrame: string): SseFrame | null {
  let eventType = "message";
  const dataLines: string[] = [];
  for (const line of rawFrame.split("\n")) {
    if (line.startsWith(":")) continue; // heartbeat comment
    if (line.startsWith("event: ")) eventType = line.slice("event: ".length).trim();
    else if (line.startsWith("data: ")) dataLines.push(line.slice("data: ".length));
  }
  if (dataLines.length === 0) return null;
  try {
    return { eventType, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return null;
  }
}
