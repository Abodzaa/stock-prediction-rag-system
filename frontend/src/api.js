const BASE = import.meta.env.VITE_API_BASE || "";

export class ApiError extends Error {
  constructor(message, options = {}) {
    super(message);
    this.name = "ApiError";
    this.status = options.status ?? null;
    this.body = options.body ?? "";
    this.retriable = Boolean(options.retriable);
  }
}

async function request(path, options = {}) {
  const response = await fetch(`${BASE}${path}`, {
    method: options.method || "GET",
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
    signal: options.signal,
  });

  const contentType = response.headers.get("content-type") || "";
  const isJson = contentType.includes("application/json");
  const payload = isJson ? await response.json().catch(() => null) : await response.text();

  if (!response.ok) {
    throw new ApiError(buildErrorMessage(response.status, payload), {
      status: response.status,
      body: payload,
      retriable: isRetriableStatus(response.status),
    });
  }

  return payload;
}

function buildErrorMessage(status, payload) {
  if (status === 502 || status === 503 || status === 504) {
    return "The research system is still warming up.";
  }

  if (payload && typeof payload === "object") {
    return payload.detail || payload.message || `Request failed (${status})`;
  }

  if (typeof payload === "string" && payload.trim()) {
    return payload.trim();
  }

  return `Request failed (${status})`;
}

export function isRetriableStatus(status) {
  return status === 0 || status === 429 || status === 502 || status === 503 || status === 504;
}

export async function withRetry(task, options = {}) {
  const retries = options.retries ?? 3;
  const baseDelay = options.baseDelay ?? 800;
  let lastError = null;

  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      return await task(attempt);
    } catch (error) {
      lastError = error;
      const retriable = error instanceof ApiError ? error.retriable : false;
      if (!retriable || attempt === retries) {
        throw error;
      }
      await wait(baseDelay * 2 ** attempt);
    }
  }

  throw lastError;
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export const api = {
  health: (options) => request("/api/health", options),
  facets: (options) => request("/api/models/facets", options),
  listModels: (params, options = {}) => {
    const query = new URLSearchParams(
      Object.entries(params).filter(([, value]) => value !== "" && value != null)
    );
    return request(`/api/models?${query.toString()}`, options);
  },
  predict: (body, options) =>
    request("/api/predict", {
      ...options,
      method: "POST",
      body,
    }),
  explainStream(params, handlers) {
    const query = new URLSearchParams(
      Object.entries(params).filter(([, value]) => value !== "" && value != null)
    );
    const eventSource = new EventSource(`${BASE}/api/explain/stream?${query.toString()}`);

    eventSource.onmessage = (event) => {
      let payload;
      try {
        payload = JSON.parse(event.data);
      } catch {
        return;
      }
      handlers[payload.type]?.(payload.data);
      if (payload.type === "done" || payload.type === "error") {
        eventSource.close();
      }
    };

    eventSource.onerror = () => {
      handlers.error?.("Explanation stream interrupted.");
      eventSource.close();
    };

    return () => eventSource.close();
  },
};
