export type AOMode = "vajra" | "classical";
export type TargetMode = "solar" | "point";

export interface ResetRequest {
  mode: AOMode;
  r0: number;
  wind_speed: number;
  target_mode: TargetMode;
}

export interface ResetResponse extends ResetRequest {
  status: "success";
}

export interface StepResponse {
  rms: number;
  strehl: number;
  regime: string;
  atm_img: string;
  res_img: string;
  wfs_img: string;
  dm_img: string;
  rms_history: number[];
  strehl_history: number[];
}

export class VajraApiError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
    this.name = "VajraApiError";
  }
}

async function postJson<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    throw new VajraApiError(`VAJRA API request failed: ${response.status}`, response.status);
  }

  return (await response.json()) as T;
}

export function resetLoop(params: ResetRequest, signal?: AbortSignal) {
  return postJson<ResetResponse>("/api/reset", params, signal);
}

export function runLoopStep(signal?: AbortSignal) {
  return postJson<StepResponse>("/api/run_step", undefined, signal);
}

export const rmsRadiansToNanometres = (rmsRadians: number) =>
  (rmsRadians * 500) / (2 * Math.PI);
