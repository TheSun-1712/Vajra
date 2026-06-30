import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
} from "recharts";
import { Pause, Play, RotateCcw, ChevronDown } from "lucide-react";
import { NightSky } from "@/components/vajra/NightSky";
import {
  resetLoop,
  rmsRadiansToNanometres,
  runLoopStep,
  type AOMode,
  type StepResponse,
} from "@/lib/vajra-api";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "VAJRA — Adaptive Optics Control | NLST Hanle" },
      {
        name: "description",
        content:
          "Real-time adaptive-optics control center for the National Large Solar Telescope at Indian Astronomical Observatory, Hanle, Ladakh.",
      },
    ],
  }),
  component: VajraDashboard,
});

const TARGETS = ["Solar Granulation", "Sunspot AR3664", "Quiet Sun Disk Center", "Limb Prominence"];
const DEFAULT_MODE: AOMode = "vajra";
const DEFAULT_TARGET = TARGETS[0];
const DEFAULT_R0_CM = 12;
const DEFAULT_WIND_MS = 8;
const STEP_INTERVAL_MS = 700;

function useReducedMotion() {
  const [r, setR] = useState(false);
  useEffect(() => {
    const m = window.matchMedia("(prefers-reduced-motion: reduce)");
    setR(m.matches);
    const fn = () => setR(m.matches);
    m.addEventListener("change", fn);
    return () => m.removeEventListener("change", fn);
  }, []);
  return r;
}

function useCountUp(value: number, digits = 3) {
  const [display, setDisplay] = useState(value);
  const fromRef = useRef(value);
  useEffect(() => {
    const from = fromRef.current;
    const to = value;
    const start = performance.now();
    const dur = 400;
    let raf = 0;
    const tick = (t: number) => {
      const p = Math.min(1, (t - start) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      setDisplay(from + (to - from) * eased);
      if (p < 1) raf = requestAnimationFrame(tick);
      else fromRef.current = to;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value]);
  return display.toFixed(digits);
}

function VajraDashboard() {
  const reduced = useReducedMotion();
  const [mode, setMode] = useState<AOMode>(DEFAULT_MODE);
  const [target, setTarget] = useState(DEFAULT_TARGET);
  const [r0, setR0] = useState(DEFAULT_R0_CM);
  const [wind, setWind] = useState(DEFAULT_WIND_MS);
  const [paused, setPaused] = useState(false);
  const [openTarget, setOpenTarget] = useState(false);
  const [online, setOnline] = useState(false);
  const [telemetry, setTelemetry] = useState<StepResponse | null>(null);
  const [history, setHistory] = useState<{ t: number; rms: number; strehl: number }[]>([]);
  const generationRef = useRef(0);

  const applyTelemetry = useCallback((data: StepResponse) => {
    setTelemetry(data);
    setOnline(true);
    const length = Math.max(data.rms_history.length, data.strehl_history.length);
    setHistory(
      Array.from({ length }, (_, index) => ({
        t: index,
        rms: rmsRadiansToNanometres(data.rms_history[index] ?? 0),
        strehl: data.strehl_history[index] ?? 0,
      })),
    );
  }, []);

  const configureBackend = useCallback(async (signal?: AbortSignal) => {
    await resetLoop(
      {
        mode,
        r0: r0 / 100,
        wind_speed: wind,
        target_mode: "solar",
      },
      signal,
    );
    const data = await runLoopStep(signal);
    applyTelemetry(data);
  }, [applyTelemetry, mode, r0, wind, target]);

  useEffect(() => {
    const controller = new AbortController();
    const generation = ++generationRef.current;
    const timer = window.setTimeout(() => {
      configureBackend(controller.signal).catch((error) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        if (generation === generationRef.current) setOnline(false);
      });
    }, 300);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [configureBackend]);

  useEffect(() => {
    if (paused || !online) return;
    const controller = new AbortController();
    let stopped = false;
    let timer: number | undefined;

    const poll = async () => {
      try {
        const data = await runLoopStep(controller.signal);
        if (stopped) return;
        applyTelemetry(data);
        timer = window.setTimeout(poll, STEP_INTERVAL_MS);
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") return;
        if (!stopped) setOnline(false);
      }
    };

    timer = window.setTimeout(poll, STEP_INTERVAL_MS);
    return () => {
      stopped = true;
      if (timer !== undefined) window.clearTimeout(timer);
      controller.abort();
    };
  }, [applyTelemetry, online, paused]);

  const rms = telemetry ? rmsRadiansToNanometres(telemetry.rms) : 0;
  const strehl = telemetry?.strehl ?? 0;

  const rmsDisplay = useCountUp(rms, 1);
  const strehlDisplay = useCountUp(strehl, 3);

  const seeingRegime = r0 > 15 ? "EXCELLENT" : r0 > 10 ? "GOOD" : r0 > 6 ? "AVERAGE" : "POOR";
  const seeingColor = r0 > 15 ? "var(--cyan)" : r0 > 6 ? "#a5d8ff" : "var(--amber)";

  return (
    <div className={reduced ? "vajra-reduce-motion min-h-screen" : "min-h-screen"}>
      <NightSky />

      <div className="relative z-10 mx-auto max-w-[1600px] px-5 py-5 font-mono text-[13px] text-foreground">
        {/* TOP BAR */}
        <header
          className="constellation-field rounded-sm border-b border-[var(--cyan)]/20 bg-transparent px-5 py-3 flex items-center justify-between vajra-fade-in"
          style={{ animationDelay: "0ms" }}
        >
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <div className="relative h-8 w-8 rounded-sm border border-[var(--cyan)]/40 flex items-center justify-center">
                <span className="text-[var(--cyan)] text-xs tracking-widest">V</span>
                <span
                  className="absolute -inset-px rounded-sm pointer-events-none"
                  style={{ boxShadow: "inset 0 0 8px rgba(34,211,238,0.4)" }}
                />
              </div>
              <div>
                <div className="text-base font-semibold tracking-[0.28em] text-foreground">VAJRA</div>
                <div className="text-[10px] uppercase tracking-[0.22em] text-[var(--slate-muted)]">
                  Adaptive Optics Control · NLST · IAO Hanle, Ladakh
                </div>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <StatusPill label="TELESCOPE" value="NLST 2.0m" />
            <StatusPill label="SITE" value="IAO Hanle · 4500 m" />
            <StatusPill
              label="LOOP"
              value={!online ? "OFFLINE" : paused ? "PAUSED" : telemetry?.regime.toUpperCase() || "CLOSED"}
              amber={paused || !online}
            />
          </div>
        </header>

        <div className="mt-4 grid grid-cols-12 gap-4">
          {/* LEFT — CONTROL CENTER */}
          <section
            className="sky-panel constellation-field rounded-sm p-4 col-span-12 lg:col-span-3 vajra-fade-in"
            style={{ animationDelay: "120ms" }}
          >
            <div className="flex items-center justify-between mb-3">
              <h2 className="panel-label">Control Center</h2>
              <span className="text-[10px] text-[var(--slate-muted)]">REV 2.4.1</span>
            </div>

            {/* AO mode segmented */}
            <div className="panel-label mb-1.5">AO Mode</div>
            <div className="relative grid grid-cols-2 rounded-sm border border-[var(--cyan)]/16 bg-[rgba(7,11,20,0.08)] p-1 mb-4 shadow-[inset_0_0_28px_rgba(34,211,238,0.04)]">
              <div
                className="absolute top-1 bottom-1 w-[calc(50%-4px)] rounded-sm transition-transform duration-300 ease-out"
                style={{
                  background: "linear-gradient(180deg, rgba(34,211,238,0.14), rgba(34,211,238,0.025))",
                  border: "1px solid rgba(34,211,238,0.36)",
                  boxShadow: "0 0 18px rgba(34,211,238,0.18), inset 0 0 18px rgba(34,211,238,0.05)",
                  transform: mode === "vajra" ? "translateX(0)" : "translateX(100%)",
                  left: 4,
                }}
              />
              {(["vajra", "classical"] as const).map((m) => (
                <button
                  key={m}
                  disabled={!online}
                  onClick={() => setMode(m)}
                  className="relative z-10 py-1.5 text-[11px] tracking-[0.18em] uppercase transition-colors hover:text-[var(--cyan)] disabled:cursor-not-allowed disabled:opacity-50"
                  style={{ color: mode === m ? "var(--cyan)" : "var(--slate-muted)" }}
                >
                  {m === "vajra" ? "VAJRA AO" : "Classical SH"}
                </button>
              ))}
            </div>

            {/* Target dropdown */}
            <div className="panel-label mb-1.5">Sensing Target</div>
            <div className="relative mb-4">
              <button
                disabled={!online}
                onClick={() => setOpenTarget((o) => !o)}
                className="w-full flex items-center justify-between rounded-sm border border-[var(--cyan)]/16 bg-[rgba(7,11,20,0.08)] px-3 py-2 text-left text-[12px] shadow-[inset_0_0_24px_rgba(34,211,238,0.035)] transition-colors hover:border-[var(--cyan)]/50 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <span>{target}</span>
                <ChevronDown size={14} className="text-[var(--slate-muted)]" />
              </button>
              {openTarget && (
                  <div className="absolute z-20 mt-1 w-full overflow-hidden rounded-sm border border-[var(--cyan)]/30 bg-[rgba(7,11,20,0.72)] backdrop-blur-md">
                  {TARGETS.map((t) => (
                    <button
                      key={t}
                      onClick={() => { setTarget(t); setOpenTarget(false); }}
                      className="block w-full px-3 py-2 text-left text-[12px] hover:bg-[var(--cyan)]/10 hover:text-[var(--cyan)]"
                    >
                      {t}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Sliders */}
            <GlowSlider
              label="Fried Parameter r0"
              unit="cm"
              min={3}
              max={25}
              step={0.5}
              value={r0}
              onChange={setR0}
              disabled={!online}
            />
            <GlowSlider
              label="Wind Velocity"
              unit="m/s"
              min={0}
              max={25}
              step={0.5}
              value={wind}
              onChange={setWind}
              disabled={!online}
            />

            {/* Stat cards */}
            <div className="grid grid-cols-2 gap-2 mt-4">
              <StatCard label="Residual RMS" value={rmsDisplay} unit="nm" tone="cyan" />
              <StatCard label="Strehl Ratio" value={strehlDisplay} unit="" tone="amber" />
            </div>

            {/* Seeing regime */}
            <div className="mt-4 flex items-center justify-between rounded-sm border border-[var(--cyan)]/12 bg-[rgba(7,11,20,0.07)] px-3 py-2 shadow-[inset_0_0_22px_rgba(251,146,60,0.025)]">
              <span className="panel-label">Seeing Regime</span>
              <span
                className="text-[11px] tracking-[0.18em] font-semibold"
                style={{ color: seeingColor }}
              >
                ● {seeingRegime}
              </span>
            </div>

            {/* Actions */}
            <button
              disabled={!online}
              onClick={() => setPaused((p) => !p)}
              className="mt-4 w-full flex items-center justify-center gap-2 rounded-md py-2.5 text-[12px] tracking-[0.18em] uppercase font-semibold transition-all hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:scale-100"
              style={{
                background: "linear-gradient(180deg, rgba(34,211,238,0.25), rgba(34,211,238,0.08))",
                border: "1px solid rgba(34,211,238,0.55)",
                color: "var(--cyan)",
                boxShadow: "0 0 16px rgba(34,211,238,0.2)",
              }}
            >
              {paused ? <Play size={14} /> : <Pause size={14} />}
              {paused ? "Resume Loop" : "Pause Loop"}
            </button>
            <button
              disabled={!online}
              onClick={() => {
                setR0(DEFAULT_R0_CM);
                setWind(DEFAULT_WIND_MS);
                setMode(DEFAULT_MODE);
                setTarget(DEFAULT_TARGET);
                setPaused(false);
              }}
              className="mt-2 w-full flex items-center justify-center gap-2 rounded-md py-2 text-[11px] tracking-[0.18em] uppercase text-[var(--slate-muted)] border border-[var(--cyan)]/15 hover:text-foreground hover:border-[var(--cyan)]/40 transition-colors disabled:cursor-not-allowed disabled:opacity-50"
            >
              <RotateCcw size={12} />
              Reset Optics
            </button>
          </section>

          {/* RIGHT — VIEWPORTS */}
          <section
            className="sky-panel constellation-field rounded-sm p-4 col-span-12 lg:col-span-9 vajra-fade-in"
            style={{ animationDelay: "240ms" }}
          >
            <div className="flex items-center justify-between mb-3">
              <h2 className="panel-label">Live Optical Viewports</h2>
              <div className="flex items-center gap-2 text-[10px] text-[var(--slate-muted)]">
                <span
                  className="inline-block h-1.5 w-1.5 rounded-full bg-[var(--cyan)]"
                  style={{ animation: reduced ? undefined : "vajra-pulse-dot 1.4s ease-in-out infinite" }}
                />
                {!online ? "OFFLINE" : paused ? "PAUSED" : "STREAMING · LIVE"}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <Viewport label="Atmosphere Phase Screen" sub="λ = 656 nm · turbulent layer 8 km">
                <TelemetryFrame src={telemetry?.atm_img} alt="Atmosphere phase screen" />
              </Viewport>
              <Viewport label="Residual Pupil Phase" sub={mode === "vajra" ? "post-correction · VAJRA" : "post-correction · SH"}>
                <TelemetryFrame src={telemetry?.res_img} alt="Residual pupil phase" />
              </Viewport>
              <Viewport label="WFS Camera Frame" sub="Shack-Hartmann · 10×10 subaps">
                <TelemetryFrame src={telemetry?.wfs_img} alt="Wavefront sensor camera frame" />
              </Viewport>
              <Viewport label="DM Actuator Map" sub="97 actuators · ±2.5 µm stroke" grayscale>
                <TelemetryFrame src={telemetry?.dm_img} alt="Deformable mirror actuator map" />
              </Viewport>
            </div>
          </section>
        </div>

        {/* BOTTOM — chart */}
        <section
          className="sky-panel constellation-field rounded-sm p-4 mt-4 vajra-fade-in"
          style={{ animationDelay: "360ms" }}
        >
          <div className="flex items-center justify-between mb-3">
            <h2 className="panel-label">Loop Performance History</h2>
            <div className="flex items-center gap-4 text-[11px]">
              <LegendSwatch color="var(--cyan)" label="Residual RMS (nm)" />
              <LegendSwatch color="var(--amber)" label="Strehl Ratio" />
            </div>
          </div>

          <div className="h-[200px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={history} margin={{ top: 5, right: 40, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="rmsArea" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#22D3EE" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#22D3EE" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(34,211,238,0.08)" vertical={false} />
                <XAxis dataKey="t" stroke="#64748B" tick={{ fontSize: 10, fontFamily: "JetBrains Mono" }} tickLine={false} axisLine={{ stroke: "rgba(34,211,238,0.15)" }} />
                <YAxis
                  yAxisId="left"
                  stroke="#22D3EE"
                  tick={{ fontSize: 10, fontFamily: "JetBrains Mono", fill: "#22D3EE" }}
                  tickLine={false}
                  axisLine={{ stroke: "rgba(34,211,238,0.2)" }}
                  width={45}
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  stroke="#FB923C"
                  domain={[0, 1]}
                  tick={{ fontSize: 10, fontFamily: "JetBrains Mono", fill: "#FB923C" }}
                  tickLine={false}
                  axisLine={{ stroke: "rgba(251,146,60,0.3)" }}
                  width={40}
                />
                <Tooltip
                  contentStyle={{
                    background: "rgba(7,11,20,0.95)",
                    border: "1px solid rgba(34,211,238,0.3)",
                    borderRadius: 4,
                    fontFamily: "JetBrains Mono",
                    fontSize: 11,
                  }}
                  labelStyle={{ color: "#64748B" }}
                />
                <Area
                  yAxisId="left"
                  type="monotone"
                  dataKey="rms"
                  stroke="#22D3EE"
                  strokeWidth={1.5}
                  fill="url(#rmsArea)"
                  isAnimationActive={false}
                  dot={false}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="strehl"
                  stroke="#FB923C"
                  strokeWidth={1.5}
                  isAnimationActive={false}
                  dot={false}
                />
                <Legend wrapperStyle={{ display: "none" }} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </section>

        <footer className="mt-4 flex items-center justify-between text-[10px] text-[var(--slate-muted)] px-1">
          <span>VAJRA · Vision-Augmented Joint Reconstruction for AO · build 2026.06</span>
          <span>UTC {new Date().toISOString().slice(11, 19)} · ALT 4500 m · AMB −12°C</span>
        </footer>
      </div>
    </div>
  );
}

function StatusPill({ label, value, amber }: { label: string; value: string; amber?: boolean }) {
  const color = amber ? "var(--amber)" : "var(--cyan)";
  return (
    <div
      className="flex items-center gap-2 rounded-full border px-3 py-1 text-[10px] tracking-[0.16em] uppercase"
      style={{ borderColor: `${color}33`, background: "rgba(0,0,0,0.25)" }}
    >
      <span
        className="inline-block h-1.5 w-1.5 rounded-full"
        style={{
          background: color,
          color,
          animation: "vajra-pulse-dot 1.6s ease-in-out infinite",
        }}
      />
      <span className="text-[var(--slate-muted)]">{label}</span>
      <span className="text-foreground">{value}</span>
    </div>
  );
}

function GlowSlider({
  label, unit, min, max, step, value, onChange,
}: {
  label: string; unit: string; min: number; max: number; step: number;
  value: number; onChange: (v: number) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="mb-3 rounded-sm border border-[var(--cyan)]/10 bg-[rgba(7,11,20,0.045)] px-2.5 py-2 shadow-[inset_0_0_24px_rgba(34,211,238,0.025)]">
      <div className="flex items-center justify-between mb-1.5">
        <span className="panel-label">{label}</span>
        <span className="text-[11px] text-foreground">
          {value.toFixed(1)} <span className="text-[var(--slate-muted)]">{unit}</span>
        </span>
      </div>
      <div className="relative h-6 flex items-center">
        <div className="absolute inset-x-0 h-1 rounded-full border border-[var(--cyan)]/10 bg-[rgba(7,11,20,0.18)]" />
        <div
          className="absolute h-1 rounded-full"
          style={{
            width: `${pct}%`,
            background: "linear-gradient(90deg, rgba(251,146,60,0.4), #FB923C)",
            boxShadow: dragging ? "0 0 10px rgba(251,146,60,0.7)" : "0 0 4px rgba(251,146,60,0.3)",
          }}
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          onMouseDown={() => setDragging(true)}
          onMouseUp={() => setDragging(false)}
          onTouchStart={() => setDragging(true)}
          onTouchEnd={() => setDragging(false)}
          className="vajra-slider relative z-10 w-full appearance-none bg-transparent cursor-pointer"
        />
      </div>
      <style>{`
        .vajra-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 14px; height: 14px; border-radius: 50%;
          background: #FB923C;
          border: 2px solid #070B14;
          box-shadow: 0 0 ${dragging ? "14px 2px" : "6px"} rgba(251,146,60,${dragging ? 0.9 : 0.5});
          cursor: grab;
        }
        .vajra-slider::-moz-range-thumb {
          width: 14px; height: 14px; border-radius: 50%;
          background: #FB923C;
          border: 2px solid #070B14;
          box-shadow: 0 0 ${dragging ? "14px 2px" : "6px"} rgba(251,146,60,${dragging ? 0.9 : 0.5});
        }
      `}</style>
    </div>
  );
}

function StatCard({
  label, value, unit, tone,
}: { label: string; value: string; unit: string; tone: "cyan" | "amber" }) {
  const color = tone === "cyan" ? "var(--cyan)" : "var(--amber)";
  return (
    <div className="rounded-sm px-2.5 py-2 border-l-2" style={{ borderColor: color, background: "linear-gradient(90deg, rgba(255,255,255,0.012), transparent)", boxShadow: `inset 0 0 22px ${color}10` }}>
      <div className="panel-label mb-1">{label}</div>
      <div className="flex items-baseline gap-1">
        <span
          className="text-2xl font-semibold tabular-nums tracking-tight"
          style={{ color, textShadow: `0 0 16px ${color}66` }}
        >
          {value}
        </span>
        {unit && <span className="text-[10px] text-[var(--slate-muted)]">{unit}</span>}
      </div>
    </div>
  );
}


function Viewport({
  label, sub, children, grayscale,
}: { label: string; sub: string; children: React.ReactNode; grayscale?: boolean }) {
  return (
    <div className="sky-panel constellation-field relative overflow-hidden rounded-sm group">
      <div className="pointer-events-none absolute inset-0 z-0 opacity-70" />
      <div className="relative z-10 flex items-center justify-between px-2 py-1.5">
        <div>
          <div className="text-[10px] tracking-[0.22em] uppercase text-foreground/85">{label}</div>
          <div className="text-[9px] text-[var(--slate-muted)] tracking-wider">{sub}</div>
        </div>
        <span className="text-[9px] text-[var(--cyan)]/60 tracking-[0.3em]">{grayscale ? "DM" : "PHS"}</span>
      </div>
      <div className="aspect-square relative -mt-2">
        {/* dark circular backing plate */}
        <div
          className="absolute inset-3 rounded-full pointer-events-none"
          style={{
            background:
              "radial-gradient(circle at 50% 50%, rgba(7,11,20,0.7) 0%, rgba(7,11,20,0.55) 70%, rgba(7,11,20,0.35) 100%)",
          }}
        />
        {/* canvas clipped to a hard circle */}
        <div
          className="absolute inset-3 rounded-full overflow-hidden opacity-95 transition-opacity duration-300 group-hover:opacity-100"
          style={{ clipPath: "circle(50% at 50% 50%)" }}
        >
          {children}
        </div>
        {/* faint aperture ring */}
        <div
          className="absolute inset-3 rounded-full pointer-events-none"
          style={{
            border: "1px solid rgba(34,211,238,0.22)",
            boxShadow: "inset 0 0 30px rgba(34,211,238,0.06), 0 0 24px rgba(34,211,238,0.05)",
          }}
        />
        <Corner pos="tl" /><Corner pos="tr" /><Corner pos="bl" /><Corner pos="br" />
      </div>

    </div>
  );
}

function Corner({ pos }: { pos: "tl" | "tr" | "bl" | "br" }) {
  const map = {
    tl: "top-1 left-1 border-l border-t",
    tr: "top-1 right-1 border-r border-t",
    bl: "bottom-1 left-1 border-l border-b",
    br: "bottom-1 right-1 border-r border-b",
  };
  return <div className={`absolute h-2 w-2 ${map[pos]} border-[var(--cyan)]/40`} />;
}


function LegendSwatch({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-1.5 text-[var(--slate-muted)]">
      <span className="inline-block h-2.5 w-2.5 rounded-[2px]" style={{ background: color, boxShadow: `0 0 6px ${color}` }} />
      {label}
    </div>
  );
}
