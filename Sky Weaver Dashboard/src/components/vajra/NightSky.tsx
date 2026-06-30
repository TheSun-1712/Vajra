import { useEffect, useMemo, useRef } from "react";
import nlstHanle from "@/assets/nlst-hanle.jpg.asset.json";

function useReducedMotion() {
  const ref = useRef(false);
  useEffect(() => {
    const m = window.matchMedia("(prefers-reduced-motion: reduce)");
    ref.current = m.matches;
  }, []);
  return ref;
}

interface Star { x: number; y: number; r: number; tw: number; hue: number; }

function mulberry(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Stars distributed across the sky, with extra density along a diagonal
// "galactic band" to evoke the Milky Way.
function makeStars(count: number, seed: number, bandWeight = 0): Star[] {
  const rnd = mulberry(seed);
  const out: Star[] = [];
  for (let i = 0; i < count; i++) {
    let x = rnd() * 100;
    let y = rnd() * 100;
    if (bandWeight > 0 && rnd() < bandWeight) {
      // Bias toward a diagonal band: y ≈ 0.35x + 25 ± narrow scatter
      x = rnd() * 100;
      const center = 0.35 * x + 22;
      const spread = (rnd() - 0.5) * 22;
      y = Math.max(0, Math.min(100, center + spread));
    }
    out.push({
      x,
      y,
      r: 0.25 + Math.pow(rnd(), 3) * 2.2,
      tw: rnd(),
      hue: rnd(),
    });
  }
  return out;
}

export function NightSky() {
  const reduced = useReducedMotion();
  const layers = useMemo(
    () => [
      { stars: makeStars(260, 11, 0.55), speed: 320, opacity: 0.5 },
      { stars: makeStars(160, 77, 0.45), speed: 220, opacity: 0.85 },
      { stars: makeStars(70, 131, 0.3), speed: 140, opacity: 1 },
    ],
    [],
  );

  return (
    <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden" aria-hidden>
      {/* Hero background photo — NLST / Hanle under the Milky Way */}
      <img
        src={nlstHanle.url}
        alt=""
        aria-hidden
        className="absolute inset-0 h-full w-full object-cover"
        style={{ objectPosition: "center 30%" }}
        draggable={false}
      />
      {/* Tonal grade so the photo reads as deep night and the UI stays legible */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "linear-gradient(180deg, rgba(7,11,20,0.45) 0%, rgba(7,11,20,0.55) 55%, rgba(2,4,10,0.85) 100%)",
        }}
      />
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at 72% 10%, rgba(27,55,104,0.35) 0%, rgba(10,20,38,0.25) 42%, rgba(2,4,10,0.55) 100%)",
          mixBlendMode: "multiply",
        }}
      />

      {/* Milky Way band — soft luminous diagonal sweep */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "linear-gradient(110deg, transparent 22%, rgba(95,190,220,0.16) 38%, rgba(255,214,164,0.22) 50%, rgba(135,170,255,0.18) 61%, transparent 80%)",
          filter: "blur(11px)",
          mixBlendMode: "screen",
        }}
      />
      {/* Galactic core glow */}
      <div
        className="absolute"
        style={{
          left: "62%",
          top: "28%",
          width: "52vw",
          height: "52vw",
          transform: "translate(-50%,-50%) rotate(20deg)",
          background:
            "radial-gradient(ellipse 60% 22% at 50% 50%, rgba(255,218,174,0.30), rgba(180,210,255,0.16) 42%, transparent 72%)",
          filter: "blur(17px)",
          mixBlendMode: "screen",
        }}
      />

      {/* Nebula clouds */}
      <div
        className="absolute inset-0 opacity-90"
        style={{
          background:
            "radial-gradient(ellipse 50% 35% at 18% 22%, rgba(34,211,238,0.18), transparent 65%), radial-gradient(ellipse 40% 28% at 86% 70%, rgba(251,146,60,0.12), transparent 70%), radial-gradient(ellipse 30% 22% at 50% 85%, rgba(120,90,200,0.12), transparent 70%)",
          mixBlendMode: "screen",
        }}
      />

      {/* Parallax star layers */}
      {layers.map((layer, idx) => (
        <div
          key={idx}
          className="absolute top-0 left-0 h-full"
          style={{
            width: "200%",
            animation: reduced.current
              ? undefined
              : `vajra-drift-1 ${layer.speed}s linear infinite`,
          }}
        >
          <svg viewBox="0 0 2000 1000" preserveAspectRatio="none" className="h-full w-full">
            {layer.stars.map((st, i) => {
              const cx = (st.x / 100) * 2000;
              const cy = (st.y / 100) * 1000;
              const twinkles = i % 9 === 0;
              const tint =
                st.hue < 0.15
                  ? "#fde68a"
                  : st.hue < 0.3
                  ? "#fbcfe8"
                  : st.hue < 0.5
                  ? "#bfdbfe"
                  : "#ffffff";
              return (
                <circle
                  key={i}
                  cx={cx}
                  cy={cy}
                  r={st.r}
                  fill={tint}
                  opacity={layer.opacity}
                  style={
                    twinkles && !reduced.current
                      ? {
                          animation: `vajra-twinkle ${2.5 + st.tw * 5}s ease-in-out ${st.tw * 6}s infinite`,
                        }
                      : undefined
                  }
                />
              );
            })}
            {/* a couple of brighter beacon stars with diffraction spikes */}
            {idx === 2 &&
              [
                { x: 320, y: 180 },
                { x: 1480, y: 280 },
                { x: 1100, y: 120 },
              ].map((p, i) => (
                <g key={`b${i}`} opacity="0.85">
                  <circle cx={p.x} cy={p.y} r={2} fill="#ffffff" />
                  <circle cx={p.x} cy={p.y} r={6} fill="#ffffff" opacity="0.18" />
                  <line x1={p.x - 14} y1={p.y} x2={p.x + 14} y2={p.y} stroke="#ffffff" strokeWidth="0.4" opacity="0.5" />
                  <line x1={p.x} y1={p.y - 14} x2={p.x} y2={p.y + 14} stroke="#ffffff" strokeWidth="0.4" opacity="0.5" />
                </g>
              ))}
          </svg>
        </div>
      ))}

    </div>
  );
}

