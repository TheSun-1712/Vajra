import { useEffect, useRef } from "react";

type Mode = "phase" | "gray" | "wfs";

function colormap(v: number, mode: Mode): [number, number, number] {
  // v in [0,1]
  if (mode === "gray") {
    const g = Math.round(v * 255);
    return [g, g, g];
  }
  if (mode === "wfs") {
    // grayscale with subtle blue tint
    const g = Math.round(v * 220);
    return [Math.round(g * 0.85), g, Math.round(g * 1.1)];
  }
  // phase: blue -> green -> red (jet-like)
  const r = Math.round(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4 * v - 3))));
  const g = Math.round(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4 * v - 2))));
  const b = Math.round(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4 * v - 1))));
  return [r, g, b];
}

interface Props {
  mode: Mode;
  size?: number;
  speed?: number;
  scale?: number;
  paused?: boolean;
  reduced?: boolean;
  className?: string;
  grid?: boolean;
}

export function HeatCanvas({
  mode,
  size = 96,
  speed = 0.002,
  scale = 0.08,
  paused = false,
  reduced = false,
  className,
  grid = false,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const tRef = useRef(0);
  const rafRef = useRef<number | null>(null);
  const pausedRef = useRef(paused);

  useEffect(() => { pausedRef.current = paused; }, [paused]);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    cv.width = size;
    cv.height = size;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(size, size);

    const draw = () => {
      const t = tRef.current;
      // pseudo-turbulence: sum of a few sinusoids with offsets
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const u = x * scale;
          const v = y * scale;
          let n =
            Math.sin(u + t) +
            Math.sin(v * 1.3 - t * 0.7) +
            Math.sin((u + v) * 0.7 + t * 0.5) +
            Math.sin(Math.sqrt(u * u + v * v) * 1.5 - t);
          n = (n + 4) / 8; // normalize ~[0,1]
          if (grid) {
            // actuator-like grid: posterize to 16x16 cells
            const cs = size / 16;
            const cx = Math.floor(x / cs);
            const cy = Math.floor(y / cs);
            const cu = cx * scale * cs;
            const cv2 = cy * scale * cs;
            n =
              (Math.sin(cu + t) +
                Math.sin(cv2 - t * 0.6) +
                Math.sin((cu + cv2) * 0.5 + t)) /
                3 +
              0.5;
            n = Math.max(0, Math.min(1, n));
            // gap between cells
            if (x % cs < 1 || y % cs < 1) n *= 0.3;
          }
          const [r, g, b] = colormap(n, mode);
          const idx = (y * size + x) * 4;
          img.data[idx] = r;
          img.data[idx + 1] = g;
          img.data[idx + 2] = b;
          img.data[idx + 3] = mode === "phase" ? 245 : mode === "wfs" ? 235 : 230;
        }
      }
      ctx.putImageData(img, 0, 0);

      if (mode === "wfs") {
        // overlay subhaperture grid + spot dots
        ctx.strokeStyle = "rgba(34,211,238,0.25)";
        ctx.lineWidth = 0.5;
        const cells = 10;
        const cs = size / cells;
        for (let i = 1; i < cells; i++) {
          ctx.beginPath();
          ctx.moveTo(i * cs, 0);
          ctx.lineTo(i * cs, size);
          ctx.moveTo(0, i * cs);
          ctx.lineTo(size, i * cs);
          ctx.stroke();
        }
        ctx.fillStyle = "rgba(251,146,60,0.95)";
        for (let i = 0; i < cells; i++) {
          for (let j = 0; j < cells; j++) {
            const jitterX = Math.sin(i * 1.3 + j + t) * 1.2;
            const jitterY = Math.cos(i + j * 1.7 - t) * 1.2;
            ctx.beginPath();
            ctx.arc(
              i * cs + cs / 2 + jitterX,
              j * cs + cs / 2 + jitterY,
              0.9,
              0,
              Math.PI * 2,
            );
            ctx.fill();
          }
        }
      }
    };

    const loop = () => {
      if (!pausedRef.current) {
        tRef.current += reduced ? 0 : speed * 16;
        draw();
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    draw();
    rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [mode, size, speed, scale, grid, reduced]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ imageRendering: "pixelated", width: "100%", height: "100%" }}
    />
  );
}
