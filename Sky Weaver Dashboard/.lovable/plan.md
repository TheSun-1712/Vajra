The 4 simulation tiles currently combine three transparency reducers (canvas alpha ~175-205, `mix-blend-screen` wrapper at opacity 0.70, and a soft radial mask that fades the edges), which makes them ghost out against the sky. I'll dial those back so the imagery reads clearly while the panels still feel integrated with the night sky.

### Changes (presentation only)

1. **`src/components/vajra/HeatCanvas.tsx`** — raise per-pixel alpha so the heatmaps are solid:
   - phase: 205 → 245
   - wfs: 185 → 235
   - gray (DM): 175 → 230

2. **`src/routes/index.tsx` — `Viewport` component**:
   - Drop `mix-blend-screen` on the canvas wrapper (screen blend washes dark blues into the sky). Keep a subtle opacity instead: `opacity 0.95`, hover `1`.
   - Soften the vignette mask so the center stays fully opaque and only the outer ring feathers:
     `radial-gradient(circle at 50%, #000 72%, rgba(0,0,0,0.85) 86%, transparent 100%)`
   - Add a faint dark backing plate behind the canvas (`bg-[rgba(7,11,20,0.35)]`) so the colors have contrast against the bright Milky Way band.
   - Keep the dashed aperture ring, corner ticks, and label chrome unchanged.

3. **`src/styles.css`** — leave `.glass-panel`, `.sky-panel`, and the background untouched so the surrounding ambience stays as-is. Remove/no longer rely on `.viewport-blend` for the tiles (it stays defined but unused).

### Outcome
The four viewports become clearly legible (phase screens, WFS spots, DM grid all readable at a glance) while the outer panel chrome, controls, and night sky retain the current low-opacity, integrated feel.