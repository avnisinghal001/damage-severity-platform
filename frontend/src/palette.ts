/** Exact project labels/RGB values from src/damage_analytics/labels.py, IDs 0–8. */
export const CLASSES = [
  { id: 0, label: "Background", rgb: [128, 128, 128] },
  { id: 1, label: "Microcrack", rgb: [230, 40, 40] },
  { id: 2, label: "Structural crack", rgb: [40, 180, 60] },
  { id: 3, label: "Minor spalling", rgb: [40, 210, 210] },
  { id: 4, label: "Moderate spalling", rgb: [130, 70, 170] },
  { id: 5, label: "Delamination", rgb: [245, 140, 30] },
  { id: 6, label: "Rebar corrosion", rgb: [240, 215, 50] },
  { id: 7, label: "Concrete crushing", rgb: [50, 90, 220] },
  { id: 8, label: "Structural deformation", rgb: [230, 60, 200] },
] as const;

export const classColor = (id: number): string => `rgb(${CLASSES[id].rgb.join(" ")})`;
