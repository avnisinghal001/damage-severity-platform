export const integer = (value: number): string => value.toLocaleString();
export const percent = (value: number): string =>
  value > 0 && value < 0.001 ? "<0.001%" : `${value.toLocaleString(undefined, { maximumFractionDigits: 3 })}%`;
export const humanize = (value: string): string => value.replaceAll("_", " ");
export const score = (value: number | null | undefined): string =>
  typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "Unavailable";
