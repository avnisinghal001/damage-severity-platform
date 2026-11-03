import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { classColor } from "../palette";
import type { Inspection } from "../types";

interface Props {
  inspection: Inspection;
  onOpenDetails(): void;
}

const SEVERITY_ORDER = [
  { id: 3, label: "Undamaged", short: "Undam.", step: 0, tone: "#3a6b4c", fill: "#dce8df" },
  { id: 1, label: "Minor", short: "Minor", step: 1, tone: "#b58126", fill: "#ede4cf" },
  { id: 2, label: "Moderate", short: "Mod.", step: 2, tone: "#be5a26", fill: "#f0ddd2" },
  { id: 0, label: "Heavy", short: "Heavy", step: 3, tone: "#a33922", fill: "#edd5cf" },
] as const;

interface ColumnDatum {
  key: string;
  label: string;
  short: string;
  value: number;
  display: string;
  active: boolean;
  accent: string;
  fill: string;
}

interface BarShapeProps {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  background?: { x?: number; y?: number; width?: number; height?: number };
  payload?: ColumnDatum;
}

function ArchitecturalColumn(props: BarShapeProps) {
  const { x = 0, y = 0, width = 0, height = 0, payload } = props;
  if (width <= 0) return null;
  const barHeight = Math.max(8, height);
  const barY = y + height - barHeight;
  const active = payload?.active ?? false;
  const capColor = active ? (payload?.accent ?? "#a34326") : "#969e8f";
  const bodyFill = active ? (payload?.fill ?? "#f0ddd2") : "#e3e7dd";

  return (
    <g>
      <rect
        x={x}
        y={barY}
        width={width}
        height={barHeight}
        rx={3}
        fill={bodyFill}
      />
      {active && (
        <rect
          x={x}
          y={barY}
          width={width}
          height={barHeight}
          rx={3}
          fill="url(#concrete-hatch)"
          opacity={0.4}
        />
      )}
      <line
        x1={x + 1}
        y1={barY + 1.25}
        x2={x + width - 1}
        y2={barY + 1.25}
        stroke={capColor}
        strokeWidth={active ? 2.5 : 1.5}
        strokeLinecap="round"
      />
    </g>
  );
}

interface AxisTickProps {
  x?: number;
  y?: number;
  payload?: { value: string; index: number };
  data: ColumnDatum[];
}

function SeverityAxisTick({ x = 0, y = 0, payload, data }: AxisTickProps) {
  const item = data[payload?.index ?? 0];
  if (!item) return null;
  if (item.active) {
    return (
      <g transform={`translate(${x},${y + 6})`}>
        <rect x={-26} y={0} width={52} height={20} rx={5} fill="#1f2421" />
        <text
          x={0}
          y={13.5}
          textAnchor="middle"
          fill="#fafbf7"
          fontSize={10}
          fontWeight={500}
          fontFamily='"JetBrains Mono", "SFMono-Regular", Consolas, monospace'
        >
          {item.short}
        </text>
      </g>
    );
  }
  return (
    <g transform={`translate(${x},${y + 6})`}>
      <text
        x={0}
        y={13.5}
        textAnchor="middle"
        fill="#687066"
        fontSize={10}
        fontWeight={400}
        fontFamily='"JetBrains Mono", "SFMono-Regular", Consolas, monospace'
      >
        {item.short}
      </text>
    </g>
  );
}

export function ResultCard({
  inspection,
}: Props) {
  const severity = inspection.severity;
  const undamaged = severity?.status === "available" && severity.label === "Undamaged";

  const detected = useMemo(
    () =>
      inspection.analytics.classes
        .filter((row) => row.class_id > 0 && row.area_pixels > 0)
        .sort((a, b) => b.area_pixels - a.area_pixels),
    [inspection.analytics.classes]
  );

  const totalPixels = Math.max(1, inspection.width * inspection.height);
  const damagedPixels = useMemo(
    () => detected.reduce((sum, row) => sum + row.area_pixels, 0),
    [detected]
  );
  const coveragePct = undamaged ? 0 : (100 * damagedPixels) / totalPixels;
  

  const crackCoveragePct = useMemo(
    () =>
      detected
        .filter((row) => row.class_id === 1 || row.class_id === 2)
        .reduce((sum, row) => sum + row.image_area_percent, 0),
    [detected]
  );

  const crackSummary = useMemo(() => {
    if (inspection.classification) {
      return inspection.classification.label === "crack" ? "Crack detected" : "No crack detected";
    }
    return crackCoveragePct >= 0.25 ? "Crack detected" : "No crack detected";
  }, [inspection.classification, crackCoveragePct]);

  const hasCrack = crackSummary === "Crack detected";

  const activeMeta = useMemo(() => {
    if (severity?.status !== "available") {
      return { step: -1, tone: "#60655f", fill: "#e4e7de" };
    }
    return (
      SEVERITY_ORDER.find((item) => item.label === severity.label) ?? {
        step: 2,
        tone: "#be5a26",
        fill: "#f0ddd2",
      }
    );
  }, [severity]);

  const chartData = useMemo<ColumnDatum[]>(() => {
    if (severity?.status === "available") {
      return SEVERITY_ORDER.map((meta) => {
        const match = severity.probabilities.find(
          (p) => p.label === meta.label || p.class_id === meta.id
        );
        const pct = Math.round((match?.probability ?? 0) * 100);
        const active = severity.label === meta.label;
        return {
          key: meta.label,
          label: meta.label,
          short: meta.short,
          value: Math.max(6, pct),
          display: `${pct}%`,
          active,
          accent: meta.tone,
          fill: meta.fill,
        };
      });
    }
    const top = detected.slice(0, 4);
    if (top.length === 0) {
      return SEVERITY_ORDER.map((meta, idx) => ({
        key: meta.label,
        label: meta.label,
        short: meta.short,
        value: idx === 0 ? 100 : 6,
        display: idx === 0 ? "100%" : "0%",
        active: idx === 0,
        accent: meta.tone,
        fill: meta.fill,
      }));
    }
    return top.map((row, index) => {
      const pct = Math.round(row.image_area_percent);
      return {
        key: row.source_label,
        label: row.source_label,
        short: row.source_label.split(" ")[0].slice(0, 6),
        value: Math.max(6, pct),
        display: `${pct}%`,
        active: index === 0,
        accent: classColor(row.class_id),
        fill: "#f0ddd2",
      };
    });
  }, [severity, detected]);

  const confidencePct =
    severity?.status === "available" ? Math.round(severity.confidence * 100) : null;

  

  return (
    <section className="result-float" aria-label="Prediction summary">
      <div className="result-card-core">
        <div className="result-header">
          <span className="result-label-text">Damage level</span>
          <div className="result-header-controls">
            <div
              className="severity-pill"
              role="img"
              aria-label={
                severity?.status === "available"
                  ? `Severity scale: ${severity.label} (${activeMeta.step + 1} of 4)`
                  : "Severity scale unavailable"
              }
            >
              {SEVERITY_ORDER.map((stepItem) => {
                const isCurrent = activeMeta.step === stepItem.step;
                const isPassed = activeMeta.step >= stepItem.step && activeMeta.step >= 0;
                return (
                  <span
                    key={stepItem.label}
                    className={`severity-step ${isCurrent ? "is-current" : ""} ${isPassed ? "is-passed" : ""}`}
                    style={isCurrent ? { backgroundColor: stepItem.tone } : undefined}
                  />
                );
              })}
            </div>
          </div>
        </div>

        <div className="result-hero">
          <div className="result-hero-top">
            <h2 className={undamaged ? "is-undamaged" : ""}>
              {severity?.status === "available" ? severity.label : "Unavailable"}
            </h2>
            <strong
              className="focus-number"
              style={{ color: undamaged ? "#3a6b4c" : activeMeta.tone }}
              aria-label="Affected image area"
            >
              {coveragePct < 0.1 && coveragePct > 0
                ? "<0.1%"
                : `${coveragePct.toFixed(1)}%`}
            </strong>
          </div>
          <div className="result-hero-sub">
            <p className="crack-summary">
              <span
                className={`crack-dot ${hasCrack ? "is-crack" : "is-clear"}`}
                aria-hidden="true"
              />
              {crackSummary}
            </p>
            <span className="focus-caption">surface area</span>
          </div>
        </div>

        <div className="result-chart-block" aria-label="Severity probability distribution">
          <div className="chart-header">
            <span>Severity profile</span>
            {confidencePct !== null && (
              <span className="mono">{confidencePct}% peak score</span>
            )}
          </div>
          <div className="recharts-stage">
            <ResponsiveContainer width="100%" height={128}>
              <BarChart
                data={chartData}
                margin={{ top: 22, right: 2, left: 2, bottom: 4 }}
                barCategoryGap={6}
              >
                <defs>
                  <pattern
                    id="concrete-hatch"
                    width="6"
                    height="6"
                    patternTransform="rotate(45 0 0)"
                    patternUnits="userSpaceOnUse"
                  >
                    <line
                      x1="0"
                      y1="0"
                      x2="0"
                      y2="6"
                      stroke={activeMeta.tone}
                      strokeWidth="1"
                      strokeOpacity="0.35"
                    />
                  </pattern>
                </defs>
                <CartesianGrid
                  vertical={false}
                  stroke="#dfe4d8"
                  strokeDasharray="2 3"
                />
                <XAxis
                  dataKey="short"
                  axisLine={{ stroke: "#d5dccb", strokeWidth: 1 }}
                  tickLine={false}
                  interval={0}
                  height={30}
                  tick={<SeverityAxisTick data={chartData} />}
                />
                <YAxis hide domain={[0, 100]} />
                <Tooltip
                  cursor={{ fill: "rgba(39, 43, 40, 0.035)" }}
                  content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const item = payload[0].payload as ColumnDatum;
                    return (
                      <div className="chart-tooltip">
                        <span>{item.label}</span>
                        <strong>{item.display}</strong>
                      </div>
                    );
                  }}
                />
                <Bar
                  dataKey="value"
                  isAnimationActive={false}
                  background={{ fill: "transparent" }}
                  shape={<ArchitecturalColumn />}
                >
                  {chartData.map((entry) => (
                    <Cell key={entry.key} fill={entry.fill} />
                  ))}
                  <LabelList
                    dataKey="display"
                    position="top"
                    offset={7}
                    formatter={(value: unknown) => String(value ?? "")}
                    style={{
                      fontSize: "10px",
                      fontFamily: '"JetBrains Mono", "SFMono-Regular", Consolas, monospace',
                      fontVariantNumeric: "tabular-nums",
                      fill: "#222623",
                      fontWeight: 500,
                    }}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </section>
  );
}
