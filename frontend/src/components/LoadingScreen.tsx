import { useEffect, useState } from "react";
import type { ProgressMap, Role } from "../local/engine";

const CREW: { role: Role; name: string; calling: string; coming: string; ready: string }[] = [
  { role: "segmentation", name: "Builders", calling: "Calling the builders", coming: "On their way", ready: "On site" },
  { role: "severity", name: "Engineers", calling: "Waking up the engineers", coming: "Getting dressed", ready: "At their desks" },
  { role: "crack", name: "Inspectors", calling: "Briefing the inspectors", coming: "Lacing up boots", ready: "Ready to inspect" },
];

const LINES = [
  "Calling the builders",
  "Waking up the engineers",
  "Briefing the inspectors",
  "Mixing a fresh batch",
  "Checking the plumb line",
  "Setting out the formwork",
  "Hanging the safety ropes",
  "Sharpening the pencils",
];

interface Props {
  progress: ProgressMap;
  ready: boolean;
  error: string | null;
  onRetry(): void;
  onGone(): void;
}

export function LoadingScreen({ progress, ready, error, onRetry, onGone }: Props) {
  const [line, setLine] = useState(0);
  const [leaving, setLeaving] = useState(false);

  useEffect(() => {
    if (ready || error) return;
    const id = window.setInterval(() => setLine((value) => (value + 1) % LINES.length), 2200);
    return () => window.clearInterval(id);
  }, [ready, error]);

  useEffect(() => {
    if (!ready) return;
    const fade = window.setTimeout(() => setLeaving(true), 450);
    const gone = window.setTimeout(onGone, 450 + 520);
    return () => { window.clearTimeout(fade); window.clearTimeout(gone); };
  }, [ready, onGone]);

  const arrived = CREW.filter((member) => progress[member.role].done).length;

  return <div className={`loader ${leaving ? "is-leaving" : ""}`} role="status" aria-live="polite">
    <div className="loader-grid" aria-hidden="true" />
    <div className="loader-inner">
      <span className="brand-mark loader-mark" aria-hidden="true">C<span>09</span></span>
      {error ? <>
        <p className="loader-line">The crew couldn't get to site</p>
        <p className="loader-sub">Please try again in a moment.</p><code className="loader-detail">{error.slice(0, 140)}</code>
        <button type="button" className="button primary" onClick={onRetry}>Call them again</button>
      </> : <>
        <p className="loader-line" key={ready ? "done" : line}>{ready ? "Everyone is on site" : LINES[line]}</p>
        <ul className="crew">
          {CREW.map((member) => {
            const item = progress[member.role];
            const fraction = item.total ? item.loaded / item.total : 0;
            const status = item.done ? member.ready : fraction > 0.02 ? member.coming : member.calling;
            return <li key={member.role} className={item.done ? "is-ready" : ""}>
              <span className="crew-name">{member.name}</span>
              <span className="crew-track" aria-hidden="true"><i style={{ transform: `scaleX(${item.done ? 1 : fraction})` }} /></span>
              <span className="crew-status">{status}</span>
            </li>;
          })}
        </ul>
        <p className="loader-foot">{arrived} of {CREW.length} on site</p>
      </>}
    </div>
  </div>;
}
