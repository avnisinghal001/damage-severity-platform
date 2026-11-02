import { percent } from "../format";
import type { ClassifierProvenance, Inspection } from "../types";

function Provenance({ model }: { model: ClassifierProvenance }) {
  return <details className="small"><summary>Model provenance and limitations</summary>
    <p>ResNet18 · epoch {model.epoch} · {model.device.toUpperCase()}</p>
    <code className="hash-wrap">{model.checkpoint_sha256}</code>
    <p>{model.dataset.release_id || "Release not specified"}</p>
    <ul>{model.limitations.map((item) => <li key={item}>{item}</li>)}</ul>
  </details>;
}

export function ClassifierResults({ inspection }: { inspection: Inspection }) {
  const crack = inspection.classification, severity = inspection.severity;
  return <section className="panel analytics-panel" aria-labelledby="classification-title">
    <p className="eyebrow">Separate image-based predictions</p>
    <h2 id="classification-title">Crack presence & component damage level</h2>
    <div className="classifier-grid">
      <div><h3>Crack classification</h3>
        {crack ? <>
          <p className="classification-label">{crack.label === "crack" ? "Crack predicted" : "No crack predicted"}</p>
          <p>Crack softmax: {percent(crack.crack_probability * 100)} · <strong>uncalibrated</strong></p>
          <p className="small">Frozen validation threshold: {crack.decision_threshold.toFixed(4)}.
            No-crack is not an undamaged-concrete or safety determination.</p>
          <Provenance model={crack.model} />
        </> : <p>Not provided for this saved inspection.</p>}
      </div>
      <div><h3>Φ-Net component damage level</h3>
        {severity?.status === "available" ? <>
          <p className="classification-label">{severity.label} <span className="small">(source ID {severity.class_id})</span></p>
          <p>Maximum softmax: {percent(severity.confidence * 100)} · <strong>uncalibrated</strong></p>
          <dl className="probability-list">{severity.probabilities.map((p) => <div key={p.class_id}>
            <dt>{p.label}</dt><dd>{percent(p.probability * 100)}</dd></div>)}</dl>
          <p className="small">An independent image classification, not a grade derived from segmentation area.</p>
          <Provenance model={severity.model} />
        </> : <p role="status">{severity?.status === "unavailable" ? severity.message
          : "Not provided for this saved inspection."}</p>}
      </div>
    </div>
    <p className="field-help">The input's domain and component framing are not automatically verified.
      These models do not assess whole-building severity, load capacity or structural safety.</p>
  </section>;
}
