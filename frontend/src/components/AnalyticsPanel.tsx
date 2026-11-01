import { useMemo, useState } from "react";
import { humanize, integer, percent, score } from "../format";
import { classColor } from "../palette";
import type { DamageRegion, Inspection } from "../types";

interface Props {
  inspection: Inspection;
  selected: DamageRegion | null;
  onSelect(id: number | null): void;
}

const PAGE_SIZE = 10;

export function AnalyticsPanel({ inspection, selected, onSelect }: Props) {
  const [filter, setFilter] = useState("all");
  const [page, setPage] = useState(0);
  const detected = inspection.analytics.classes.filter((row) => row.class_id > 0 && row.area_pixels > 0);
  const area = detected.reduce((sum, row) => sum + row.area_pixels, 0);
  const connected = detected.reduce((sum, row) => sum + (row.connected_regions ?? 0), 0);
  const regions = useMemo(() => inspection.analytics.regions.filter((row) =>
    filter === "all" || row.class_id === Number(filter)), [inspection, filter]);
  const pageCount = Math.max(1, Math.ceil(regions.length / PAGE_SIZE));
  const visible = regions.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  return <div className="analytics-stack">
    <div className="stat-grid" aria-label="Image-space summary">
      <div className="panel stat"><span>Detected categories</span><strong>{detected.length}<small> / 8 damage classes</small></strong></div>
      <div className="panel stat"><span>Predicted image coverage</span><strong>{percent(100 * area / (inspection.width * inspection.height))}</strong>
        <small>{integer(area)} foreground pixels</small></div>
      <div className="panel stat"><span>Class-wise connected regions</span><strong>{integer(connected)}</strong>
        <small>Mask components, not verified physical instances</small></div>
    </div>
    <section className="panel analytics-panel" aria-labelledby="categories-title">
      <div className="section-header"><div><p className="eyebrow">Image-space measurements</p><h2 id="categories-title">Detected categories</h2></div>
        <span className="subtle-tag">IDs 1–8</span></div>
      {detected.length === 0 ? <p className="empty-message">No foreground damage pixels were predicted. This does not establish structural safety.</p>
        : <div className="table-scroll" tabIndex={0} aria-label="Scrollable category measurements"><table><caption className="sr-only">Predicted class areas and connected-region counts</caption>
          <thead><tr><th scope="col">Category</th><th scope="col">Image coverage</th><th scope="col">Pixels</th><th scope="col">Regions</th></tr></thead>
          <tbody>{detected.map((row) => <tr key={row.class_id}>
            <th scope="row"><span className="category-name"><span className="swatch" style={{ backgroundColor: classColor(row.class_id) }} aria-hidden="true" />{row.source_label}</span></th>
            <td><span className="coverage-value">{percent(row.image_area_percent)}</span><span className="coverage-track" aria-hidden="true">
              <span style={{ width: `${row.image_area_percent}%`, backgroundColor: classColor(row.class_id) }} /></span></td>
            <td className="numeric">{integer(row.area_pixels)}</td><td className="numeric">{integer(row.connected_regions ?? 0)}</td>
          </tr>)}</tbody></table></div>}
      <p className="field-help">Percentages divide by all image pixels, including background. They are not physical surface-area measurements.</p>
    </section>

    <section className="panel analytics-panel" aria-labelledby="regions-title">
      <div className="section-header"><div><p className="eyebrow">Inspect a prediction</p><h2 id="regions-title">Connected regions</h2></div>
        <div className="region-filter"><label htmlFor="region-class">Category</label>
          <select id="region-class" value={filter} onChange={(event) => { setFilter(event.target.value); setPage(0); onSelect(null); }}>
            <option value="all">All detected categories</option>
            {detected.map((row) => <option key={row.class_id} value={row.class_id}>{row.source_label}</option>)}
          </select></div></div>
      <p className="muted">Select a reported region to highlight its bounding box in the viewer.</p>
      {regions.length === 0 ? <p className="empty-message">No reported regions match this selection.</p>
        : <><div className="table-scroll" tabIndex={0} aria-label="Scrollable connected regions"><table className="region-table"><caption className="sr-only">Select a region and inspect its native pixel bounding box</caption>
          <thead><tr><th scope="col">Region</th><th scope="col">Category</th><th scope="col">Coverage</th><th scope="col">Bounding box (px)</th><th scope="col">Review</th></tr></thead>
          <tbody>{visible.map((row) => <tr key={row.region_id} className={selected?.region_id === row.region_id ? "selected-row" : ""}>
            <th scope="row"><button type="button" className="region-button" aria-pressed={selected?.region_id === row.region_id}
              aria-label={`Highlight region ${row.region_id}, ${row.source_label}`} onClick={() => onSelect(row.region_id)}>#{row.region_id}</button></th>
            <td>{row.source_label}</td><td>{percent(row.image_area_percent)}</td>
            <td className="mono small">[{row.bbox_xyxy_exclusive.slice(0, 2).join(", ")}] → [{row.bbox_xyxy_exclusive.slice(2).join(", ")}]</td>
            <td><span className="small">{row.uncertainty.review_flags.length ? "Review requested"
              : row.uncertainty.status === "unknown_no_model_scores" ? "Scores unavailable" : "Uncalibrated"}</span></td>
          </tr>)}</tbody></table></div>
          <div className="pagination"><span className="muted small">{integer(regions.length)} reported regions · page {page + 1} of {pageCount}</span>
            <div><button type="button" className="button secondary compact" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button>
              <button type="button" className="button secondary compact" disabled={page + 1 >= pageCount} onClick={() => setPage(page + 1)}>Next</button></div></div></>}
      <p className="field-help">Boxes use native pixel coordinates [x₀, y₀, x₁, y₁] with exclusive maximum coordinates. Region reporting filters do not change class-area totals.</p>

      {selected && <section className="region-detail" aria-labelledby="selected-region-title">
        <div className="section-header"><h3 id="selected-region-title" aria-live="polite">Region #{selected.region_id} · {selected.source_label}</h3>
          <button type="button" className="text-button" onClick={() => onSelect(null)}>Clear selection</button></div>
        <dl className="detail-grid">
          <div><dt>Area in image pixels</dt><dd>{integer(selected.area_pixels)} px · {percent(selected.image_area_percent)}</dd></div>
          <div><dt>Touches image boundary</dt><dd>{selected.touches_image_boundary ? "Yes" : "No"}</dd></div>
          <div><dt>Mean maximum softmax · uncalibrated</dt><dd>{score(selected.uncertainty.mean_maximum_softmax)}</dd></div>
          <div><dt>Mean normalized entropy · uncalibrated</dt><dd>{score(selected.uncertainty.mean_normalized_entropy)}</dd></div>
        </dl>
        <p className="small">Score status: <span className="status-words">{humanize(selected.uncertainty.status)}</span>. These values are not calibrated probabilities of a correct prediction.</p>
        {selected.uncertainty.review_flags.length > 0 && <ul className="review-flags">
          {selected.uncertainty.review_flags.map((flag) => <li key={flag}>{humanize(flag)}</li>)}</ul>}
        {selected.spalling_assessment.label && <p className="small">Source-category label: <strong>{selected.spalling_assessment.label}</strong> (unvalidated). This is inherited from segmentation, not an independent severity assessment.</p>}
      </section>}
    </section>
    <section className="scope-note" aria-label="Scientific interpretation">
      <strong>Interpret predictions within their scope.</strong>
      <p>Model confidence is uncalibrated. Image coverage is not physical damage area. Structural severity and safety are not assessed.</p>
      <p className="small">Uncertainty availability: {humanize(inspection.analytics.uncertainty.status)}.</p>
      <details><summary>Measurement limitations</summary><ul>
        {inspection.analytics.limitations.map((item) => <li key={item}>{item}</li>)}
      </ul></details>
    </section>
  </div>;
}
