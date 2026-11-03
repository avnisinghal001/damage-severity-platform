import { useState } from "react";

/** The photo inside a viewfinder while a scan line sweeps over it. Sized like the result viewer. */
export function Scanner({ src }: { src: string }) {
  const [ratio, setRatio] = useState<number | null>(null);
  return <div className="canvas-viewport scanner" role="status" aria-label="Scanning your photo">
    <div className={`scanner-frame ${ratio ? "is-ready" : ""}`} style={ratio ? { "--ratio": ratio } as React.CSSProperties : undefined}>
      <div className="scanner-clip">
        <img src={src} alt="" draggable={false}
          onLoad={(event) => setRatio(event.currentTarget.naturalWidth / event.currentTarget.naturalHeight)} />
        <i className="scanner-tint" />
        <i className="scanner-grid" />
        <i className="scanner-line" />
      </div>
      <b className="scanner-corner tl" /><b className="scanner-corner tr" />
      <b className="scanner-corner bl" /><b className="scanner-corner br" />
    </div>
  </div>;
}
