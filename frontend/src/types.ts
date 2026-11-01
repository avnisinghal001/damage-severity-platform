/** The fields consumed from Task 05 and the embedded Task 04 analytics schema. */
export type ClassId = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8;
export type BoundingBox = [number, number, number, number];

export interface ClassArea {
  class_id: ClassId;
  source_label: string;
  area_pixels: number;
  image_area_percent: number;
  connected_regions: number | null;
}

export interface RegionUncertainty {
  status: string;
  calibrated: boolean;
  mean_maximum_softmax: number | null;
  mean_normalized_entropy: number | null;
  review_flags: string[];
}

export interface SpallingAssessment {
  status: string;
  label: string | null;
  candidate_source_labels: string[];
  unsupported_class_ids: number[];
  supervision: string;
  structural_severity_or_safety: string;
}

export interface DamageRegion {
  region_id: number;
  class_id: ClassId;
  source_label: string;
  area_pixels: number;
  image_area_percent: number;
  bbox_xyxy_exclusive: BoundingBox;
  centroid_xy_pixel_index: [number, number];
  touches_image_boundary: boolean;
  uncertainty: RegionUncertainty;
  spalling_assessment: SpallingAssessment;
}

export interface ModelMetadata {
  checkpoint_sha256: string;
  checkpoint_epoch: number;
  dataset: { release_id?: string };
  device: "cpu" | "mps";
  class_mapping: { class_id: ClassId; source_label: string }[];
  alignment: string;
  confidence: string;
  severity_scope: string;
}

export interface Inspection {
  id: string;
  created_at: string;
  status: "completed";
  width: number;
  height: number;
  image: { url: string; sha256: string; media_type: "image/png" };
  mask: { url: string; sha256: string; media_type: "image/png" };
  detected_classes: ClassArea[];
  model: ModelMetadata;
  analytics: {
    schema_version: number;
    measurement_space: "image_pixels";
    image: { width: number; height: number; area_pixels: number };
    classes: ClassArea[];
    regions: DamageRegion[];
    uncertainty: { calibrated: boolean; status: string };
    spalling_summary: SpallingAssessment;
    limitations: string[];
  };
}

export interface Readiness {
  status: "ready" | "not_ready";
  model: boolean;
  database: boolean;
}
