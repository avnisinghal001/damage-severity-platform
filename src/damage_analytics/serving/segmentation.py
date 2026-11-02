"""Task 09 LR-ASPP inference plus an explicit compatibility view for the Task 06 dashboard."""

from dataclasses import asdict

import torch
from PIL import Image

from damage_analytics.analytics.regions import spalling_assessment
from damage_analytics.api.config import Settings
from damage_analytics.api.inference import Prediction
from damage_analytics.labels import CLASS_NAMES
from damage_analytics.serving.classifiers import CrackClassifier, check_pin
from damage_analytics.task09.config import FeaturePolicy
from damage_analytics.task09.pipeline import Segmenter


class DatasetPin:
    """Checkpoint-embedded training identity; serving never loads or evaluates a dataset."""

    def __init__(self, identity: dict):
        self._identity = identity

    def identity(self) -> dict:
        return self._identity


def compatibility_analytics(report: dict, policy: FeaturePolicy) -> dict:
    """Translate existing pixel measurements without recomputing regions or changing masks."""
    width, height = report["image"]["width"], report["image"]["height"]
    counts = report["coverage"]["raw_class_pixels"]
    classes = [
        {
            "class_id": 0,
            "source_label": CLASS_NAMES[0],
            "area_pixels": counts[0],
            "image_area_percent": 100 * counts[0] / (width * height),
            "connected_regions": None,
        }
    ]
    classes += [
        {
            "class_id": c["class_id"],
            "source_label": c["label"],
            "area_pixels": c["raw_pixels"],
            "image_area_percent": c["image_percent"],
            "connected_regions": c["raw_region_count"],
        }
        for c in report["classes"]
    ]
    regions = []
    for number, row in enumerate(report["regions"], 1):
        scores, flags = row["score_statistics"], []
        if scores["available"]:
            if scores["mean_maximum_softmax"] < policy.maximum_softmax_below:
                flags.append("low_uncalibrated_maximum_softmax")
            if scores["mean_normalized_entropy"] > policy.normalized_entropy_above:
                flags.append("high_uncalibrated_entropy")
        box = row["bbox_xyxy"]
        regions.append(
            {
                "region_id": number,
                "class_id": row["class_id"],
                "source_label": row["label"],
                "area_pixels": row["area_pixels"],
                "image_area_percent": row["image_percent"],
                "bbox_xyxy_exclusive": box,
                "centroid_xy_pixel_index": [v - 0.5 for v in row["centroid_xy"]],
                "touches_image_boundary": box[0] == 0
                or box[1] == 0
                or box[2] == width
                or box[3] == height,
                "uncertainty": {
                    "calibrated": False,
                    "status": "review_requested"
                    if flags
                    else "uncalibrated_scores_available"
                    if scores["available"]
                    else "unknown_no_model_scores",
                    "mean_maximum_softmax": scores["mean_maximum_softmax"],
                    "mean_normalized_entropy": scores["mean_normalized_entropy"],
                    "review_flags": flags,
                },
                "spalling_assessment": spalling_assessment(
                    [row["class_id"]], uncertain_prediction=bool(flags)
                ),
            }
        )
    return {
        "schema_version": 1,
        "measurement_space": "image_pixels",
        "image": {"width": width, "height": height, "area_pixels": width * height},
        "classes": classes,
        "regions": regions,
        "uncertainty": {"calibrated": False, "status": "uncalibrated_softmax_and_entropy_only"},
        "spalling_summary": spalling_assessment(
            [i for i in range(1, 9) if counts[i]], uncertain_prediction=True
        ),
        "overall_structural_severity": {"status": "not_assessed", "score": None},
        "limitations": [
            "Coverage uses raw segmentation pixels; reporting filters do not change coverage.",
            "Connected regions are not physical instances; measurements use image pixels.",
            "Segmentation categories are not Phi-Net labels or whole-building safety scores.",
            "Confidence is uncalibrated; arbitrary photographs are not a validated input domain.",
        ],
    }


class IntegratedPredictor:
    """One LR-ASPP and one crack ResNet18 per worker, reused for all admitted requests."""

    def __init__(self, settings: Settings):
        check_pin(settings.checkpoint, settings.checkpoint_sha256)
        payload = torch.load(settings.checkpoint, map_location="cpu", weights_only=True)
        meta = payload["metadata"]
        if (
            payload.get("task09_format") != 1
            or meta["purpose"] != "training"
            or meta["model_name"] != "lraspp_mobilenet_v3_large"
            or meta["class_names"] != list(CLASS_NAMES)
        ):
            raise ValueError("Integrated backend requires the selected trained Task 09 LR-ASPP")
        self.segmenter = Segmenter(
            settings.checkpoint,
            DatasetPin(meta["dataset"]),
            device=settings.device,
            cpu_threads=settings.cpu_threads,
            maximum_image_pixels=settings.max_pixels,
        )
        self.segmenter.model.requires_grad_(False)
        check_pin(settings.checkpoint, settings.checkpoint_sha256)
        self.crack = CrackClassifier(
            settings.crack_checkpoint, settings.crack_checkpoint_sha256, self.segmenter.device
        )
        self.policy = FeaturePolicy()  # No artifact removal: retain every predicted pixel/region.
        self.metadata = {
            "checkpoint_sha256": settings.checkpoint_sha256,
            "checkpoint_epoch": payload["epoch"],
            "dataset": meta["dataset"],
            "architecture": {
                "name": meta["model_name"],
                "freeze_batch_norm": meta["freeze_batch_norm"],
            },
            "preprocessing": asdict(self.segmenter.data),
            "class_mapping": [
                {"class_id": i, "source_label": name} for i, name in enumerate(CLASS_NAMES)
            ],
            "device": str(self.segmenter.device),
            "training_code_version": meta.get("execution", {}).get("git_commit"),
            "confidence": "Uncalibrated softmax and normalized entropy",
            "alignment": "EXIF-oriented resolution; argmax, unpad, nearest-neighbor restoration",
            "severity_scope": "Independent Phi-Net HTTP image prediction; no paired features",
        }

    def predict(self, image: Image.Image) -> Prediction:
        report, masks, oriented = self.segmenter.predict(image, self.policy)
        if oriented.size != image.size:
            raise ValueError("The backend must orient and sanitize the upload before inference")
        return Prediction(
            masks["raw"],
            compatibility_analytics(report, self.policy),
            features=report,
            crack=self.crack.predict(oriented),
        )
