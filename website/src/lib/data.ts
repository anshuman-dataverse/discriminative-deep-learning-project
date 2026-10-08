// Results and samples exported by milestone2/scripts/05_export_web.py.
import classesJson from "@/data/classes.json";
import m1Json from "@/data/m1.json";
import metricsJson from "@/data/metrics.json";
import samplesJson from "@/data/samples.json";
import trainingJson from "@/data/training.json";

export type LevelSummary = {
  images: number;
  objects: number;
  objects_correct_id_and_location: number;
  object_accuracy: number;
  wrong_id: number;
  missed: number;
  false_alarms: number;
  images_fully_correct: number;
  image_accuracy: number;
  mean_iou_correct: number;
};

export type Metrics = {
  test_images: number;
  mAP50: number;
  "mAP50-95": number;
  precision: number;
  recall: number;
  inference_ms_per_image: number;
  best_f1: { confidence: number; f1: number } | null;
  "at_conf_0.5": LevelSummary;
  by_layout: Record<string, LevelSummary>;
  class_tiers: Record<string, { count: number; mean_AP50: number | null; classes: string[] }>;
  per_class: { object_id: string; precision: number; recall: number; AP50: number; "AP50-95": number }[];
  errors: { wrong_id_pairs: Record<string, number>; missed_by_class: Record<string, number> };
  dataset: {
    images: Record<string, number>;
    objects: Record<string, number>;
    layouts: Record<string, number>;
    min_instances_per_class: Record<string, number>;
    leakage_check: Record<string, number | Record<string, number>>;
    label_validation: { images_checked: number; boxes_checked: number; invalid: number };
  };
};

export type TrainingRow = Record<string, number>;

export type M1 = {
  best_model: string;
  num_classes: number;
  models: {
    model: string;
    params_M: number;
    best_val_acc: number;
    test_acc: number;
    test_top5: number;
    test_macro_f1: number;
    train_time_min: number;
    inference_ms_per_img: number;
  }[];
  per_class_f1: Record<string, number>;
};

export type Composite = { id: string; layout: string; src: string; labels: string };

export const classes = classesJson as { detector: string[]; classifier: string[] };
export const metrics = metricsJson as unknown as Metrics;
export const training = trainingJson as TrainingRow[];
export const m1 = m1Json as unknown as M1;
export const samples = samplesJson as {
  composites: Composite[];
  singles: Record<string, string[]>;
  backgrounds: string[];
};

export const objectIds = Object.keys(samples.singles);
export const thumb = (id: string) => samples.singles[id]?.[0];
export const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;
