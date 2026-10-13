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
  by_grid: Record<string, LevelSummary>;
  class_tiers: Record<string, { count: number; mean_AP50: number | null; classes: string[] }>;
  per_class: { object_id: string; precision: number; recall: number; AP50: number; "AP50-95": number }[];
  errors: { wrong_id_pairs: Record<string, number>; missed_by_class: Record<string, number> };
  dataset: {
    images: Record<string, number>;
    objects: Record<string, number>;
    grids: Record<string, Record<string, number>>;
    min_instances_per_class: Record<string, number>;
    leakage_check: {
      "source_paths_in_2+_splits": number;
      "source_fingerprints_in_2+_splits": number;
      unique_source_images: Record<string, number>;
    };
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

export type Grid = "2x2" | "3x3" | "4x4" | "5x4" | "5x5";
export type Composite = { id: string; grid: Grid; width: number; height: number; src: string; labels: string };
export type Single = { src: string; box: [number, number, number, number] };

export const classes = classesJson as { detector: string[]; classifier: string[] };
export const metrics = metricsJson as unknown as Metrics;
export const training = trainingJson as TrainingRow[];
export const m1 = m1Json as unknown as M1;
export const samples = samplesJson as unknown as {
  composites: Composite[];
  singles: Record<string, Single[]>;
};

export const GRIDS: Grid[] = ["2x2", "3x3", "4x4", "5x4", "5x5"];
export const TILE = 224;
export const gridShape = (g: Grid) => {
  const [cols, rows] = g.split("x").map(Number);
  return { cols, rows };
};

export const objectIds = Object.keys(samples.singles);
export const photos: Record<string, string[]> = Object.fromEntries(
  Object.entries(samples.singles).map(([id, s]) => [id, s.map((x) => x.src)]),
);
export const thumb = (id: string) => photos[id]?.[0];
export const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;
