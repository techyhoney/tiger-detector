"""Central configuration for the tiger classifier (Stage 2 of the pipeline).

Everything the rest of the code needs is here. To go from the current
binary (tiger vs not_tiger) model to a full multi-class species model
later, you only edit CLASSES and FOLDER_TO_CLASS below -- no other file
needs to change.
"""
from pathlib import Path

# --- Paths -------------------------------------------------------------
# Project dir = folder containing this file. Data lives one level up
# (the dataset_crops folder with tiger/ leopard/ elephant/ negative/).
PROJECT_DIR = Path(__file__).resolve().parent
# Class folders live in dataset_crops/dataset/{tiger,leopard,elephant,negative}.
# Override with the TIGER_DATA_ROOT env var if you move the data.
DATA_ROOT = Path(
    __import__("os").environ.get("TIGER_DATA_ROOT", PROJECT_DIR.parent / "dataset")
)
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CKPT_PATH = OUTPUT_DIR / "tiger_classifier.pt"   # best checkpoint
ONNX_PATH = OUTPUT_DIR / "tiger_classifier.onnx"

# --- Classes -----------------------------------------------------------
# Build: 4-CLASS HEAD, tiger-only decision. The model learns all four
# classes (cleaner boundary + a running start on multi-species), but at
# inference we only threshold on P(tiger). Index order = output order.
CLASSES = ["negative", "tiger", "leopard", "elephant"]
POSITIVE_CLASS = "tiger"                 # the class we ultimately care about

# Which source folders map into which class label. Each species is its
# own class now. To collapse back to binary, point leopard/elephant/
# negative all at a single "not_tiger" label instead.
FOLDER_TO_CLASS = {
    "tiger":    "tiger",
    "leopard":  "leopard",
    "elephant": "elephant",
    "negative": "negative",
}

# Only these subfolders of DATA_ROOT are scanned as data. Anything else
# (this tiger_detector/ folder, .DS_Store, etc.) is ignored.
SOURCE_FOLDERS = list(FOLDER_TO_CLASS.keys())

# --- Image / normalization --------------------------------------------
IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]        # ImageNet stats (pretrained backbone)
NORM_STD = [0.229, 0.224, 0.225]
LETTERBOX_FILL = 114                      # gray pad, YOLO-style

# --- Model -------------------------------------------------------------
MODEL_NAME = "mobilenetv3_large_100"      # timm id; Pi-5 friendly (~12 ms/crop)
# Fallbacks if accuracy is short: "tf_efficientnet_lite0", "efficientnet_b0"

# --- Training ----------------------------------------------------------
SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15
BATCH_SIZE = 64
NUM_WORKERS = 4

FREEZE_EPOCHS = 3        # phase 1: train head only, backbone frozen
FINETUNE_EPOCHS = 27     # phase 2: unfreeze and fine-tune everything
HEAD_LR = 1e-3
FINETUNE_LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 8  # stop if val macro-F1 doesn't improve for N epochs

# --- Inference ---------------------------------------------------------
# Default decision threshold on P(tiger). Tune on val in evaluate.py.
DECISION_THRESHOLD = 0.5


def class_to_idx() -> dict:
    return {c: i for i, c in enumerate(CLASSES)}


def positive_idx() -> int:
    return CLASSES.index(POSITIVE_CLASS)
