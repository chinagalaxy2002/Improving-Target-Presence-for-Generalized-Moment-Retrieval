import os
from pathlib import Path

KIT = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("GMR_PROJECT_ROOT", "/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval"))
REPO = Path("/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval")
REFERENCE = REPO / "ablations/end_to_end_geq_pv_20261026"
OUTPUT = PROJECT / "geq_epoch_study/20261031"
FIGURES = PROJECT / "newidea/figure/gmr_mr_detr_test_20261030/geq_finetune/epoch_study"
BACKBONES = ("moment_detr", "eatr", "qd_detr", "cg_detr")
MODES = ("full", "frozen")
SEED = 2023
MAX_EPOCHS = 20
