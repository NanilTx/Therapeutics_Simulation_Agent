from pathlib import Path
import os
DATA_DIR = Path(os.getenv("TSA_DATA_DIR", "./data")).resolve()
RANDOM_SEED = int(os.getenv("TSA_SEED", 1337))
LATENT_DIM = int(os.getenv("TSA_LATENT_DIM", 16))
