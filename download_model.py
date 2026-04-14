#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import pywhispercpp.utils as utils

model_size = os.getenv("WHISPER_MODEL", "small")
models_dir = os.getenv("WHISPER_MODELS_DIR")

print(f"Downloading model {model_size} to {models_dir}...")

model_path = utils.download_model(model_size, download_dir=models_dir)

print(f"Model downloaded to: {model_path}")
