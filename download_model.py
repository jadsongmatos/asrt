#!/usr/bin/env python3
import os
import sys
from pathlib import Path

from pywhispercpp.model import Model

model_size = os.getenv("WHISPER_MODEL", "small")
models_dir = os.getenv("WHISPER_MODELS_DIR")

print(f"Downloading model {model_size} to {models_dir}...")

model = Model(model_size, models_dir=models_dir, download=True)

print(f"Model downloaded to: {Path(models_dir).expanduser()}")
