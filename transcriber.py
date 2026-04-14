from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from pywhispercpp.model import Model


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration: float
    segments: list


class Transcriber:
    def __init__(
        self,
        model_size: str = "small",
        n_threads: int = 4,
        models_dir: str | None = None,
    ):
        self.model_size = model_size
        self.n_threads = n_threads
        self.model = Model(
            model_size,
            models_dir=models_dir,
            n_threads=n_threads,
            print_realtime=False,
        )
        self._lock = threading.Lock()

    def transcribe(
        self,
        audio_path: Union[str, np.ndarray],
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        temperature: float = 0.0,
        task: str = "transcribe",
        speed_up: bool = False,
    ) -> TranscriptionResult:
        with self._lock:
            params = {}
            if language:
                params["language"] = language
            if initial_prompt:
                params["initial_prompt"] = initial_prompt
            if temperature:
                params["temperature"] = temperature
            if task == "translate":
                params["translate"] = True
            if speed_up:
                params["speed_up"] = True

            segments_iter = self.model.transcribe(audio_path, **params)
            segments = []
            duration = 0.0

            for seg in segments_iter:
                segments.append(
                    Segment(
                        start=seg.t0 / 1000.0,
                        end=seg.t1 / 1000.0,
                        text=seg.text,
                    )
                )
                duration = seg.t1 / 1000.0

            text = "".join(s.text for s in segments).strip()
            detected_lang = language if language else "en"

            return TranscriptionResult(
                text=text,
                language=detected_lang,
                duration=duration,
                segments=segments,
            )
