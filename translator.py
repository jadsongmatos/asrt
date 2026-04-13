from __future__ import annotations

import logging
import threading

import argostranslate.package
import argostranslate.translate

log = logging.getLogger("legenda.translator")


class Translator:
    """Thin wrapper around Argos Translate.

    Installs language pair packages on demand (first use is slow because
    the package has to be downloaded). Falls back to pivot through English
    when a direct package is not available.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._installed: set[tuple[str, str]] = set()
        self._index_ready = False

    def _refresh_index(self) -> None:
        if self._index_ready:
            return
        argostranslate.package.update_package_index()
        self._index_ready = True

    def _install_pair(self, src: str, tgt: str) -> bool:
        if src == tgt or (src, tgt) in self._installed:
            return True
        self._refresh_index()
        available = argostranslate.package.get_available_packages()
        match = next(
            (p for p in available if p.from_code == src and p.to_code == tgt),
            None,
        )
        if match is None:
            return False
        log.info("downloading argos package %s->%s", src, tgt)
        argostranslate.package.install_from_path(match.download())
        self._installed.add((src, tgt))
        return True

    def ensure(self, src: str, tgt: str) -> bool:
        """Guarantees packages are installed to translate `src` -> `tgt`."""
        if src == tgt:
            return True
        with self._lock:
            if self._install_pair(src, tgt):
                return True
            if src != "en" and tgt != "en":
                if self._install_pair(src, "en") and self._install_pair("en", tgt):
                    return True
            return False

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        text = text.strip()
        if not text or src_lang == tgt_lang:
            return text
        if not self.ensure(src_lang, tgt_lang):
            raise RuntimeError(
                f"no argos translate package available for {src_lang}->{tgt_lang}"
            )
        return argostranslate.translate.translate(text, src_lang, tgt_lang)
