import json
import os
from .logger import setup_logger

logger = setup_logger(name="DefectManifest")


class DefectManifest:
    """Ground-truth манифест дефектов, сгенерированный create_realistic_defects.py.

    Заменяет live-VLM в демо-режиме: детекция = O(1) lookup эпизода по индексу
    кадра. Никакого лага инференса — бокс рисуется в тот же кадр, где дефект
    появился на видео. Эпизоды отсортированы по start, поиск линейный (их ~16).
    """

    def __init__(self, path):
        self.path = path
        self.episodes = []
        self.meta = {}
        self.loaded = False
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.episodes = sorted(data.get("episodes", []), key=lambda e: e["start"])
                self.meta = {k: v for k, v in data.items() if k != "episodes"}
                self.loaded = True
                logger.info(f"Loaded {len(self.episodes)} defect episodes from {path}")
            except Exception as e:
                logger.error(f"Failed to load manifest {path}: {e}")
        else:
            logger.warning(f"Manifest not found: {path}. Detection disabled.")

    def lookup(self, frame_idx):
        """Вернуть эпизод дефекта, активный на данном индексе кадра, иначе None."""
        for ep in self.episodes:
            if ep["start"] <= frame_idx <= ep["end"]:
                return ep
            if ep["start"] > frame_idx:
                break  # эпизоды отсортированы — дальше только позже
        return None
