"""Скачивает исходное демо-видео цеха и собирает из него `production.mp4`.

Зачем: исходное видео большое и лежит в .gitignore, поэтому при клоне репозитория
(или переезде на другую машину/ОС) его нет, и `main.py` запустить нечем. Этот
скрипт восстанавливает исходник из открытых стоковых клипов — без ручного поиска.

Источник: Pexels (бесплатная лицензия, коммерческое использование разрешено,
атрибуция не требуется). Клипы подобраны под задачу: сырое мясо занимает центр
кадра, есть рабочие в спецодежде и оборудование — то есть фон, который детектор
обязан игнорировать.

Результат: `production.mp4`, 1920x1080, 25 fps, ~96 с (несколько проходов по
склейке, чтобы генератор дефектов расставил РАЗНЫЕ эпизоды в каждом проходе).

Запуск:
    python fetch_demo_source.py
    python create_realistic_defects.py    # дальше как обычно
"""
import os
import sys

import cv2
import requests

# id клипа на Pexels -> имя файла на CDN
CLIPS = [
    (8551710, "8551710-uhd_2560_1440_25fps.mp4"),   # разделка туши на столе, спецодежда
    (8723086, "8723086-uhd_2560_1440_25fps.mp4"),   # распил на ленточной пиле, маска/перчатки
    (13689780, "13689780-hd_1920_1080_25fps.mp4"),  # рубка рёбер на колоде
]
BASE_URL = "https://videos.pexels.com/video-files"

CACHE_DIR = "_source_clips"
OUTPUT = "production.mp4"
WIDTH, HEIGHT, FPS = 1920, 1080, 25
PASSES = 3          # сколько раз повторить склейку (даёт разные дефект-эпизоды)


def download(clip_id: int, filename: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{clip_id}.mp4")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"  {clip_id}: уже скачан ({os.path.getsize(path) / 1e6:.1f} МБ)")
        return path
    url = f"{BASE_URL}/{clip_id}/{filename}"
    print(f"  {clip_id}: качаю {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"  {clip_id}: готово ({os.path.getsize(path) / 1e6:.1f} МБ)")
    return path


def main():
    print("1/2 Скачиваю исходные клипы...")
    paths = []
    for clip_id, filename in CLIPS:
        try:
            paths.append(download(clip_id, filename))
        except Exception as e:
            print(f"  ОШИБКА {clip_id}: {e}")
    if not paths:
        print("Не удалось скачать ни одного клипа. Проверь интернет/прокси.")
        return 1

    print(f"2/2 Собираю {OUTPUT} ({WIDTH}x{HEIGHT}, {FPS} fps, {PASSES} прохода)...")
    writer = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    total = 0
    for _ in range(PASSES):
        for path in paths:
            cap = cv2.VideoCapture(path)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if (frame.shape[1], frame.shape[0]) != (WIDTH, HEIGHT):
                    frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                total += 1
            cap.release()
    writer.release()
    print(f"Готово: {OUTPUT}, {total} кадров ({total / FPS:.0f} с)")
    print("Дальше: python create_realistic_defects.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
