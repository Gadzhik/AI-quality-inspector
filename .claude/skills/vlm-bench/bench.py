"""Замер vision-модели LM Studio на задаче проекта.

Берёт кадры демо-видео по ground-truth манифесту (дефектные — из середины
эпизодов, чистые — между ними), прогоняет через боевой путь `src/analyzer.py`
+ `src/box_refine.py` и печатает detection/false alarms/время.

Мерит именно ДЕТЕКЦИЮ: модель, которая быстро отвечает «чисто» на всё,
бесполезна, а по одному latency этого не видно.

Важно: здесь НЕТ гашения тревоги по `box_refine.on_meat`, которое применяет
`main_vlm.py`. Это сделано нарочно — замер показывает сырое поведение модели,
а не поведение системы после предохранителей. Поэтому ложных тревог тут будет
больше, чем в живом режиме.

Запуск (из корня проекта):
    python .claude/skills/vlm-bench/bench.py qwen/qwen3-vl-8b
    python .claude/skills/vlm-bench/bench.py google/gemma-4-12b-qat --limit 6
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import cv2  # noqa: E402

from src import box_refine  # noqa: E402
from src.analyzer import AIAnalyzer  # noqa: E402

VIDEO = "production_with_realistic_defects.mp4"
MANIFEST = "defects_manifest.json"


def to_pixels(box, w, h):
    """Бокс модели -> пиксели кадра. Шкала по максимальной координате."""
    if not box or len(box) != 4:
        return None
    ymin, xmin, ymax, xmax = box
    m = max(box)
    if m <= 1.0:
        sx, sy = w, h
    else:
        denom = 448.0 if m <= 448 else 1000.0
        sx, sy = w / denom, h / denom
    return [int(xmin * sx), int(ymin * sy), int(xmax * sx), int(ymax * sy)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=None, help="id модели в LM Studio")
    ap.add_argument("--limit", type=int, default=0, help="сколько эпизодов брать")
    ap.add_argument("--no-refine", action="store_true", help="без уточнения рамки")
    args = ap.parse_args()

    if not os.path.exists(MANIFEST) or not os.path.exists(VIDEO):
        print("Нет демо-видео или манифеста. Сначала:")
        print("  python fetch_demo_source.py && python create_realistic_defects.py")
        return 1

    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    episodes = manifest["episodes"]
    if args.limit:
        episodes = episodes[:args.limit]

    analyzer = AIAnalyzer(model_name=args.model) if args.model else AIAnalyzer()
    if not analyzer.ping():
        print("LM Studio не отвечает на localhost:1234 — запусти `lms server start`.")
        return 1
    print(f"модель: {analyzer.model_name}\n")

    cap = cv2.VideoCapture(VIDEO)
    scenes = [("defect", (e["start"] + e["end"]) // 2, e) for e in episodes]
    prev = 0
    for e in episodes:
        mid = (prev + e["start"]) // 2
        if mid > prev + 5:
            scenes.append(("clean", mid, None))
        prev = e["end"]

    det = hit = fp = nd = nc = 0
    lats = []
    for kind, idx, ep in scenes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        t = time.perf_counter()
        res = analyzer.analyze_frame(frame)
        lats.append(time.perf_counter() - t)

        is_def = bool(res and res.get("defect"))
        boxes = (res or {}).get("boxes") or []
        box = to_pixels(boxes[0], w, h) if boxes else None
        if is_def and not args.no_refine:
            box = box_refine.refine(frame, box) or box

        note = ""
        if kind == "defect":
            nd += 1
            if is_def:
                det += 1
                gt = ep["box"]
                gcx, gcy = (gt[0] + gt[2]) // 2, (gt[1] + gt[3]) // 2
                if box and box[0] <= gcx <= box[2] and box[1] <= gcy <= box[3]:
                    hit += 1
                    note = "рамка верна"
                else:
                    note = "рамка мимо"
            else:
                note = "ПРОПУСК"
        else:
            nc += 1
            if is_def:
                fp += 1
                note = "ЛОЖНАЯ ТРЕВОГА"
        print(f"  {kind:6s} кадр {idx:5d}: defect={is_def}  {lats[-1]:.1f}s  {note}", flush=True)

    print(f"\nдефектные кадры: детекция {det}/{nd}, рамка на дефекте {hit}/{nd}")
    print(f"чистые кадры:    ложных тревог {fp}/{nc}")
    if lats:
        print(f"время на кадр:   среднее {sum(lats) / len(lats):.1f}s, "
              f"максимум {max(lats):.1f}s")
    cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
