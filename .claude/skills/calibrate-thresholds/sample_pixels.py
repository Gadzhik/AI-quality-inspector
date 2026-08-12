"""Замер цветовых и текстурных характеристик по точкам кадра.

Печатает по каждой точке R-G, G-B, S, V и дисперсию лапласиана (текстуру) и
сохраняет кадр с пронумерованными точками, чтобы сопоставить числа с объектами.

Запуск:
    python sample_pixels.py production.mp4 --frame 120
    python sample_pixels.py frame.jpg --points 880,300 250,200
    python sample_pixels.py production.mp4 --frame 300 --grid 6x4
"""
import argparse
import os
import sys

import cv2
import numpy as np


def load_frame(source, frame_idx):
    if source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(f"Не открывается: {source}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"Нет кадра {frame_idx}")
        return frame
    img = cv2.imread(source)
    if img is None:
        raise SystemExit(f"Не читается: {source}")
    return img


def stats(patch):
    b, g, r = [patch[:, :, i].astype(np.float32).mean() for i in range(3)]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    lap = cv2.Laplacian(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    return r - g, g - b, hsv[:, :, 1].mean(), hsv[:, :, 2].mean(), lap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="видеофайл или изображение")
    ap.add_argument("--frame", type=int, default=0, help="номер кадра для видео")
    ap.add_argument("--points", nargs="*", default=[], help="точки вида x,y")
    ap.add_argument("--grid", default="", help="сетка вида 6x4 вместо точек")
    ap.add_argument("--radius", type=int, default=30, help="полуразмер патча")
    ap.add_argument("--out", default="_pixel_samples.jpg")
    args = ap.parse_args()

    frame = load_frame(args.source, args.frame)
    h, w = frame.shape[:2]

    points = []
    if args.points:
        for p in args.points:
            x, y = p.split(",")
            points.append((int(x), int(y)))
    else:
        cols, rows = (args.grid.split("x") if args.grid else ("5", "4"))
        cols, rows = int(cols), int(rows)
        for j in range(rows):
            for i in range(cols):
                points.append((int(w * (i + 0.5) / cols), int(h * (j + 0.5) / rows)))

    vis = frame.copy()
    rr = args.radius
    print(f"кадр {w}x{h}, патч {rr * 2}x{rr * 2}\n")
    print(f"{'#':>3}  {'x':>5} {'y':>5}  {'R-G':>7} {'G-B':>7} {'S':>7} {'V':>7} {'текстура':>9}")
    for n, (x, y) in enumerate(points, 1):
        x0, y0 = max(0, x - rr), max(0, y - rr)
        patch = frame[y0:min(h, y + rr), x0:min(w, x + rr)]
        if patch.size == 0:
            continue
        rg, gb, s, v, lap = stats(patch)
        print(f"{n:>3}  {x:>5} {y:>5}  {rg:>7.1f} {gb:>7.1f} {s:>7.1f} {v:>7.1f} {lap:>9.1f}")
        cv2.circle(vis, (x, y), rr, (0, 255, 255), 2)
        cv2.putText(vis, str(n), (x - 10, y - rr - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3, cv2.LINE_AA)

    scale = 1280 / w
    cv2.imwrite(args.out, cv2.resize(vis, (1280, int(h * scale))))
    print(f"\nкадр с точками -> {args.out} (открой его, чтобы сопоставить номера с объектами)")
    print("\nориентиры этого проекта:")
    print("  мясо      R-G 29..51   S 50..120   текстура 44..267")
    print("  жир туши  R-G -1..2    S 13..16    (по цвету неотличим от одежды)")
    print("  кафель    R-G 51..68   S 173..205  текстура ~4")
    print("  фартук    R-G ~33      S ~209      текстура ~2")
    print("  стол      R-G ~10      G-B 32..38  текстура ~28")
    print("  сгусток   R-G ~45      S ~220      V 35..60")
    return 0


if __name__ == "__main__":
    sys.exit(main())
