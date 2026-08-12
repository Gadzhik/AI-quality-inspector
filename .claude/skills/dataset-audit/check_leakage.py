"""Проверка утечки между двумя выборками изображений (dHash + расстояние Хэмминга).

Для каждого изображения из второй папки ищет ближайшее в первой. Точные дубликаты
и почти-дубликаты означают, что разбиение на train/test испорчено и метрике на нём
верить нельзя.

Запуск:
    python check_leakage.py <train_dir> <test_dir>
    python check_leakage.py data/locbeef/train data/locbeef/test --recursive
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def dhash(path, size=8):
    """64-битный перцептивный хеш. draft() ускоряет декодирование больших JPEG."""
    try:
        im = Image.open(path)
        im.draft("L", (size * 8, size * 8))
        im = im.convert("L").resize((size + 1, size), Image.BILINEAR)
    except Exception:
        return None
    a = np.asarray(im, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def collect(folder, recursive):
    pattern = "**/*" if recursive else "*"
    files = [p for p in glob.glob(os.path.join(folder, pattern), recursive=recursive)
             if p.lower().endswith(EXTS)]
    hashes, keep = [], []
    for p in sorted(files):
        h = dhash(p)
        if h is not None:
            hashes.append(h)
            keep.append(p)
    return keep, (np.stack(hashes) if hashes else np.empty((0, 8), np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_dir")
    ap.add_argument("test_dir")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--examples", type=int, default=5, help="сколько худших пар показать")
    args = ap.parse_args()

    train_files, TR = collect(args.train_dir, args.recursive)
    test_files, TE = collect(args.test_dir, args.recursive)
    if not len(TR) or not len(TE):
        print("Не найдено изображений — проверь пути.")
        return 1
    print(f"train {len(train_files)} | test {len(test_files)}")

    dists, nearest = [], []
    for i in range(len(TE)):
        d = POPCOUNT[np.bitwise_xor(TR, TE[i])].sum(1)
        j = int(d.argmin())
        dists.append(int(d[j]))
        nearest.append(j)
    dists = np.array(dists)

    n = len(dists)
    print("\nрасстояние до ближайшего обучающего кадра:")
    for thr in (0, 3, 8, 16):
        cnt = int((dists <= thr).sum())
        print(f"  <= {thr:2d} : {cnt:5d}/{n}  ({cnt / n:.1%})")
    print(f"  медиана {np.median(dists):.0f}, среднее {dists.mean():.1f}")

    leak = (dists <= 8).mean()
    print()
    if leak > 0.10:
        print(f"ВЕРДИКТ: сплит испорчен ({leak:.0%} тестовых кадров имеют почти-дубликат "
              f"в обучающей выборке). Строй своё разбиение по кластерам.")
    else:
        print(f"ВЕРДИКТ: заметной утечки нет ({leak:.0%}).")

    if args.examples:
        print(f"\nближайшие пары (первые {args.examples}):")
        for i in np.argsort(dists)[:args.examples]:
            print(f"  d={dists[i]:2d}  {os.path.basename(test_files[i])}"
                  f"  <->  {os.path.basename(train_files[nearest[i]])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
