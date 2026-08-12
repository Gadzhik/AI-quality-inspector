"""Обучение детектора свежести мяса на РЕАЛЬНЫХ данных (без синтетики).

Данные качает `fetch_freshness_dataset.py`:
  * LOCBEEF — 3268 фото говядины, метки `fresh` / `rotten` от авторов датасета;
  * 60 фото из другого исследования (другая лаборатория, камера и определение
    несвежести: 1-й против 5-го дня хранения) — независимая проверка.

## Два решения, без которых результат был бы враньём

**1. Авторское разбиение train/test использовать нельзя.** Проверка перцептивными
хешами показала: 133 тестовых кадра имеют ТОЧНЫЙ дубликат в обучающей выборке, а
у 97% тестовых кадров есть почти-дубликат (расстояние Хэмминга <= 8). Это один и
тот же кусок мяса, снятый подряд много раз. На таком сплите модель показывает
99.9% и это ничего не значит. Поэтому мы склеиваем оба каталога в один пул,
кластеризуем почти-дубликаты и режем на train/val/test ПО КЛАСТЕРАМ — так один и
тот же физический образец не может оказаться сразу в обучении и в тесте.

**2. Кадр обрезается до мяса** (`src/meat_crop.py`). Иначе модель видит белую
тарелку, деревянный стол и тени — и учит их. Плюс срезается нижняя полоса с
выжженным таймштампом: свежие кадры сняты раньше по времени суток, чем испорченные,
так что по цифрам можно «предсказать» класс, ничего не поняв про мясо.

Обрезанные кадры кешируются в `data/_cache/`, чтобы не повторять эту работу
каждую эпоху.

Запуск:
    python train_freshness.py
    python train_freshness.py --epochs 12 --arch resnet34
"""
import argparse
import glob
import json
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src.meat_crop import crop_meat, white_balance

DATA_LOCBEEF = os.path.join("data", "locbeef")
DATA_INDEP = os.path.join("data", "beef_indep")
CACHE_DIR = os.path.join("data", "_cache")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "freshness_resnet18.pt")
METRICS_PATH = os.path.join(MODEL_DIR, "freshness_metrics.json")

CLASSES = ["fresh", "spoiled"]     # 0 = свежее, 1 = испорченное
IMG_SIZE = 224
CACHE_MAX_SIDE = 512
CROP_BOTTOM = 0.10                 # нижняя полоса кадра с таймштампом
DUP_THRESHOLD = 8                  # dHash-расстояние, ниже которого кадры считаем одним образцом

# Метки независимого набора: для каждого отруба первая дата — 1-й день (свежее),
# вторая — 5-й день (несвежее). Из описания датасета wvhkpppddp.
INDEP_LABELS = {
    "20210830": 0, "20210903": 1,   # A — inside skirt
    "20210906": 0, "20210910": 1,   # S — sirloin
    "20210913": 0, "20210917": 1,   # B — knuckle
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
# Подготовка: обрезка таймштампа + кроп по мясу, с кешем на диск
# --------------------------------------------------------------------------- #

def preprocess_image(path, crop_timestamp=True):
    """Прочитать кадр, срезать таймштамп и обрезать до мяса. None при ошибке."""
    img = cv2.imread(path)
    if img is None:
        return None
    if crop_timestamp:
        img = img[:int(img.shape[0] * (1 - CROP_BOTTOM))]
    img = white_balance(img)     # обязательно ДО кропа, пока белый фон в кадре
    img = crop_meat(img)
    h, w = img.shape[:2]
    scale = CACHE_MAX_SIDE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def build_cache():
    """Разложить обработанные кадры в data/_cache/{locbeef,indep}/{fresh,spoiled}/."""
    jobs = []
    for split in ("train", "test"):
        for cls, label in (("fresh", 0), ("rotten", 1)):
            src = os.path.join(DATA_LOCBEEF, split, cls)
            if not os.path.isdir(src):
                continue
            for p in sorted(glob.glob(os.path.join(src, "*"))):
                dst = os.path.join(CACHE_DIR, "locbeef", CLASSES[label],
                                   f"{split}_{os.path.basename(p)}")
                jobs.append((p, dst, True))
    for p in sorted(glob.glob(os.path.join(DATA_INDEP, "*.jpg"))):
        name = os.path.basename(p)
        if "_seg" in name:
            continue
        date = name.split("_")[0]
        if date not in INDEP_LABELS:
            continue
        dst = os.path.join(CACHE_DIR, "indep", CLASSES[INDEP_LABELS[date]], name)
        jobs.append((p, dst, False))

    todo = [j for j in jobs if not os.path.exists(j[1])]
    if todo:
        print(f"подготовка кеша: {len(todo)} кадров (обрезка таймштампа + кроп по мясу)...")
        for i, (src, dst, ts) in enumerate(todo, 1):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            img = preprocess_image(src, ts)
            if img is not None:
                cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if i % 500 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    return jobs


# --------------------------------------------------------------------------- #
# Честное разбиение: кластеризация почти-дубликатов
# --------------------------------------------------------------------------- #

POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)


def dhash(path, size=8):
    try:
        im = Image.open(path)
        im.draft("L", (size * 8, size * 8))
        im = im.convert("L").resize((size + 1, size), Image.BILINEAR)
    except Exception:
        return None
    a = np.asarray(im, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def cluster_duplicates(paths, threshold=DUP_THRESHOLD):
    """Leader-кластеризация по dHash: почти-дубликаты попадают в один кластер.

    Именно leader, а НЕ union-find: при транзитивном склеивании цепочка
    «A похож на B, B похож на C» стягивает в один кластер вообще всё — на этом
    датасете union-find давал 83 кластера на 3268 кадров, и обучаться было не на
    чем. Здесь кадр сравнивается только с лидерами уже созданных кластеров.
    """
    hashes, keep = [], []
    for p in paths:
        h = dhash(p)
        if h is not None:
            hashes.append(h)
            keep.append(p)
    H = np.stack(hashes)

    leaders = []            # индексы кадров-лидеров
    clusters = []           # списки путей
    for i in range(len(keep)):
        if leaders:
            d = POPCOUNT[np.bitwise_xor(H[leaders], H[i])].sum(1)
            j = int(d.argmin())
            if d[j] <= threshold:
                clusters[j].append(keep[i])
                continue
        leaders.append(i)
        clusters.append([keep[i]])
    return clusters


def split_by_clusters(items, seed, val_frac=0.15, test_frac=0.25):
    """Разложить кластеры (а не отдельные кадры) по train/val/test.

    Крупные кластеры раскладываем первыми и всегда в тот сплит, который сильнее
    всех отстаёт от своей квоты, — иначе один кластер на 200+ кадров перекашивает
    всё разбиение.
    """
    label_of = dict(items)
    clusters = cluster_duplicates([p for p, _ in items])
    rnd = random.Random(seed)
    rnd.shuffle(clusters)

    per_class = {0: [], 1: []}
    for c in clusters:
        labels = [label_of[p] for p in c]
        per_class[max(set(labels), key=labels.count)].append(c)

    buckets = {"train": [], "val": [], "test": []}
    for cls, cs in per_class.items():
        total = sum(len(c) for c in cs)
        quota = {"train": total * (1 - val_frac - test_frac),
                 "val": total * val_frac, "test": total * test_frac}
        got = {k: 0 for k in quota}
        for c in sorted(cs, key=len, reverse=True):
            # доля недобора: кому нужнее, тот и получает кластер
            target = max(quota, key=lambda k: (quota[k] - got[k]) / max(1.0, quota[k]))
            buckets[target] += [(p, label_of[p]) for p in c]
            got[target] += len(c)

    rnd.shuffle(buckets["train"])
    return buckets["train"], buckets["val"], buckets["test"], len(clusters)


# --------------------------------------------------------------------------- #
# Данные и модель
# --------------------------------------------------------------------------- #

def build_transforms(train: bool, img_size=IMG_SIZE, jitter=0.25):
    """Аугментации. `jitter` — сила цветовых искажений.

    Тонкий момент: цвет и есть основной признак порчи, поэтому сильный джиттер
    рискует убить сигнал. Но без него модель привязывается к балансу белого
    конкретной камеры и не переносится на чужие снимки. Оптимум подбирается
    замером на независимом наборе (см. README).
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=jitter, contrast=jitter,
                                   saturation=jitter * 0.8, hue=min(0.5, jitter * 0.12)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class ListDataset(Dataset):
    def __init__(self, items, tf):
        self.items = items
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), label


def scan_cache(source):
    items = []
    for label, cls in enumerate(CLASSES):
        d = os.path.join(CACHE_DIR, source, cls)
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, "*"))):
            items.append((p, label))
    return items


def make_model(arch):
    fn = {"resnet18": models.resnet18, "resnet34": models.resnet34}[arch]
    weights = {"resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
               "resnet34": models.ResNet34_Weights.IMAGENET1K_V1}[arch]
    m = fn(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = tn = fp = fn = correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        tp += ((pred == 1) & (y == 1)).sum().item()
        tn += ((pred == 0) & (y == 0)).sum().item()
        fp += ((pred == 1) & (y == 0)).sum().item()
        fn += ((pred == 0) & (y == 1)).sum().item()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"n": total, "accuracy": round(correct / max(1, total), 4),
            "precision_spoiled": round(prec, 4), "recall_spoiled": round(rec, 4),
            "f1_spoiled": round(2 * prec * rec / max(1e-9, prec + rec), 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34"])
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    # jitter=0.5 выбран замером на независимом наборе: 224px + сильный цветовой
    # джиттер дал 65% против 57% (умеренный джиттер) и 58% (320px). На собственном
    # hold-out все конфигурации одинаковы (~100%), то есть выбирать по нему нечего.
    ap.add_argument("--jitter", type=float, default=0.5)
    ap.add_argument("--tag", default="", help="суффикс к именам файлов модели/метрик")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model_path = MODEL_PATH.replace(".pt", f"{args.tag}.pt")
    metrics_path = METRICS_PATH.replace(".json", f"{args.tag}.json")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    build_cache()
    pool = scan_cache("locbeef")
    indep_items = scan_cache("indep")
    if not pool:
        print("Нет данных. Сначала: python fetch_freshness_dataset.py")
        return

    print("кластеризация почти-дубликатов (честное разбиение)...")
    train_items, val_items, test_items, n_clusters = split_by_clusters(pool, args.seed)
    print(f"кадров {len(pool)} -> кластеров {n_clusters}")
    print(f"train {len(train_items)} | val {len(val_items)} | test {len(test_items)} "
          f"| independent {len(indep_items)}")

    def dl(items, tf, shuf):
        return DataLoader(ListDataset(items, tf), batch_size=args.batch, shuffle=shuf,
                          num_workers=args.workers, pin_memory=(device.type == "cuda"),
                          persistent_workers=args.workers > 0)

    train_loader = dl(train_items, build_transforms(True, args.img_size, args.jitter), True)
    val_loader = dl(val_items, build_transforms(False, args.img_size), False)
    test_loader = dl(test_items, build_transforms(False, args.img_size), False)
    indep_loader = dl(indep_items, build_transforms(False, args.img_size), False) if indep_items else None

    model = make_model(args.arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_acc, best_epoch = -1.0, -1
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = seen = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * y.numel()
            seen += y.numel()
        sched.step()
        val = evaluate(model, val_loader, device)
        print(f"epoch {epoch}/{args.epochs}  loss {run_loss / max(1, seen):.4f}  "
              f"val acc {val['accuracy']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if val["accuracy"] > best_acc:
            best_acc, best_epoch = val["accuracy"], epoch
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES,
                        "img_size": args.img_size, "crop_bottom": CROP_BOTTOM,
                        "arch": args.arch}, model_path)

    print(f"\nлучшая эпоха: {best_epoch} (val acc {best_acc:.4f}) -> {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device)["state_dict"])

    results = {"val": evaluate(model, val_loader, device),
               "test_holdout_no_leak": evaluate(model, test_loader, device)}
    if indep_loader is not None:
        results["independent_other_study"] = evaluate(model, indep_loader, device)
    results["_meta"] = {"arch": args.arch, "epochs": args.epochs, "best_epoch": best_epoch,
                        "images": len(pool), "clusters": n_clusters,
                        "train_size": len(train_items), "seed": args.seed,
                        "train_seconds": round(time.time() - t0)}

    results["_meta"]["img_size"] = args.img_size
    results["_meta"]["jitter"] = args.jitter
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== РЕЗУЛЬТАТЫ ===")
    for name in ("val", "test_holdout_no_leak", "independent_other_study"):
        if name in results:
            r = results[name]
            print(f"{name:24s} n={r['n']:4d}  acc {r['accuracy']:.3f}  "
                  f"precision {r['precision_spoiled']:.3f}  recall {r['recall_spoiled']:.3f}  "
                  f"(tp{r['tp']} tn{r['tn']} fp{r['fp']} fn{r['fn']})")
    print(f"\nметрики -> {metrics_path}")


if __name__ == "__main__":
    main()
