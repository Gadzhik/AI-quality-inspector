"""Скачивает РЕАЛЬНЫЕ размеченные датасеты мяса для обучения детектора свежести.

Синтетика здесь не используется вообще: это фотографии настоящей говядины с
метками, поставленными авторами исследований, а не нами.

Что качается (оба — Mendeley Data, лицензия CC BY 4.0):

1. LOCBEEF (`nhs6mjg6yy`) — 3268 фото местной ачехской говядины, снятых с 07:00
   до 22:00, метки `fresh` / `rotten`. Готовое разбиение train/test от авторов.
   Это ОБУЧАЮЩИЙ набор. ~5.2 ГБ.

2. Fresh / non-fresh beef samples (`wvhkpppddp`) — 60 фото говядины трёх отрубов
   (inside skirt, knuckle, sirloin), снятых на 1-й и 5-й день после покупки,
   плюс 60 сегментированных версий. ~16 МБ.
   Это НЕЗАВИСИМЫЙ ТЕСТ: другая лаборатория, другая камера, другое освещение,
   другое определение «несвежести». На нём видно, выучила ли модель мясо или
   особенности съёмки конкретного датасета.

Запуск:
    python fetch_freshness_dataset.py
Результат: `data/locbeef/{train,test}/{fresh,rotten}/` и `data/beef_indep/`.
"""
import os
import sys
import zipfile

import requests

MENDELEY_API = "https://data.mendeley.com/public-api/datasets/{}/files?folder_id=root&version=1"

DATA_DIR = "data"
LOCBEEF_DIR = os.path.join(DATA_DIR, "locbeef")
INDEP_DIR = os.path.join(DATA_DIR, "beef_indep")


def list_files(dataset_id):
    r = requests.get(MENDELEY_API.format(dataset_id), timeout=60)
    r.raise_for_status()
    return r.json()


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"    уже есть: {os.path.basename(path)}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with requests.get(url, stream=True, timeout=1800) as r:
        r.raise_for_status()
        done = 0
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
                done += len(chunk)
                if done % (1 << 28) == 0:
                    print(f"      {done / 1e9:.1f} ГБ...", flush=True)
    print(f"    готово: {os.path.basename(path)} ({os.path.getsize(path) / 1e6:.0f} МБ)")


def fetch_locbeef():
    print("1/2 LOCBEEF (обучающий набор, ~5.2 ГБ)")
    os.makedirs(LOCBEEF_DIR, exist_ok=True)
    files = {f["filename"]: f["content_details"]["download_url"] for f in list_files("nhs6mjg6yy")}
    for name in ("train.zip", "test.zip"):
        archive = os.path.join(LOCBEEF_DIR, name)
        download(files[name], archive)
        target = os.path.join(LOCBEEF_DIR, name[:-4])
        if not os.path.isdir(target):
            print(f"    распаковываю {name}...")
            with zipfile.ZipFile(archive) as z:
                z.extractall(LOCBEEF_DIR)
    for split in ("train", "test"):
        for cls in ("fresh", "rotten"):
            d = os.path.join(LOCBEEF_DIR, split, cls)
            n = len(os.listdir(d)) if os.path.isdir(d) else 0
            print(f"    {split}/{cls}: {n}")


def fetch_independent():
    print("2/2 Независимый тестовый набор (~16 МБ)")
    os.makedirs(INDEP_DIR, exist_ok=True)
    files = list_files("wvhkpppddp")
    for f in files:
        # `_seg` — сегментированные версии, для оценки они не нужны
        if "_seg" in f["filename"]:
            continue
        download(f["content_details"]["download_url"], os.path.join(INDEP_DIR, f["filename"]))
    n = len([x for x in os.listdir(INDEP_DIR) if x.lower().endswith(".jpg")])
    print(f"    изображений: {n}")


def main():
    try:
        fetch_locbeef()
        fetch_independent()
    except Exception as e:
        print(f"ОШИБКА: {e}")
        return 1
    print("\nГотово. Дальше: python train_freshness.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
