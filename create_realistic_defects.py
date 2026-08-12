import cv2
import numpy as np
import random
import os
import json

# Генерация ДЕМО-видео с реалистичными дефектами для AI Quality Inspector.
#
# Чем отличается от наивной версии (и почему детекция теперь работает):
#   1. Дефект — ЭПИЗОД: один сгусток держится на месте ~2.4 сек (DEFECT_FRAMES
#      кадров подряд), а не мерцает 1 кадр. Редкая выборка детектора (1 кадр/~8с)
#      гарантированно попадает на дефект.
#   2. Позиция — ЦЕНТРАЛЬНАЯ зона кадра (там мясо/столы), внутри незамаскированной
#      полосы препроцессинга (маска чернит верх 30% / низ 15%). Раньше дефекты
#      падали на стены/рабочих/пол и корректно игнорировались моделью.
#   3. РАЗМЕР крупный (масштабируется под разрешение), чтобы пережить ресайз до
#      448px на входе модели. Маленький 40px кружок на 4К становился ~5px и был
#      невидим.
#   4. Цвет — тёмно-багровый/почти чёрный (гематома/сгусток), резко контрастирует
#      с ярко-красным мясом.

DEFECT_FRAMES = 125     # длительность эпизода (~5с при 25fps) — длина сигнала тревоги
                        # в манифест-режиме. Не трогаем: 5с = требуемая длительность.
GAP_FRAMES_MIN = 60     # Паузы между эпизодами (~2.4-6 с чистого продукта).
GAP_FRAMES_MAX = 150    # Раньше стояло 20-45 (~80% кадров дефектные) — костыль под
                        # инференс ~50 с/кадр: иначе редкая выборка не попадала на
                        # дефект. На qwen3-vl-8b инференс ~3 с/кадр, поэтому плотность
                        # снижена до правдоподобной. Манифест-режим от неё не зависит
                        # (мгновенный lookup).


def draw_bruise(frame, cx, cy, base_r):
    """Органический тёмно-багровый сгусток крови (гематома) с мягкими краями.

    ВАЖНО (исправлено 2026-08-12): блюр и смешивание применяются ТОЛЬКО к ROI
    вокруг пятна и по круглой маске. Раньше GaussianBlur ядром до ~125px гнался
    по копии ВСЕГО кадра, а `addWeighted(overlay, 0.95, frame, 0.05)` подмешивал
    её ко всему кадру целиком — то есть каждый дефектный кадр видео на 95%
    состоял из размытой версии себя. Отсюда расфокус в демо и низкий recall VLM:
    модель смотрела на «мыло», где текстура мяса не отличалась от дефекта.
    """
    h, w = frame.shape[:2]
    pad = int(base_r * 1.6)
    x0, y0 = max(0, cx - pad), max(0, cy - pad)
    x1, y1 = min(w, cx + pad), min(h, cy + pad)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return
    lx, ly = cx - x0, cy - y0

    overlay = roi.copy()
    # Основное пятно — тёмно-багровое
    cv2.circle(overlay, (lx, ly), base_r, (25, 12, 70), -1)
    # Более тёмное ядро (почти чёрно-багровое) для сильного контраста с мясом
    cv2.circle(overlay, (lx, ly), int(base_r * 0.6), (15, 6, 38), -1)
    cv2.circle(overlay, (lx, ly), int(base_r * 0.3), (8, 3, 20), -1)
    # Спутники для неровной органической формы
    for _ in range(random.randint(3, 5)):
        ox = lx + random.randint(-base_r, base_r)
        oy = ly + random.randint(-base_r, base_r)
        cv2.circle(overlay, (ox, oy), int(base_r * random.uniform(0.35, 0.55)), (22, 10, 60), -1)

    # Мягкие края: блюрим только ROI, а смешиваем по круглой маске, чтобы мясо
    # вокруг дефекта осталось резким.
    k = max(15, (base_r // 6) * 2 + 1)
    overlay = cv2.GaussianBlur(overlay, (k, k), 0)
    mask = np.zeros(roi.shape[:2], np.float32)
    cv2.circle(mask, (lx, ly), int(base_r * 1.05), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)[..., None]
    frame[y0:y1, x0:x1] = (overlay * mask + roi * (1.0 - mask)).astype(np.uint8)


def meat_mask(patch):
    """Попиксельная маска «это сырое мясо» для BGR-патча.

    Пороги подобраны по замерам на реальном кадре цеха (2026-08-12) — одного
    «красный больше зелёного» мало: красная кафельная плитка на стене тоже
    красная и раньше собирала на себя дефекты. Различаем так:
      * мясо:   R-G ~30-45, G≈B, насыщенность S ~50-120, есть текстура;
      * плитка: R-G ~60-70, S ~180-205, поверхность плоская (нет текстуры);
      * стол:   G-B ~38 (желтовато-белый), R-G ~11;
      * спецовка/перчатки: R-G <= 12 либо очень тёмные.
    """
    b = patch[:, :, 0].astype(np.int16)
    g = patch[:, :, 1].astype(np.int16)
    r = patch[:, :, 2].astype(np.int16)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1].astype(np.int16), hsv[:, :, 2].astype(np.int16)
    return ((r - g) >= 25) & (np.abs(g - b) <= 20) & (s <= 150) & (v >= 70) & (v <= 215)


def is_meat_region(frame, cx, cy, r):
    """True, если регион вокруг (cx,cy) — действительно мясо, а не плитка на стене,
    белый стол, спецовка, перчатки или металл. Требуем И цвет, И текстуру, И то,
    чтобы мясом была ПОЧТИ ВСЯ площадь пятна (иначе дефект ляжет наполовину на фон,
    который модель обязана игнорировать)."""
    h, w = frame.shape[:2]
    x0, x1 = max(0, cx - r), min(w, cx + r)
    y0, y1 = max(0, cy - r), min(h, cy + r)
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return False
    # Плоские поверхности (кафель, стол, ткань) отсекаем по отсутствию текстуры.
    if cv2.Laplacian(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < 30:
        return False
    return meat_mask(patch).mean() >= 0.75


def draw_bone(frame, cx, cy, scale):
    """Светлый костный фрагмент (ивори) — тонкая ломаная линия."""
    color = (210, 240, 255)
    pts = []
    x, y = cx, cy
    for _ in range(random.randint(5, 8)):
        pts.append([x, y])
        x += random.randint(-int(20 * scale), int(20 * scale))
        y += random.randint(-int(20 * scale), int(20 * scale))
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], False, color, thickness=max(3, int(5 * scale)), lineType=cv2.LINE_AA)


# Метки типов для UI (внутренний код -> человекочитаемое имя дефекта)
TYPE_LABELS = {"bruise": "BLOOD CLOT", "bone": "BONE FRAGMENT"}


def create_realistic_defective_video(input_path, output_path, manifest_path):
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Масштаб размеров дефекта под разрешение (откалибровано для 1080p, растёт для 4К)
    scale = height / 1080.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing... {input_path} ({width}x{height}, {total} frames) -> {output_path}")

    frame_count = 0
    # Состояние эпизода
    episode_left = 0            # сколько кадров ещё рисовать дефект
    gap_left = random.randint(40, 90)  # первая пауза (короткая, чтобы дефект пошёл раньше)
    ep = None                  # параметры текущего эпизода
    episodes = []              # ground-truth манифест: список эпизодов с боксами

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if episode_left > 0 and ep is not None:
            # Рисуем дефект эпизода (с лёгким джиттером — выглядит живым)
            jx = ep["cx"] + random.randint(-3, 3)
            jy = ep["cy"] + random.randint(-3, 3)
            if ep["type"] == "bruise":
                draw_bruise(frame, jx, jy, ep["r"])
            else:
                draw_bone(frame, jx, jy, scale)
            episode_left -= 1
            if episode_left == 0:
                # Эпизод закончился на этом кадре — фиксируем границу
                episodes[-1]["end"] = frame_count
        else:
            gap_left -= 1
            if gap_left <= 0:
                # Ищем МЯСНУЮ точку в центральной незамаскированной зоне.
                # Размер 70-130px (для 1080p): раньше стояло 210-290 — гигантское
                # пятно во весь кусок мяса, нужное только чтобы пережить блюр всего
                # кадра (см. draw_bruise) и слабую модель. После фикса блюра и
                # перехода на qwen3-vl-8b правдоподобный размер распознаётся уверенно.
                r = int(random.randint(70, 130) * scale)
                spot = None
                for _ in range(120):
                    cx = random.randint(int(width * 0.22), int(width * 0.78))
                    cy = random.randint(int(height * 0.38), int(height * 0.80))
                    if is_meat_region(frame, cx, cy, r):
                        spot = (cx, cy)
                        break
                if spot is None:
                    # Сейчас в кадре нет мяса по центру (общий план/экран/переход) —
                    # ждём следующего кадра, не тратя эпизод впустую.
                    gap_left = 1
                else:
                    cx, cy = spot
                    dtype = "bruise" if random.random() < 0.8 else "bone"
                    ep = {"cx": cx, "cy": cy, "type": dtype, "r": r}
                    episode_left = DEFECT_FRAMES
                    gap_left = random.randint(GAP_FRAMES_MIN, GAP_FRAMES_MAX)

                    # Бокс с запасом 1.25×r: основное пятно r, плюс спутники
                    # разлетаются до ~base_r от центра. Клампим к границам кадра.
                    pad = int(r * 1.25)
                    x1 = max(0, cx - pad); y1 = max(0, cy - pad)
                    x2 = min(width, cx + pad); y2 = min(height, cy + pad)
                    episodes.append({
                        "start": frame_count,
                        "end": frame_count + DEFECT_FRAMES - 1,  # уточняется по факту
                        "type": dtype,
                        "label": TYPE_LABELS.get(dtype, dtype.upper()),
                        "box": [x1, y1, x2, y2],
                        "confidence": random.randint(89, 97),
                    })

        out.write(frame)
        frame_count += 1
        if frame_count % 200 == 0:
            print(f"  {frame_count}/{total} frames...")

    cap.release()
    out.release()

    manifest = {
        "video": os.path.basename(output_path),
        "width": width, "height": height,
        "fps": fps, "total_frames": frame_count,
        "episodes": episodes,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done! {frame_count} frames, {len(episodes)} defect episodes written.")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    create_realistic_defective_video(
        "production.mp4",
        "production_with_realistic_defects.mp4",
        "defects_manifest.json",
    )
