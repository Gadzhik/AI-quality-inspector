"""Поиск областей мяса в кадре производственной сцены.

Зачем отдельный модуль. `meat_crop.meat_bbox` вырезает «самое красное пятно» —
это верно для студийного фото, где кусок мяса лежит один на белом фоне, и именно
так готовились данные для обучения классификатора. На кадре цеха то же правило
выбирает красную кафельную плитку на стене: она больше и краснее любой туши.
Менять `meat_bbox` нельзя — препроцессинг инференса обязан совпадать с обучением,
поэтому сценовый поиск живёт здесь, отдельно.

Метод: попиксельная маска мяса (`meat_crop.meat_mask` — она уже отсекает плитку
по пересыщенности), морфология, связные компоненты, отбор по площади, доле мяса
внутри бокса и наличию текстуры. Возвращаются несколько областей, а не одна:
в кадре может лежать несколько кусков.
"""
import cv2
import numpy as np

from .meat_crop import meat_fraction, meat_mask

WORK_WIDTH = 640          # на такой ширине ищем: быстрее и устойчивее к шуму
MIN_AREA_FRAC = 0.010     # мин. площадь области, доля кадра
MAX_REGIONS = 3
MIN_MEAT_FRAC = 0.10      # доля мясных пикселей внутри финального бокса
MIN_TEXTURE = 25.0        # дисперсия лапласиана: у плитки ~4, у мяса 44-267
PAD = 0.04                # запас вокруг области
DENSITY_WIN = 0.06        # окно плотности, доля ширины кадра
DENSITY_MIN = 0.10        # какая доля мясных пикселей в окне уже считается тушей

# Почему через плотность, а не «красные пиксели напрямую». Замер на кадрах цеха
# (баранина): красное мясо даёт R-G ~29-51 и уверенно ловится правилом, но
# ПОВЕРХНОСТЬ ТУШИ в основном белый жир — R-G ~-1..+2, S ~13-16, то есть по цвету
# он неотличим от серой кофты рабочего. Поэтому красные участки используем как
# зёрна: строим карту их плотности, и связная область вокруг зёрен и есть туша.
# Плитку (S ~173) и фартук (S ~209) правило отсекает ещё на уровне пикселей,
# белый стол — по G-B ~33, а гладкие поверхности добивает проверка текстуры.


def find_meat_regions(bgr: np.ndarray, max_regions: int = MAX_REGIONS):
    """Список областей мяса: [{'box': [x1,y1,x2,y2], 'meat': доля, 'area': px}, ...].

    Отсортированы по площади. Пустой список — мяса в кадре не найдено.
    """
    h, w = bgr.shape[:2]
    scale = WORK_WIDTH / float(w)
    small = cv2.resize(bgr, (WORK_WIDTH, max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1 else bgr.copy()
    sh, sw = small.shape[:2]

    seeds = meat_mask(small).astype(np.float32)
    win = max(9, int(sw * DENSITY_WIN) | 1)
    density = cv2.boxFilter(seeds, -1, (win, win), normalize=True)
    mask = (density >= DENSITY_MIN).astype(np.uint8) * 255
    # Смыкаем разрывы внутри туши, затем убираем мелкий мусор (капли на столе,
    # красные пятна на одежде).
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = MIN_AREA_FRAC * sw * sh

    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x, y, bw, bh = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        px, py = int(bw * PAD), int(bh * PAD)
        x1, y1 = max(0, x - px), max(0, y - py)
        x2, y2 = min(sw, x + bw + px), min(sh, y + bh + py)
        patch = small[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        frac = meat_fraction(patch)
        if frac < MIN_MEAT_FRAC:
            continue
        texture = cv2.Laplacian(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if texture < MIN_TEXTURE:
            continue
        inv = 1.0 / scale if scale < 1 else 1.0
        out.append({
            "box": [int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv)],
            "meat": round(frac, 3),
            "area": int(area * inv * inv),
        })

    out.sort(key=lambda r: r["area"], reverse=True)
    return out[:max_regions]
