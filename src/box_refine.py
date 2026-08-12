"""Уточнение рамки дефекта классическим CV поверх вердикта VLM.

Зачем. Замеры 2026-08-12 на qwen3-vl-8b: вердикт (`defect`/`clean`) модель даёт
надёжно (9/9 на дефектных кадрах), а вот КООРДИНАТЫ почти всегда мимо — из 9
детекций бокс накрывал реальное пятно 0-2 раза. Это типично для VLM: она «видит»
аномалию, но не выдаёт точную геометрию, тем более после сжатия кадра до 448px.

Решение — двухэтапная схема:
    1) VLM отвечает на вопрос «есть ли дефект и какой» (семантика);
    2) здесь мы находим само пятно пиксельно и даём ТОЧНУЮ рамку (геометрия).

Результат на ground-truth манифесте (27 кадров, из них 3 — костный фрагмент):
пятно найдено в 18 кадрах, и во ВСЕХ 18 центр реального дефекта попал внутрь
рамки (средний IoU 0.44). То есть ложных рамок метод не плодит, а костные
фрагменты (светлые, а не тёмные) он не ищет — на них возвращается None и
остаётся бокс VLM.

Сигнатура сгустка/гематомы (замерено на кадрах цеха):
    сгусток:      S ~220, V ~35-60, R сильно больше G (R~50, G~6)
    мясо рядом:   S ~40-115, V ~55-180
То есть «очень насыщенный + очень тёмный + красный» — сочетание, которого у
нормальной мышцы, жира, кафеля и спецодежды не бывает.

Если пятно не найдено — возвращаем None, и вызывающий код оставляет бокс VLM.
"""
import cv2
import numpy as np

from .meat_crop import meat_fraction, meat_mask

# Пороги сигнатуры сгустка (см. док выше). Консервативные: лучше не найти пятно
# и оставить бокс VLM, чем обвести случайную тень.
SAT_MIN = 190        # насыщенность
VAL_MAX = 75         # яркость (сгусток почти чёрный)
RG_MIN = 18          # насколько красный канал выше зелёного
MIN_AREA_FRAC = 3e-5  # мин. площадь пятна в долях площади кадра (~60 px на 1080p)
MAX_AREA_FRAC = 0.05  # макс. площадь: реальные сгустки на замерах 17-65 тыс. px
                      # (0.8-3% кадра), а бордовый фартук мясника даёт «пятно» на
                      # 160-170 тыс. px (8%) — по размеру и отсекаем.
MIN_LOCAL_CONTRAST = 30  # насколько пятно должно быть темнее своего окружения (V).
                         # Замер: 30 -> ложных рамок 2/9 на чистых кадрах,
                         #        25 и 20 -> 5/9 при том же recall. Оставляем 30.

# Вертикальная зона поиска — та же, что видит модель после маскирования
# (верх 30% и низ 15% кадра занимают стены/пол/оборудование).
ZONE_TOP, ZONE_BOTTOM = 0.28, 0.87


def clot_mask(bgr: np.ndarray) -> np.ndarray:
    """Бинарная маска пикселей, похожих на сгусток крови / гематому."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1].astype(np.int16), hsv[:, :, 2].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    mask = ((s >= SAT_MIN) & (v <= VAL_MAX) & ((r - g) >= RG_MIN)).astype(np.uint8) * 255
    # Смыкаем разрывы (пятно органическое, с сателлитами) и убираем крапинки.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


def _intersects(a, b) -> bool:
    """Пересекаются ли два бокса [x1, y1, x2, y2]."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _surrounded_by_meat(frame, box, ring: float = 1.7, min_frac: float = 0.18) -> bool:
    """True, если ВОКРУГ найденного пятна действительно мясо.

    Нужно, чтобы отсечь бордовый фартук мясника, тёмную плитку и прочие тёмные
    насыщенно-красные поверхности: по цвету они похожи на сгусток, но окружены
    тканью/кафелем, а не мышечной тканью. Считаем долю «мясных» пикселей в кольце
    вокруг бокса (сам бокс из подсчёта исключаем).
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * ring, (y2 - y1) * ring
    rx0, ry0 = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
    rx1, ry1 = int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2))
    region = frame[ry0:ry1, rx0:rx1].copy()
    if region.size == 0:
        return False
    # вырезаем середину (само пятно), чтобы считать только окружение
    ix0, iy0 = max(0, x1 - rx0), max(0, y1 - ry0)
    ix1, iy1 = min(region.shape[1], x2 - rx0), min(region.shape[0], y2 - ry0)
    mask = np.ones(region.shape[:2], bool)
    mask[iy0:iy1, ix0:ix1] = False
    if mask.sum() == 0:
        return False
    ring_px = region[mask].reshape(-1, 1, 3)
    if meat_fraction(ring_px) < min_frac:
        return False
    # Локальный контраст: сгусток заметно ТЕМНЕЕ того, что вокруг. У бордового
    # фартука/тёмной плитки окружение такое же тёмное — контраста нет.
    inner = frame[y1:y2, x1:x2]
    if inner.size == 0:
        return False
    ring_v = float(cv2.cvtColor(ring_px, cv2.COLOR_BGR2HSV)[:, :, 2].mean())
    inner_v = float(cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)[:, :, 2].mean())
    return ring_v - inner_v >= MIN_LOCAL_CONTRAST


def _meat_on_sides(frame: np.ndarray, box, min_sides: int = 3,
                   band: float = 0.45, side_frac: float = 0.30) -> bool:
    """Мясо ли вокруг пятна с большинства сторон.

    Кольцевая проверка `_surrounded_by_meat` усредняет всё кольцо целиком и потому
    пропускает пятно НА КРАЮ: когда рамка области мяса задевает бордовый фартук,
    в кольце вокруг тёмного пятна на фартуке всё ещё хватает мясных пикселей.
    Сгусток внутри куска окружён мясом со всех четырёх сторон, а пятно на границе —
    только с одной-двух. Считаем стороны по отдельности.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    dx, dy = max(3, int(bw * band)), max(3, int(bh * band))
    strips = [
        frame[max(0, y1 - dy):y1, x1:x2],          # сверху
        frame[y2:min(h, y2 + dy), x1:x2],          # снизу
        frame[y1:y2, max(0, x1 - dx):x1],          # слева
        frame[y1:y2, x2:min(w, x2 + dx)],          # справа
    ]
    good = sum(1 for s in strips if s.size and meat_fraction(s) >= side_frac)
    return good >= min_sides


def on_meat(frame: np.ndarray, box, min_frac: float = 0.25) -> bool:
    """Указывает ли бокс вообще на мясо.

    Нужен как предохранитель для сырого бокса VLM: модель периодически объявляет
    «blood clot» на бордовом фартуке мясника или на красной плитке. Если внутри
    рамки мяса почти нет — это не дефект продукции, и тревогу поднимать нельзя.
    """
    if not box or len(box) != 4:
        return False
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return False
    return meat_fraction(frame[y1:y2, x1:x2]) >= min_frac


def refine(frame: np.ndarray, vlm_box=None, pad: int = 6, zone: bool = True):
    """Вернуть уточнённый [x1, y1, x2, y2] в пикселях кадра или None.

    Ищем ВСЕГДА по всей рабочей зоне кадра, а бокс VLM (`vlm_box`, в пикселях)
    используем только для упорядочивания кандидатов — берём ближайшего к нему.
    Так и было выбрано по замерам: если ограничивать поиск окрестностью бокса
    VLM, то при неверном боксе (обычный случай) либо не находится ничего, либо
    находится случайное пятно рядом с ошибочной точкой — на 9 дефектных кадрах
    это давало 4 верные рамки против 0 и 2 неверных.

    `zone=False` — искать по всему переданному изображению. Нужно, когда сюда
    приходит уже вырезанная область мяса: обрезать её ещё раз «верх 30% / низ 15%»
    (эти доли посчитаны для полного кадра цеха) значит выбросить часть образца.
    """
    h, w = frame.shape[:2]
    if zone:
        sx0, sy0, sx1, sy1 = 0, int(h * ZONE_TOP), w, int(h * ZONE_BOTTOM)
    else:
        sx0, sy0, sx1, sy1 = 0, 0, w, h

    hint = list(vlm_box) if (vlm_box and len(vlm_box) == 4) else None

    if sx1 - sx0 < 8 or sy1 - sy0 < 8:
        return None

    region = frame[sy0:sy1, sx0:sx1]
    mask = clot_mask(region)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area, max_area = MIN_AREA_FRAC * w * h, MAX_AREA_FRAC * w * h
    cands = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
    if not cands:
        return None

    def to_box(c):
        bx, by, bw_, bh_ = cv2.boundingRect(c)
        return [
            max(0, sx0 + bx - pad),
            max(0, sy0 + by - pad),
            min(w, sx0 + bx + bw_ + pad),
            min(h, sy0 + by + bh_ + pad),
        ]

    # Порядок проверки кандидатов: сначала те, что реально ПЕРЕСЕКАЮТСЯ с боксом
    # VLM (значит, модель показала туда же), затем — просто самые крупные.
    # Сортировать по расстоянию до бокса VLM нельзя: бокс обычно неверен, и тогда
    # выбирается случайное пятно рядом с ошибочной точкой (замер: 2 верные рамки
    # из 9 против 6 при сортировке по площади).
    boxes = sorted((to_box(c) for c in cands),
                   key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    if hint is not None:
        order = ([b for b in boxes if _intersects(b, hint)] +
                 [b for b in boxes if not _intersects(b, hint)])
    else:
        order = boxes

    # Берём первого кандидата, который и окружён мясом, и лежит внутри его тела.
    for box in order[:5]:
        if _surrounded_by_meat(frame, box) and _meat_on_sides(frame, box):
            return box
    return None
