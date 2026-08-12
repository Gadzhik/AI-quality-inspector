"""Вырезание области мяса из кадра — общий препроцессинг для обучения и инференса.

Зачем. На фото датасета мясо лежит на белой тарелке на деревянном столе. Если
подавать модели кадр целиком, она может выучить тарелку, стол, тени и угол съёмки
вместо самого мяса — и рассыпаться на любых других снимках. Обрезка до мяса
убирает этот «фон-подсказку».

Метод: мясо — самая красная область кадра. Берём канал `a` в LAB (красный-зелёный),
порог по нему, крупнейшую связную компоненту и её bbox с запасом. Если ничего
уверенного не нашлось (например, кадр целиком занят мясом), возвращаем весь кадр.

ВАЖНО: обучение и инференс обязаны использовать одну и ту же функцию, иначе
модель на проде увидит не то, на чём училась.
"""
import cv2
import numpy as np

MARGIN = 0.08          # запас вокруг найденной области, доля от её размера
MIN_AREA_FRAC = 0.01   # если «мясо» меньше 1% кадра — считаем, что не нашли
WHITE_PCT = 97         # перцентиль яркости, который считаем «белым» фоном


def white_balance(bgr: np.ndarray) -> np.ndarray:
    """Нормализация освещения по белому фону кадра.

    Зачем. Модель, обученная на одном датасете, легко привязывается к балансу
    белого конкретной камеры и к тёплому свету помещения: «испорченное» для неё
    означает просто «темнее и желтее, чем я привыкла». На снимках другой
    лаборатории такая модель называет свежим вообще всё.

    В обеих съёмках мясо лежит на белом (пенопластовая тарелка / белая бумага),
    поэтому самые светлые пиксели кадра — это заведомо нейтральный белый. Приводим
    их к серому и тем самым убираем разницу камер и освещения.

    ВАЖНО: считать нужно ДО обрезки по мясу, пока фон ещё в кадре.
    """
    f = bgr.astype(np.float32)
    ref = np.percentile(f.reshape(-1, 3), WHITE_PCT, axis=0)
    ref[ref < 1.0] = 1.0
    gain = ref.mean() / ref
    return np.clip(f * gain, 0, 255).astype(np.uint8)


def meat_mask(patch: np.ndarray) -> np.ndarray:
    """Попиксельная маска «это сырое мясо»: R заметно выше G, G≈B, умеренная
    насыщенность, не пересвет и не темень. Пороги замерены по кадрам цеха.

    Единственный источник правды о том, «что такое мясо по цвету»: этой маской
    пользуются поиск мясных областей, уточнение рамки дефекта и классификатор.
    """
    b = patch[:, :, 0].astype(np.int16)
    g = patch[:, :, 1].astype(np.int16)
    r = patch[:, :, 2].astype(np.int16)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1].astype(np.int16), hsv[:, :, 2].astype(np.int16)
    return ((r - g) >= 25) & (np.abs(g - b) <= 20) & (s <= 150) & (v >= 70) & (v <= 215)


def meat_fraction(patch: np.ndarray) -> float:
    """Доля «мясных» пикселей в патче."""
    if patch.size == 0:
        return 0.0
    return float(meat_mask(patch).mean())


def looks_like_meat(bgr: np.ndarray, min_frac: float = 0.35, min_texture: float = 25.0) -> bool:
    """Является ли вырезанная область действительно мясом.

    Предохранитель против «уверенной чепухи»: классификатор свежести обучен
    отвечать «свежее/испорчено» на ЛЮБУЮ картинку и отказываться не умеет. На
    видео цеха самой красной крупной областью оказывается кафельная плитка на
    стене — её вырезало как «мясо», и модель уверенно объявляла её испорченной.

    Плитку отсекают два признака: она пересыщена (S ~180-205 против ~50-120 у
    мяса) и абсолютно плоская по текстуре (дисперсия лапласиана ~4 против 50-135).
    """
    if bgr is None or bgr.size == 0:
        return False
    if cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < min_texture:
        return False
    return meat_fraction(bgr) >= min_frac


def meat_bbox(bgr: np.ndarray):
    """Вернуть [x1, y1, x2, y2] области мяса или None."""
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32)
    # Порог Оцу по «краснота» — устойчивее фиксированного числа к разному свету.
    a8 = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(a8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA_FRAC * w * h:
        return None
    x, y, bw, bh = cv2.boundingRect(c)
    mx, my = int(bw * MARGIN), int(bh * MARGIN)
    return [max(0, x - mx), max(0, y - my), min(w, x + bw + mx), min(h, y + bh + my)]


def crop_meat(bgr: np.ndarray) -> np.ndarray:
    """Кадр, обрезанный до области мяса (или исходный, если не нашли)."""
    box = meat_bbox(bgr)
    if box is None:
        return bgr
    x1, y1, x2, y2 = box
    if x2 - x1 < 16 or y2 - y1 < 16:
        return bgr
    return bgr[y1:y2, x1:x2]
