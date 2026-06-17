"""Профессиональный индустриальный HUD-оверлей для AI Quality Inspector.

Текст рисуется настоящим TTF-шрифтом (Bahnschrift, DIN-стиль) через Pillow —
чёткий антиалиасинг, промышленный «приборный» вид. Векторные шрифты OpenCV
(Hershey) выглядят зубчато/убого, поэтому не используются для текста; cv2
рисует только геометрию (панели, рамки, линии, точки).

Геометрия рисуется на numpy-кадре, текст копится в список и накладывается
одним проходом Pillow в конце (одна BGR<->RGB конвертация на кадр). Все размеры
масштабируются от высоты кадра (калибровка под 1080p).
"""
import os
import cv2
import math
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Палитра (BGR — для cv2-геометрии)
DARK   = (20, 20, 22)
GREEN  = (110, 210, 120)
RED    = (60, 55, 235)
AMBER  = (40, 180, 235)
TXT    = (235, 235, 235)
MUTE   = (150, 150, 150)
WHITE  = (255, 255, 255)
SEPCOL = (70, 70, 74)

# Индустриальный шрифт. Bahnschrift = DIN 1451 (промышленная/транспортная
# типографика), средний вес — не тонкий и не жирный. Фоллбэк на Consolas/Segoe.
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/bahnschrift.ttf",
    "C:/Windows/Fonts/consolab.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
]
_FONT_PATH = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)
_font_cache = {}


def _font(px):
    px = max(9, int(round(px)))
    f = _font_cache.get(px)
    if f is None:
        f = ImageFont.truetype(_FONT_PATH, px) if _FONT_PATH else ImageFont.load_default()
        _font_cache[px] = f
    return f


def _bgr2rgb(c):
    return (c[2], c[1], c[0])


def _panel(img, p1, p2, color=DARK, alpha=0.58):
    ov = img.copy()
    cv2.rectangle(ov, p1, p2, color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _dot(img, center, color, r):
    cv2.circle(img, center, r, color, -1, cv2.LINE_AA)


def _corner_box(img, box, color, thick, frac=0.22):
    """Бокс угловыми маркерами (L-образные углы) — стиль промышленного трекера."""
    x1, y1, x2, y2 = box
    L = int(min(x2 - x1, y2 - y1) * frac)
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, thick, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, thick, cv2.LINE_AA)


def render(frame, state):
    """Нарисовать весь HUD поверх кадра. state — dict со статусом инспектора."""
    h, w = frame.shape[:2]
    ui = h / 1080.0
    pulse = 0.5 + 0.5 * math.sin(time.time() * 4.0)  # 0..1, плавный
    defect = state.get("defect", False)
    accent = RED if defect else GREEN

    texts = []  # (str, (x_baseline_left, y_baseline), font, rgb, stroke)

    def fnt(px):
        return _font(px * ui)

    def tw(s, px):
        return int(fnt(px).getlength(s))

    def T(s, x, y, px, color, stroke=0):
        texts.append((s, (int(x), int(y)), fnt(px), _bgr2rgb(color), stroke))

    # --- Тонкая акцентная рамка кадра (зелёная норма / красная тревога) ---
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), accent, max(2, int(3 * ui)))

    # --- Бокс дефекта (угловые маркеры + лейбл-тег) ---
    box = state.get("box")
    if defect and box:
        x1, y1, x2, y2 = box
        glow = int(6 * ui)
        _corner_box(frame, (x1 - glow, y1 - glow, x2 + glow, y2 + glow), RED, max(1, int(2 * ui)))
        _corner_box(frame, box, (80, 75, 255), max(3, int(4 * ui)))

        tag = f"{state.get('label', 'DEFECT')}   {state.get('confidence', 0)}%"
        tag_px = 22
        pad = int(12 * ui)
        ty2 = max(int(36 * ui), y1 - int(10 * ui))
        ty1 = ty2 - int(34 * ui)
        _panel(frame, (x1, ty1), (x1 + tw(tag, tag_px) + pad * 2, ty2), RED, 0.85)
        T(tag, x1 + pad, ty2 - int(11 * ui), tag_px, WHITE)

    # --- Верхняя панель ---
    top_h = int(66 * ui)
    _panel(frame, (0, 0), (w, top_h), DARK, 0.60)
    cv2.line(frame, (0, top_h), (w, top_h), accent, max(1, int(2 * ui)))

    pad = int(28 * ui)
    title = "AI QUALITY INSPECTOR"
    T(title, pad, int(34 * ui), 27, WHITE)
    T("MEAT DEFECT DETECTION", pad, int(54 * ui), 14, MUTE)
    sep_x = pad + tw(title, 27) + int(22 * ui)
    cv2.line(frame, (sep_x, int(16 * ui)), (sep_x, int(50 * ui)), SEPCOL, 1, cv2.LINE_AA)

    # правый кластер: LIVE • модель • время (справа налево)
    x = w - pad
    stamp = time.strftime("%Y-%m-%d   %H:%M:%S")
    T(stamp, x - tw(stamp, 19), int(43 * ui), 19, TXT)
    x -= tw(stamp, 19) + int(22 * ui)
    cv2.line(frame, (x, int(16 * ui)), (x, int(50 * ui)), SEPCOL, 1, cv2.LINE_AA)
    x -= int(22 * ui)

    model = f"MODEL  {state.get('model', 'N/A')}"
    T(model, x - tw(model, 19), int(43 * ui), 19, MUTE)
    x -= tw(model, 19) + int(22 * ui)
    cv2.line(frame, (x, int(16 * ui)), (x, int(50 * ui)), SEPCOL, 1, cv2.LINE_AA)
    x -= int(22 * ui)

    lw = tw("LIVE", 21)
    T("LIVE", x - lw, int(44 * ui), 21, WHITE)
    live_r = max(4, int((5 + 2 * pulse) * ui))
    _dot(frame, (x - lw - int(16 * ui), int(38 * ui)), RED, live_r)

    # --- Баннер тревоги (плавный пульс фона, без вкл/выкл мигания) ---
    if defect:
        msg = "DEFECT DETECTED"
        bpx = 32
        bw = tw(msg, bpx)
        bpad = int(28 * ui)
        bx1 = w // 2 - bw // 2 - bpad
        bx2 = w // 2 + bw // 2 + bpad
        by1 = top_h + int(20 * ui)
        by2 = by1 + int(52 * ui)
        _panel(frame, (bx1, by1), (bx2, by2), RED, 0.45 + 0.30 * pulse)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (90, 85, 255), max(1, int(2 * ui)))
        T(msg, w // 2 - bw // 2, by2 - int(16 * ui), bpx, WHITE)

    # --- Нижняя панель ---
    bot_h = int(58 * ui)
    by = h - bot_h
    _panel(frame, (0, by), (w, h), DARK, 0.60)
    cv2.line(frame, (0, by), (w, by), accent, max(1, int(2 * ui)))
    cy = by + int(38 * ui)

    # статус-чип слева
    if defect:
        chip = "ALERT   /   LINE STOP RECOMMENDED"
        chip_col = (90, 90, 255)
        _dot(frame, (pad, cy - int(7 * ui)), RED, max(4, int(7 * ui)))
    else:
        chip = "MONITORING   /   QUALITY OK"
        chip_col = GREEN
        _dot(frame, (pad, cy - int(7 * ui)), GREEN, max(4, int(7 * ui)))
    chip_x = pad + int(22 * ui)
    T(chip, chip_x, cy, 21, chip_col)
    chip_w = tw(chip, 21)

    # метрики по центру-слева
    mx = chip_x + chip_w + int(44 * ui)
    metrics = [
        ("DEFECTS", str(state.get("defects_found", 0)), RED if defect else TXT),
        ("FPS", str(state.get("fps", 0)), AMBER),
        ("FRAME", str(state.get("frame_idx", 0)), MUTE),
    ]
    for name, val, col in metrics:
        T(name, mx, by + int(25 * ui), 13, MUTE)
        T(val, mx, by + int(47 * ui), 21, col)
        seg = max(tw(name, 13), tw(val, 21)) + int(36 * ui)
        cv2.line(frame, (mx + seg - int(18 * ui), by + int(12 * ui)),
                 (mx + seg - int(18 * ui), h - int(12 * ui)), SEPCOL, 1, cv2.LINE_AA)
        mx += seg

    # AI-движок справа
    aw = tw("ACTIVE", 19)
    rx = w - pad - aw
    T("ACTIVE", rx, cy, 19, GREEN)
    _dot(frame, (rx - int(14 * ui), cy - int(6 * ui)), GREEN, max(3, int(5 * ui)))
    ew = tw("AI ENGINE", 19)
    T("AI ENGINE", rx - int(14 * ui) - int(10 * ui) - ew, cy, 19, MUTE)

    # --- Один проход Pillow: накладываем весь текст поверх геометрии ---
    if texts:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for s, org, font, rgb, stroke in texts:
            draw.text(org, s, font=font, fill=rgb, anchor="ls",
                      stroke_width=stroke, stroke_fill=rgb)
        frame[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
