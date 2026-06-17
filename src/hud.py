"""Профессиональный индустриальный HUD-оверлей для AI Quality Inspector.

Тёмный полупрозрачный интерфейс: верхняя/нижняя панели, угловые маркеры бокса
дефекта (не сплошной прямоугольник — выглядит как промышленный детектор),
плавный пульс тревоги (без дёрганого мигания). Все размеры масштабируются от
высоты кадра (калибровка под 1080p), чтобы UI смотрелся одинаково на любом
разрешении. cv2 рисует в координатах полного кадра — окно само масштабирует.
"""
import cv2
import math
import time

# Палитра (BGR)
DARK   = (20, 20, 22)
GREEN  = (110, 210, 120)
RED    = (60, 55, 235)
AMBER  = (40, 180, 235)
TXT    = (235, 235, 235)
MUTE   = (150, 150, 150)
WHITE  = (255, 255, 255)

FONT_H = cv2.FONT_HERSHEY_DUPLEX    # заголовки
FONT_B = cv2.FONT_HERSHEY_SIMPLEX   # тело


def _panel(img, p1, p2, color=DARK, alpha=0.58):
    ov = img.copy()
    cv2.rectangle(ov, p1, p2, color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _text(img, s, org, scale, color, thick=1, font=FONT_B):
    cv2.putText(img, s, org, font, scale, color, thick, cv2.LINE_AA)


def _tw(s, scale, thick=1, font=FONT_B):
    return cv2.getTextSize(s, font, scale, thick)[0][0]


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

    # --- Тонкая акцентная рамка кадра (зелёная норма / красная тревога) ---
    bt = max(2, int(3 * ui))
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), accent, bt)

    # --- Бокс дефекта (угловые маркеры + лейбл-тег) ---
    box = state.get("box")
    if defect and box:
        x1, y1, x2, y2 = box
        # внешний «гало» с пульсом + чёткий внутренний контур
        glow = int(6 * ui)
        _corner_box(frame, (x1 - glow, y1 - glow, x2 + glow, y2 + glow),
                    RED, max(1, int(2 * ui)))
        _corner_box(frame, box, (80, 75, 255), max(3, int(4 * ui)))

        tag = f"{state.get('label', 'DEFECT')}   {state.get('confidence', 0)}%"
        ts = 0.7 * ui
        tt = max(1, int(2 * ui))
        tw = _tw(tag, ts, tt, FONT_H)
        pad = int(12 * ui)
        ty2 = max(0, y1 - int(10 * ui))
        ty1 = ty2 - int(34 * ui)
        tx1 = x1
        tx2 = x1 + tw + pad * 2
        _panel(frame, (tx1, ty1), (tx2, ty2), RED, 0.85)
        _text(frame, tag, (tx1 + pad, ty2 - int(10 * ui)), ts, WHITE, tt, FONT_H)

    # --- Верхняя панель ---
    top_h = int(66 * ui)
    _panel(frame, (0, 0), (w, top_h), DARK, 0.60)
    cv2.line(frame, (0, top_h), (w, top_h), accent, max(1, int(2 * ui)))

    pad = int(28 * ui)
    _text(frame, "AGRIKO", (pad, int(30 * ui)), 0.95 * ui, WHITE, max(1, int(2 * ui)), FONT_H)
    brand_w = _tw("AGRIKO", 0.95 * ui, max(1, int(2 * ui)), FONT_H)
    _text(frame, "QUALITY CONTROL SYSTEM", (pad, int(52 * ui)), 0.45 * ui, MUTE, 1, FONT_B)
    # тонкий вертикальный разделитель после бренда
    sep_x = pad + brand_w + int(24 * ui)
    cv2.line(frame, (sep_x, int(16 * ui)), (sep_x, int(50 * ui)), (70, 70, 74), 1, cv2.LINE_AA)

    # правый кластер: LIVE • модель • время (выстраиваем справа налево)
    x = w - pad
    stamp = time.strftime("%Y-%m-%d  %H:%M:%S")
    sw = _tw(stamp, 0.55 * ui, 1, FONT_B)
    _text(frame, stamp, (x - sw, int(42 * ui)), 0.55 * ui, TXT, 1, FONT_B)
    x -= sw + int(24 * ui)
    cv2.line(frame, (x, int(16 * ui)), (x, int(50 * ui)), (70, 70, 74), 1, cv2.LINE_AA)
    x -= int(24 * ui)

    model = f"MODEL  {state.get('model', 'N/A')}"
    mw = _tw(model, 0.55 * ui, 1, FONT_B)
    _text(frame, model, (x - mw, int(42 * ui)), 0.55 * ui, MUTE, 1, FONT_B)
    x -= mw + int(24 * ui)
    cv2.line(frame, (x, int(16 * ui)), (x, int(50 * ui)), (70, 70, 74), 1, cv2.LINE_AA)
    x -= int(24 * ui)

    live = "LIVE"
    lw = _tw(live, 0.6 * ui, max(1, int(2 * ui)), FONT_H)
    _text(frame, live, (x - lw, int(43 * ui)), 0.6 * ui, WHITE, max(1, int(2 * ui)), FONT_H)
    # пульсирующая красная точка LIVE (пульс радиусом, цвет строго красный)
    live_r = max(4, int((5 + 2 * pulse) * ui))
    _dot(frame, (x - lw - int(16 * ui), int(38 * ui)), RED, live_r)

    # --- Баннер тревоги (плавный пульс фона, без вкл/выкл мигания) ---
    if defect:
        msg = "DEFECT DETECTED"
        bs = 0.95 * ui
        bt2 = max(2, int(2 * ui))
        bw = _tw(msg, bs, bt2, FONT_H)
        bpad = int(26 * ui)
        bx1 = w // 2 - bw // 2 - bpad
        bx2 = w // 2 + bw // 2 + bpad
        by1 = top_h + int(20 * ui)
        by2 = by1 + int(50 * ui)
        _panel(frame, (bx1, by1), (bx2, by2), RED, 0.45 + 0.30 * pulse)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (90, 85, 255), max(1, int(2 * ui)))
        _text(frame, msg, (w // 2 - bw // 2, by2 - int(16 * ui)), bs, WHITE, bt2, FONT_H)

    # --- Нижняя панель ---
    bot_h = int(58 * ui)
    by = h - bot_h
    _panel(frame, (0, by), (w, h), DARK, 0.60)
    cv2.line(frame, (0, by), (w, by), accent, max(1, int(2 * ui)))
    cy = by + int(37 * ui)

    # статус-чип слева
    if defect:
        _dot(frame, (pad, cy - int(6 * ui)), RED, max(4, int(7 * ui)))
        _text(frame, "ALERT  /  LINE STOP RECOMMENDED", (pad + int(20 * ui), cy),
              0.6 * ui, (90, 90, 255), max(1, int(2 * ui)), FONT_H)
        chip_w = _tw("ALERT  /  LINE STOP RECOMMENDED", 0.6 * ui, max(1, int(2 * ui)), FONT_H)
    else:
        _dot(frame, (pad, cy - int(6 * ui)), GREEN, max(4, int(7 * ui)))
        _text(frame, "MONITORING  /  QUALITY OK", (pad + int(20 * ui), cy),
              0.6 * ui, GREEN, max(1, int(2 * ui)), FONT_H)
        chip_w = _tw("MONITORING  /  QUALITY OK", 0.6 * ui, max(1, int(2 * ui)), FONT_H)

    # метрики по центру-слева
    mx = pad + int(20 * ui) + chip_w + int(40 * ui)
    metrics = [
        ("DEFECTS", str(state.get("defects_found", 0)), RED if defect else TXT),
        ("FPS", str(state.get("fps", 0)), AMBER),
        ("FRAME", str(state.get("frame_idx", 0)), MUTE),
    ]
    for name, val, col in metrics:
        _text(frame, name, (mx, by + int(24 * ui)), 0.42 * ui, MUTE, 1, FONT_B)
        _text(frame, val, (mx, by + int(46 * ui)), 0.6 * ui, col, max(1, int(2 * ui)), FONT_B)
        seg = max(_tw(name, 0.42 * ui), _tw(val, 0.6 * ui, max(1, int(2 * ui)))) + int(34 * ui)
        cv2.line(frame, (mx + seg - int(18 * ui), by + int(12 * ui)),
                 (mx + seg - int(18 * ui), h - int(12 * ui)), (70, 70, 74), 1, cv2.LINE_AA)
        mx += seg

    # AI-движок справа
    eng = "AI ENGINE"
    ew = _tw(eng, 0.55 * ui, 1, FONT_B)
    aw = _tw("ACTIVE", 0.55 * ui, max(1, int(2 * ui)), FONT_H)
    rx = w - pad - aw
    _text(frame, "ACTIVE", (rx, cy), 0.55 * ui, GREEN, max(1, int(2 * ui)), FONT_H)
    _dot(frame, (rx - int(14 * ui), cy - int(5 * ui)), GREEN, max(3, int(5 * ui)))
    _text(frame, eng, (rx - int(14 * ui) - int(10 * ui) - ew, cy), 0.55 * ui, MUTE, 1, FONT_B)
