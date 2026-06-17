"""Демо-плеер в РЕАЛЬНОМ VLM-режиме (без манифеста).

Детекция идёт через живой инференс LM Studio (`analyzer.analyze_frame`), а не по
ground-truth манифесту. VLM тяжёлая (~45-77 с/кадр), поэтому:
  - анализ крутится в фоновом потоке (frame-skip: новый старт только после
    завершения предыдущего);
  - при дефекте замораживаем проанализированный кадр (snapshot), чтобы красный
    бокс лёг точно на пятно, а не на уехавшее за минуту живое видео.

Требует запущенный LM Studio с загруженной vision-моделью (см. README).
Демо-режим без VLM — `main.py`.
"""
import cv2
import time
import threading
from src.camera import CameraStream
from src.analyzer import AIAnalyzer
from src import hud
from src.logger import setup_logger

logger = setup_logger(name="MainVLM")

VIDEO_SOURCE = "production_with_realistic_defects.mp4"
ALARM_HOLD = 15.0   # сколько секунд держать заморозку+бокс после детекции


def to_pixels(box, w, h):
    """VLM-бокс [ymin,xmin,ymax,xmax] -> [x1,y1,x2,y2] в пикселях.
    Шкала по max-координате: <=1 нормализ., <=448 превью, иначе 0-1000 (Gemma)."""
    if not box or len(box) != 4:
        return None
    ymin, xmin, ymax, xmax = box
    m = max(box)
    if m <= 1.0:
        sx, sy = w, h
    else:
        denom = 448.0 if m <= 448 else 1000.0
        sx, sy = w / denom, h / denom
    return [int(xmin * sx), int(ymin * sy), int(xmax * sx), int(ymax * sy)]


def main():
    camera = CameraStream(src=VIDEO_SOURCE).start()
    analyzer = AIAnalyzer()
    model_label = analyzer.model_name.split("/")[-1].upper()

    if not analyzer.ping():
        logger.critical("AI ENGINE OFFLINE — запусти LM Studio с загруженной моделью.")
        camera.stop()
        return
    logger.info("VLM online. Live inference mode.")

    cv2.namedWindow("AI Quality Inspector — VLM", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Quality Inspector — VLM", 1280, 720)

    st = {"defect": False, "label": None, "confidence": 0, "box": None,
          "snapshot": None, "alarm_end": 0.0, "clean_streak": 0}
    is_analyzing = False
    defects_found = 0
    fps_t = time.time(); fps_c = 0; fps = 0

    def run_analysis(captured, w, h):
        nonlocal is_analyzing, defects_found
        try:
            t = time.time()
            res = analyzer.analyze_frame(captured)
            dt = time.time() - t
            if not res or res.get("error"):
                return
            raw = res.get("defect", False)
            is_def = raw in (True, "true", "blood clot", "blood_clot")
            if is_def:
                box = to_pixels((res.get("boxes") or [[]])[0], w, h)
                label = (res.get("reason") or res.get("type") or "DEFECT").upper().replace("_", " ")
                st.update(defect=True, label=label,
                          confidence=int(res.get("confidence", 0)),
                          box=box, snapshot=captured.copy(),
                          alarm_end=time.time() + ALARM_HOLD, clean_streak=0)
                defects_found += 1
                logger.info(f"DEFECT {label} conf {st['confidence']}% box {box} ({dt:.0f}s)")
            else:
                st["clean_streak"] += 1
                if st["clean_streak"] >= 1:
                    st.update(defect=False, box=None, snapshot=None)
                logger.info(f"clean ({dt:.0f}s)")
        except Exception as e:
            logger.error(f"analysis error: {e}")
        finally:
            is_analyzing = False

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            h, w = frame.shape[:2]
            now = time.time()

            fps_c += 1
            if now - fps_t >= 1.0:
                fps = fps_c; fps_c = 0; fps_t = now

            if not is_analyzing:
                is_analyzing = True
                threading.Thread(target=run_analysis, args=(frame.copy(), w, h),
                                 daemon=True).start()

            # заморозка проанализированного кадра, пока активна тревога
            display = frame
            active = st["defect"] and st["snapshot"] is not None and now <= st["alarm_end"]
            if active:
                display = st["snapshot"].copy()
            elif st["defect"] and now > st["alarm_end"]:
                st["defect"] = False

            state = {
                "defect": st["defect"] and active,
                "label": st["label"], "confidence": st["confidence"],
                "box": st["box"] if active else None,
                "model": model_label, "fps": fps,
                "defects_found": defects_found,
                "frame_idx": camera.get_frame_index(),
            }
            hud.render(display, state)
            cv2.imshow("AI Quality Inspector — VLM", display)

            if (cv2.waitKey(25) & 0xFF) == ord("q"):
                break
    finally:
        logger.info("Shutting down...")
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
