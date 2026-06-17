import cv2
import time
from src.camera import CameraStream
from src.analyzer import AIAnalyzer
from src.dataset_collector import DatasetCollector
from src.defect_manifest import DefectManifest
from src import hud
from src.logger import setup_logger

logger = setup_logger(name="MainApp")

VIDEO_SOURCE = "production_with_realistic_defects.mp4"
MANIFEST_PATH = "defects_manifest.json"


def main():
    try:
        camera = CameraStream(src=VIDEO_SOURCE).start()
        manifest = DefectManifest(MANIFEST_PATH)
        collector = DatasetCollector()

        # Анализатор нужен лишь для лейбла модели в HUD — не пингуем LM Studio,
        # детекция идёт по ground-truth манифесту (мгновенно, без VLM-лага).
        try:
            model_label = AIAnalyzer().model_name.split("/")[-1].upper()
        except Exception:
            model_label = "GEMMA-4-12B-QAT"

        if not manifest.loaded:
            logger.warning("No manifest loaded — running without defect detection. "
                           "Run create_realistic_defects.py to generate it.")

        cv2.namedWindow("AI Quality Inspector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AI Quality Inspector", 1280, 720)

        logger.info("Press 'q' to quit, 's'/'d' to save samples.")

        fps_start = time.time()
        fps_counter = 0
        fps = 0
        defects_found = 0
        last_episode_start = None  # чтобы считать каждый эпизод один раз

        while True:
            frame = camera.read()
            if frame is None:
                logger.error("Failed to read frame or video ended unexpectedly")
                continue

            frame_idx = camera.get_frame_index()

            # FPS
            fps_counter += 1
            now = time.time()
            if now - fps_start >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start = now

            # --- Детекция: мгновенный lookup по индексу кадра (ground truth) ---
            ep = manifest.lookup(frame_idx)
            defect = ep is not None

            if defect and ep["start"] != last_episode_start:
                # Новый эпизод дефекта вошёл в кадр — считаем один раз
                last_episode_start = ep["start"]
                defects_found += 1
                logger.info(f"DEFECT @frame {frame_idx}: {ep['label']} "
                            f"conf {ep['confidence']}% box {ep['box']}")

            state = {
                "defect": defect,
                "label": ep["label"] if defect else None,
                "confidence": ep["confidence"] if defect else 0,
                "box": ep["box"] if defect else None,
                "model": model_label,
                "fps": fps,
                "defects_found": defects_found,
                "frame_idx": frame_idx,
            }
            hud.render(frame, state)

            cv2.imshow("AI Quality Inspector", frame)

            key = cv2.waitKey(25) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                collector.save_positive(frame, {
                    "source_video": VIDEO_SOURCE, "confidence": 100,
                    "type": "none", "defect": False, "source": "manual_label",
                })
                logger.info("Saved positive (clean) sample.")
            elif key == ord("d"):
                collector.save_negative(frame, {
                    "source_video": VIDEO_SOURCE, "confidence": 100,
                    "type": ep["type"] if defect else "manual",
                    "defect": True, "source": "manual_label",
                })
                logger.info("Saved negative (defect) sample.")

    except Exception as e:
        logger.error(f"Critical error in main loop: {e}")
    finally:
        logger.info("Shutting down...")
        if "camera" in locals():
            camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
