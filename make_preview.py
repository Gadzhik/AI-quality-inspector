"""Генерация README-ассетов из пайплайна (headless, без записи экрана):
рендерит HUD поверх сегмента видео -> _preview_raw.mp4 (для GIF) и сохраняет
кадр с дефектом -> assets/screenshot.jpg. Гарантирует актуальный HUD без
сторонних надписей. GIF собирается отдельно через ffmpeg."""
import cv2
from src.analyzer import DEFAULT_MODEL
from src.defect_manifest import DefectManifest
from src import hud

VIDEO = "production_with_realistic_defects.mp4"
MANIFEST = "defects_manifest.json"
MODEL = DEFAULT_MODEL.split("/")[-1].upper()

# Границы превью и кадр для скриншота считаем ПО МАНИФЕСТУ: раньше они были
# захардкожены под конкретное демо-видео и после перегенерации указывали в пустоту.
START = END = SHOT_FRAME = 0


def build_state(ep, idx, defects_found, fps=25):
    defect = ep is not None
    return {
        "defect": defect,
        "label": ep["label"] if defect else None,
        "confidence": ep["confidence"] if defect else 0,
        "box": ep["box"] if defect else None,
        "model": MODEL, "fps": fps,
        "defects_found": defects_found, "frame_idx": idx,
    }


def pick_range(manifest):
    """Отрезок «чисто -> дефект -> чисто -> дефект -> чисто» вокруг первых двух
    эпизодов манифеста и кадр из середины первого эпизода для скриншота."""
    eps = manifest.episodes
    if not eps:
        return 0, 400, 100
    first = eps[0]
    last = eps[min(1, len(eps) - 1)]
    start = max(0, first["start"] - 60)
    end = last["end"] + 60
    shot = (first["start"] + first["end"]) // 2
    return start, end, shot


def main():
    global START, END, SHOT_FRAME
    manifest = DefectManifest(MANIFEST)
    START, END, SHOT_FRAME = pick_range(manifest)
    cap = cv2.VideoCapture(VIDEO)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out = cv2.VideoWriter("_preview_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, START)

    defects_found = 1
    last_ep = None
    for idx in range(START, END):
        ok, frame = cap.read()
        if not ok:
            break
        ep = manifest.lookup(idx)
        if ep and ep["start"] != last_ep:
            last_ep = ep["start"]
            defects_found += 1
        hud.render(frame, build_state(ep, idx, defects_found, int(fps)))
        out.write(frame)
    out.release()
    print(f"preview frames {START}-{END} -> _preview_raw.mp4")

    # Скриншот: отдельный кадр с дефектом, полный кадр
    cap.set(cv2.CAP_PROP_POS_FRAMES, SHOT_FRAME)
    ok, frame = cap.read()
    if ok:
        hud.render(frame, build_state(manifest.lookup(SHOT_FRAME), SHOT_FRAME, 3))
        cv2.imwrite("assets/screenshot.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("screenshot -> assets/screenshot.jpg")
    cap.release()


if __name__ == "__main__":
    main()
