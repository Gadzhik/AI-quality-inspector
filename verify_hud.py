"""Headless-проверка HUD: рендерит оверлей на реальном дефектном и чистом кадре
видео и сохраняет PNG для визуальной инспекции (без GUI-окна)."""
import cv2
from src.defect_manifest import DefectManifest
from src import hud

VIDEO = "production_with_realistic_defects.mp4"
MANIFEST = "defects_manifest.json"


def grab(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def render_at(cap, manifest, idx, model, out):
    frame = grab(cap, idx)
    if frame is None:
        print(f"  no frame at {idx}")
        return
    ep = manifest.lookup(idx)
    defect = ep is not None
    state = {
        "defect": defect,
        "label": ep["label"] if defect else None,
        "confidence": ep["confidence"] if defect else 0,
        "box": ep["box"] if defect else None,
        "model": model, "fps": 30,
        "defects_found": 3 if defect else 2, "frame_idx": idx,
    }
    hud.render(frame, state)
    cv2.imwrite(out, frame)
    print(f"  frame {idx} defect={defect} -> {out}")


def main():
    manifest = DefectManifest(MANIFEST)
    print(f"episodes: {len(manifest.episodes)}")
    cap = cv2.VideoCapture(VIDEO)
    model = "GEMMA-4-12B-QAT"

    # дефектный кадр = середина первого эпизода
    if manifest.episodes:
        ep0 = manifest.episodes[0]
        mid = (ep0["start"] + ep0["end"]) // 2
        render_at(cap, manifest, mid, model, "_verify_defect.png")
        # второй эпизод для разнообразия
        if len(manifest.episodes) > 3:
            ep = manifest.episodes[3]
            render_at(cap, manifest, (ep["start"] + ep["end"]) // 2, model, "_verify_defect2.png")

    # чистый кадр = между эпизодами (start первого - 30, если есть)
    clean_idx = max(5, manifest.episodes[0]["start"] - 30) if manifest.episodes else 30
    render_at(cap, manifest, clean_idx, model, "_verify_clean.png")

    cap.release()


if __name__ == "__main__":
    main()
