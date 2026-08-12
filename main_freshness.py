"""Демонстрация ОБУЧЕННОЙ модели на РЕАЛЬНЫХ фотографиях мяса.

Отличие от остальных точек входа:
  * `main.py`      — синтетические дефекты, детекция из ground-truth манифеста;
  * `main_vlm.py`  — живой инференс готовой VLM (без обучения) по видео;
  * `main_freshness.py` (этот) — модель, ОБУЧЕННАЯ нами на реальных размеченных
    фото говядины. Ни синтетики, ни подсказок: вердикт целиком от модели,
    область — из карты активаций (CAM).

По умолчанию гоняет независимый набор из другого исследования (`data/beef_indep`),
то есть данные, которых модель в обучении не видела вообще, и печатает, где она
права, а где ошиблась.

`--source` принимает и видеофайл. Учтите: модель обучена на студийных фото
отдельных кусков мяса, поэтому на видео цеха её вывод — не результат, а
демонстрация границ применимости (другой ракурс, свет, масштаб, в кадре люди и
оборудование). Смотреть на это стоит именно как на проверку «а что будет».

Запуск:
    python main_freshness.py                       # независимый набор + отчёт
    python main_freshness.py --source data/_cache/locbeef/spoiled
    python main_freshness.py --source production.mp4 --show
"""
import argparse
import glob
import os
import threading
import time

import cv2

from src import box_refine, hud
from src.freshness import FreshnessInspector
from src.meat_regions import find_meat_regions
from src.logger import setup_logger

logger = setup_logger(name="Freshness")

INDEP_DIR = os.path.join("data", "beef_indep")
OUT_DIR = "freshness_results"

# Разметка независимого набора (1-й день против 5-го) — только чтобы посчитать,
# сколько модель угадала. Модели она не показывается.
INDEP_LABELS = {
    "20210830": 0, "20210903": 1,
    "20210906": 0, "20210910": 1,
    "20210913": 0, "20210917": 1,
}


def truth_for(path):
    date = os.path.basename(path).split("_")[0]
    return INDEP_LABELS.get(date)


MIN_CONFIDENCE = 85     # ниже — считаем, что модель не уверена, и молчим
ALARM_STREAK = 3        # столько уверенных вердиктов подряд нужно для тревоги
BOX_HOLD = 1.5          # сколько секунд держать рамку после анализа


def run_video(args, inspector):
    """Прогон обученной модели по видео с ПЛАВНЫМ воспроизведением.

    Видео крутится с родной частотой кадров, а инференс идёт в фоновом потоке и
    обновляет вердикт по мере готовности — иначе получается слайд-шоу (кадр раз в
    несколько секунд), как было в первой версии.

    Тревога поднимается только если совпали три условия: в кадре действительно
    мясо, уверенность выше порога и подряд идёт несколько таких вердиктов. Без
    этого модель уверенно объявляла испорченной красную плитку на стене.

    Разметки для видео нет, поэтому точность тут не считается — это проверка
    применимости модели, обученной на студийных фото, к съёмке цеха.
    """
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        logger.error(f"Не открывается видео: {args.source}")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay = max(1, int(1000 / fps))
    logger.info(f"Видео {args.source}: {total} кадров, {fps:.0f} fps")

    # `label` — текущее состояние анализа, `alarm_label` — подпись К САМОЙ ТРЕВОГЕ.
    # Это разные вещи: рамка держится BOX_HOLD секунд, и за это время успевает
    # отработать следующий анализ. Раньше поле было одно, и активная красная рамка
    # получала подпись от нового вердикта — на записи демо видно «НЕТ ОБРАЗЦА 95%»
    # и «FRESH 95%» на рамке дефекта.
    st = {"spoiled": False, "confidence": 0, "box": None, "until": 0.0,
          "streak": 0, "label": "—", "alarm_label": "DEFECT",
          "regions": [], "defects": [], "seen": 0.0}
    stats = {"fresh": 0, "spoiled": 0, "no_sample": 0, "unsure": 0,
             "alarms": 0, "frames_with_meat": 0, "defect_boxes": 0}
    analyzing = [False]

    def analyze(frame, idx):
        """Полный проход по кадру: найти мясо -> оценить свежесть -> искать дефекты."""
        try:
            regions = find_meat_regions(frame)
            st["regions"] = [r["box"] for r in regions]
            st["seen"] = time.time()
            if not regions:
                stats["no_sample"] += 1
                st.update(streak=0, label="НЕТ МЯСА", defects=[])
                return
            stats["frames_with_meat"] += 1

            defects, verdicts = [], []
            for r in regions:
                x1, y1, x2, y2 = r["box"]
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # 1) свежесть — обученной моделью, по каждой области отдельно
                res = inspector.predict(crop)
                if res["valid"]:
                    stats[res["label"]] += 1
                    verdicts.append(res)
                    if res["confidence"] < MIN_CONFIDENCE:
                        stats["unsure"] += 1

                # 2) дефекты — поиск тёмных сгустков внутри области (CV, не модель)
                d = box_refine.refine(crop, zone=False)
                if d:
                    defects.append([x1 + d[0], y1 + d[1], x1 + d[2], y1 + d[3]])

            st["defects"] = defects
            stats["defect_boxes"] += len(defects)

            spoiled = [v for v in verdicts if v["spoiled"] and v["confidence"] >= MIN_CONFIDENCE]
            if defects:
                # Найденный сгусток — сам по себе повод для тревоги, накопления не ждём:
                # это прямое наблюдение пятна, а не вероятностный вердикт классификатора.
                if not st["spoiled"]:
                    stats["alarms"] += 1
                st.update(spoiled=True, confidence=95, box=defects[0],
                          until=time.time() + BOX_HOLD, alarm_label="DEFECT")
                logger.info(f"кадр {idx}: ТРЕВОГА тёмное пятно {defects[0]}")
            elif spoiled:
                st["streak"] += 1
                if st["streak"] >= ALARM_STREAK:
                    if not st["spoiled"]:
                        stats["alarms"] += 1
                    best = max(spoiled, key=lambda v: v["confidence"])
                    st.update(spoiled=True, confidence=best["confidence"], box=best["box"],
                              until=time.time() + BOX_HOLD, alarm_label="SPOILED")
                    logger.info(f"кадр {idx}: ТРЕВОГА spoiled {best['confidence']}%")
            elif verdicts:
                st.update(streak=0, label="FRESH")
            else:
                # Мясо в кадре есть, но ни одна область не прошла проверку
                # `looks_like_meat` внутри инференса — модель воздержалась.
                # Это НЕ то же самое, что «мяса нет», и путать их нельзя.
                st.update(streak=0, label="ВОЗДЕРЖАЛАСЬ")
        except Exception as e:
            logger.error(f"ошибка анализа: {e}")
        finally:
            analyzing[0] = False

    # Запись демо-ролика прямо из конвейера: так его можно пересобрать одной
    # командой после любой правки, вместо переснятия экрана (в записи экрана
    # остаются рамка окна и панель задач, а баги фиксируются навсегда).
    writer = None
    if args.record:
        os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (args.record_width,
                                       int(args.record_width * 9 / 16)))
        logger.info(f"запись демо -> {args.record}")

    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        now = time.time()

        if not analyzing[0]:
            analyzing[0] = True
            threading.Thread(target=analyze, args=(frame.copy(), idx), daemon=True).start()

        active = st["spoiled"] and now <= st["until"]
        if st["spoiled"] and not active:
            st["spoiled"] = False

        state = {
            "defect": active,
            "label": st["alarm_label"] if active else None,
            "confidence": st["confidence"] if active else 0,
            "box": st["box"] if active else None,
            "model": inspector.model_name.upper(),
            "fps": int(fps), "defects_found": stats["alarms"], "frame_idx": idx,
            # Честный статус: «QUALITY OK» пишем только когда модель реально
            # оценила образец и он свежий. Остальные состояния называем своими
            # именами, чтобы отсутствие анализа не выглядело как успешная проверка.
            "status": None if active else {
                "НЕТ МЯСА": "НЕТ ОБРАЗЦА   /   МЯСО В КАДРЕ НЕ НАЙДЕНО",
                "ВОЗДЕРЖАЛАСЬ": "ОБРАЗЕЦ НЕ ОЦЕНЁН   /   МОДЕЛЬ ВОЗДЕРЖАЛАСЬ",
            }.get(st["label"]),
        }
        vis = frame.copy()
        # Найденные области мяса рисуем сами: HUD рассчитан на одну рамку дефекта,
        # а областей может быть несколько. Жёлтый — «здесь ищем», красный — дефект.
        if time.time() - st["seen"] <= BOX_HOLD:
            for rb in st["regions"]:
                cv2.rectangle(vis, (rb[0], rb[1]), (rb[2], rb[3]), (0, 220, 220), 2)
                cv2.putText(vis, "MEAT", (rb[0] + 6, rb[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 220), 2, cv2.LINE_AA)
            for db in st["defects"]:
                cv2.rectangle(vis, (db[0], db[1]), (db[2], db[3]), (0, 0, 255), 3)
        hud.render(vis, state)
        if active and saved < 12:
            cv2.imwrite(os.path.join(OUT_DIR, f"video_{idx:05d}.jpg"),
                        cv2.resize(vis, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 88])
            saved += 1

        if writer is not None:
            writer.write(cv2.resize(vis, (args.record_width,
                                          int(args.record_width * 9 / 16))))

        if args.show:
            cv2.imshow("Freshness on video", cv2.resize(vis, (1280, 720)))
            if (cv2.waitKey(delay) & 0xFF) == ord("q"):
                break
        if args.limit and idx >= args.limit:
            break

    cap.release()
    if writer is not None:
        writer.release()
        logger.info(f"демо записано: {args.record}")
    if args.show:
        cv2.destroyAllWindows()
    print(f"\nпоказано кадров: {idx} из {total}")
    print(f"анализов: мясо не найдено {stats['no_sample']}, "
          f"fresh {stats['fresh']}, spoiled {stats['spoiled']} "
          f"(из них неуверенных {stats['unsure']})")
    print(f"тревог поднято: {stats['alarms']}")
    print("разметки для видео нет — точность тут не считается, это проверка применимости")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=INDEP_DIR, help="папка с фото")
    ap.add_argument("--model", default=os.path.join("models", "freshness_resnet18.pt"))
    ap.add_argument("--limit", type=int, default=0, help="ограничить число кадров")
    ap.add_argument("--show", action="store_true", help="показывать окно")
    ap.add_argument("--record", default="", help="записать демо-ролик в этот файл (видео)")
    ap.add_argument("--record-width", type=int, default=1280, help="ширина записи")
    args = ap.parse_args()

    is_video = args.source.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    if is_video:
        files = []
    else:
        files = sorted(f for f in glob.glob(os.path.join(args.source, "*"))
                       if f.lower().endswith((".jpg", ".jpeg", ".png")) and "_seg" not in f)
        if args.limit:
            files = files[:args.limit]
        if not files:
            logger.error(f"Нет изображений в {args.source}")
            return

    inspector = FreshnessInspector(args.model)
    logger.info(f"Модель: {inspector.model_name}, устройство: {inspector.device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    if is_video:
        return run_video(args, inspector)

    correct = graded = spoiled_found = processed = 0
    for i, path in enumerate(files, 1):
        img = cv2.imread(path)
        if img is None:
            continue
        processed += 1
        res = inspector.predict(img)
        spoiled_found += int(res["spoiled"])

        truth = truth_for(path)
        mark = ""
        if truth is not None:
            graded += 1
            ok = int(res["spoiled"]) == truth
            correct += ok
            mark = "верно" if ok else f"ОШИБКА (на самом деле {'испорчено' if truth else 'свежее'})"

        state = {
            "defect": res["spoiled"],
            "label": "SPOILED" if res["spoiled"] else None,
            "confidence": res["confidence"],
            "box": res["box"] if res["spoiled"] else None,
            "model": inspector.model_name.upper(),
            "fps": 0, "defects_found": spoiled_found, "frame_idx": i,
        }
        vis = img.copy()
        hud.render(vis, state)
        out = os.path.join(OUT_DIR, os.path.basename(path))
        cv2.imwrite(out, cv2.resize(vis, (1280, int(1280 * vis.shape[0] / vis.shape[1]))),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        logger.info(f"{os.path.basename(path)}: {res['label']} {res['confidence']}% {mark}")

        if args.show:
            cv2.imshow("Freshness", cv2.resize(vis, (1000, int(1000 * vis.shape[0] / vis.shape[1]))))
            if (cv2.waitKey(700) & 0xFF) == ord("q"):
                break

    if args.show:
        cv2.destroyAllWindows()

    # Считаем именно обработанные кадры: просмотр можно прервать клавишей `q`,
    # и печатать полный размер папки в этом случае — врать.
    tail = "" if processed == len(files) else f" из {len(files)} (просмотр прерван)"
    print(f"\nобработано {processed}{tail}, «испорчено» у модели: {spoiled_found}")
    if graded:
        print(f"точность на размеченных: {correct}/{graded} = {correct / graded:.1%}")
    print(f"кадры с HUD -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
