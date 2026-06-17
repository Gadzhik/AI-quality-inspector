import cv2
import time
import os
import threading
from datetime import datetime
from src.camera import CameraStream
from src.analyzer import AIAnalyzer
from src.connector import TelemetryConnector
from src.dataset_collector import DatasetCollector
from src.logger import setup_logger

logger = setup_logger(name="MainApp")

def main():
    try:
        # Initialize components
        camera = CameraStream(src='production_with_realistic_defects.mp4').start()
        analyzer = AIAnalyzer()
        
        # Ping Ollama API on startup
        if not analyzer.ping():
            print("CRITICAL: AI ENGINE OFFLINE")
            logger.critical("AI ENGINE OFFLINE. Shutting down.")
            camera.stop()
            return
            
        if analyzer.check_few_shot_ready():
            logger.info("Dataset has sufficient samples (>= 3). Few-Shot Learning Mode ENABLED.")
        else:
            logger.warning("Dataset has < 3 samples in positive or negative folders. Few-Shot Learning Mode DISABLED.")

        connector = TelemetryConnector()
        collector = DatasetCollector()
        video_source = 'production_with_realistic_defects.mp4'
        
        logger.info("Press 'q' to quit, 's' to save a negative sample.")
        
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
        cv2.namedWindow("AI Quality Inspector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AI Quality Inspector", 1280, 720)
        
        # State management variables
        current_status = {
            "defect": False,
            "type": "none",
            "inference_time": None,
            "error": False,
            "last_completed_time": time.time(),
            "consecutive_defects": 0,
            "consecutive_clean": 0,
            "few_shot": False,
            "active_boxes": [],
            "alarm_end_time": 0.0,
            "snapshot_frame": None,
            "snapshot_time": 0.0
        }
        
        is_analyzing = False
        last_analysis_time = time.time()
        
        frame_buffer = []
        total_defects_count = 0
        
        while True:
            current_time = time.time()
            frame = camera.read()
            
            if frame is None:
                # Due to looping, this shouldn't happen unless camera breaks or file can't be looped
                logger.error("Failed to read frame or video ended unexpectedly")
                continue

            # FPS Calculation
            fps_counter += 1
            if current_time - fps_start_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = current_time

            # Reset status if no completed analysis for > 30 seconds
            if current_status["defect"] and (current_time - current_status["last_completed_time"]) > 30.0:
                current_status["defect"] = False
                current_status["error"] = False
                current_status["type"] = "none"
                current_status["consecutive_defects"] = 0
                current_status["consecutive_clean"] = 0

            # Буферизация кадров для выбора наилучшего по резкости
            analysis_frame = frame.copy()
            gray_frame = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
            blur_val = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
            
            # Порог низкий (2.5): исходное видео — сильно motion-blur, мясо-кадры
            # имеют var~3-6, а title/логотипы — var 40-892. Старый порог 40 пропускал
            # ТОЛЬКО резкие титры → анализатор никогда не видел дефекты на мясе.
            # >=2.5 отсекает только чёрные/замороженные кадры переходов.
            if blur_val >= 2.5:
                frame_buffer.append((blur_val, analysis_frame))
                if len(frame_buffer) > 5:
                    frame_buffer.pop(0)

            # Frame skipping (Пропуск кадров): don't launch new if one is running
            if not is_analyzing:
                if not frame_buffer:
                    continue
                    
                is_analyzing = True
                last_analysis_time = current_time
                
                # Берём САМЫЙ СВЕЖИЙ кадр, а не самый резкий: max-var на этом footage
                # систематически выбирал title/логотипы (резкие) и обходил мясо.
                # Свежий кадр репрезентативен — мясо доминирует на таймлайне.
                best_blur_val, best_frame = frame_buffer[-1]
                logger.info(f"Selected latest buffered frame with var={best_blur_val:.1f}")
                
                # Очищаем буфер для новой серии
                frame_buffer.clear()
                
                def run_analysis(captured_frame):
                    nonlocal current_status, is_analyzing, total_defects_count # Added total_defects_count to nonlocal
                    try: # Added try-except block
                        logger.info("Analyzing frame in background...")
                        start_ai_time = time.time()
                        result = analyzer.analyze_frame(captured_frame)
                        ai_time_taken = time.time() - start_ai_time
                        logger.info(f"Анализ завершен за {ai_time_taken:.2f} секунд")
                        
                        if result:
                            if result.get("error"):
                                # If JSON parsing failed, just skip the frame without triggering UNCERTAIN
                                return
                            current_status["error"] = False
                            
                            raw_defect = result.get("defect", False)
                            is_defect = raw_defect in [True, "true", "blood clot"]
                            confidence = result.get("confidence", 0)
                            
                            if is_defect:
                                boxes = result.get("boxes", [])
                                
                                # Increment defects strictly by 1 per detected frame
                                total_defects_count += 1
                                
                                current_status["defect"] = True
                                current_status["type"] = result.get("type", "Unknown")
                                current_status["consecutive_defects"] += 1
                                current_status["consecutive_clean"] = 0
                                current_status["active_boxes"] = boxes
                                current_status["last_defect_time"] = time.time()
                                current_status["last_completed_time"] = time.time()
                                current_status["inference_time"] = round(float(ai_time_taken), 2)
                                # Замораживаем ИМЕННО проанализированный кадр + держим
                                # алярм 20с: бокс рисуется на этом snapshot → совпадает
                                # с пятном (живое видео уехало бы за 45с инференса).
                                current_status["snapshot_frame"] = captured_frame.copy()
                                current_status["alarm_end_time"] = time.time() + 20.0
                            else:
                                current_status["consecutive_defects"] = 0
                                current_status["consecutive_clean"] += 1
                                current_status["inference_time"] = round(float(ai_time_taken), 2)
                                if current_status["consecutive_clean"] >= 2:
                                    current_status["defect"] = False
                        else:
                            current_status["error"] = True
                    except Exception as e:
                        logger.error(f"Error during analysis: {e}")
                        current_status["error"] = True
                    finally:
                        is_analyzing = False

                threading.Thread(target=run_analysis, args=(best_frame,), daemon=True).start()

            # Visualization
            height, width = frame.shape[:2]

            # Во время алярма показываем ЗАМОРОЖЕННЫЙ проанализированный кадр (snapshot),
            # чтобы красный бокс лёг точно на дефект, а не на уехавшее живое видео.
            if current_status["defect"] and current_status.get("snapshot_frame") is not None \
                    and current_time <= current_status.get("alarm_end_time", 0):
                frame = current_status["snapshot_frame"].copy()

            if current_status.get("few_shot", False):
                text_size = cv2.getTextSize("MODE: FEW-SHOT (AI LEARNING)", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (width//2 - text_size[0]//2 - 10, 10), (width//2 + text_size[0]//2 + 10, 40), (0, 0, 0), -1)
                cv2.putText(frame, "MODE: FEW-SHOT (AI LEARNING)", (width//2 - text_size[0]//2, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            if current_status["error"]:
                # Print API error instead of full yellow frame
                cv2.circle(frame, (width - 190, 25), 6, (0, 255, 255), -1)
                cv2.putText(frame, "NETWORK LAG", (width - 170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif current_status["defect"]:
                if current_time > current_status.get("alarm_end_time", 0):
                    current_status["defect"] = False

                if current_status["defect"]:
                    # Solid Red Alert logic, thickness 2, margin 1
                    cv2.rectangle(frame, (1, 1), (width - 1, height - 1), (0, 0, 255), 2)
                    
                    # Live streaming boxes mapping
                    boxes = current_status.get("active_boxes", [])
                    for box in boxes:
                        try:
                            if len(box) == 4:
                                ymin, xmin, ymax, xmax = box

                                # Определяем шкалу координат по максимальному значению:
                                #   <= 1.0  -> нормализовано [0..1]
                                #   <= 448  -> абсолют в 448px превью
                                #   иначе   -> шкала 0..1000 (конвенция Gemma)
                                max_coord = max(ymin, xmin, ymax, xmax)
                                if max_coord <= 1.0:
                                    x1_b, x2_b = int(xmin * width), int(xmax * width)
                                    y1_b, y2_b = int(ymin * height), int(ymax * height)
                                else:
                                    denom = 448.0 if max_coord <= 448 else 1000.0
                                    x1_b, x2_b = int(xmin / denom * width), int(xmax / denom * width)
                                    y1_b, y2_b = int(ymin / denom * height), int(ymax / denom * height)

                                cv2.rectangle(frame, (x1_b, y1_b), (x2_b, y2_b), (0, 0, 255), 3)
                        except Exception:
                            pass
                        
                    # Flashing "!!! DEFECT DETECTED !!!"
                    if int(current_time * 2) % 2 == 0:
                        text_size = cv2.getTextSize("!!! DEFECT DETECTED !!!", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
                        cv2.rectangle(frame, (width//2 - text_size[0]//2 - 20, 40), (width//2 + text_size[0]//2 + 20, 100), (0, 0, 255), -1)
                        cv2.putText(frame, "!!! DEFECT DETECTED !!!", (width//2 - text_size[0]//2, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)

            if not current_status["defect"] and not current_status["error"]:
                # Green Frame "QUALITY OK", margin 1
                cv2.rectangle(frame, (1, 1), (width - 1, height - 1), (0, 255, 0), 2)
                
                # "QUALITY OK" Label
                cv2.rectangle(frame, (50, 40), (350, 90), (0, 255, 0), -1)
                cv2.putText(frame, "QUALITY OK", (70, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
            
            # UI Indicator for AI status (red/yellow/green circle)
            if is_analyzing:
                ai_status_color = (0, 255, 255)  # Yellowish
                status_text = "AI ANALYZING..."
            else:
                ai_status_color = (0, 0, 255) if current_status["error"] else (0, 255, 0)
                status_text = "AI ERROR" if current_status["error"] else "AI IDLE"
            
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.circle(frame, (width - 450 - text_size[0] - 15, height - 40), 10, ai_status_color, -1)
            cv2.putText(frame, status_text, (width - 450 - text_size[0], height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ai_status_color, 2)

            # Stats Overlay (Model, FPS & Time) - Modified as per instruction
            stats_bg_height = 140
            cv2.rectangle(frame, (0, int(height - stats_bg_height)), (320, height), (0, 0, 0), -1) 
            
            time_since_last = current_time - current_status["last_completed_time"]
            cv2.putText(frame, f"Last update: {int(time_since_last)}s ago", (10, int(height - 100)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            model_label = analyzer.model_name.split("/")[-1].upper()
            cv2.putText(frame, f"Model: {model_label}", (10, int(height - 70)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps}", (10, int(height - 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            inference_time = current_status.get("inference_time", 0)
            time_text = f"AI Time: {round(inference_time, 2)}s" if inference_time else "AI Time: 0.00s"
            cv2.putText(frame, time_text, (width - 380, height - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # --- Timestamp & Defect Count Overlay ---
            current_date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            defect_color = (0, 0, 255) if current_status["defect"] else (255, 255, 255)
            cv2.putText(frame, f"DEFECTS FOUND: {total_defects_count}", (width - 380, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, defect_color, 2)
            cv2.putText(frame, current_date_str, (width - 380, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display the frame
            cv2.imshow("AI Quality Inspector", frame)
            
            key = cv2.waitKey(25) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                meta = {
                    "source_video": video_source,
                    "inference_time": current_status["inference_time"] if current_status["inference_time"] else 0.0,
                    "confidence": 100,
                    "type": "none",
                    "defect": False,
                    "source": "manual_label"
                }
                collector.save_positive(frame, meta)
                cv2.putText(frame, "STANDARD SAVED (POSITIVE)", (50, int(height/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
                cv2.imshow("AI Quality Inspector", frame)
                cv2.waitKey(1500) # 1.5 sec explicit wait
            elif key == ord("d"):
                meta = {
                    "source_video": video_source,
                    "inference_time": current_status["inference_time"] if current_status["inference_time"] else 0.0,
                    "confidence": 100,
                    "type": current_status["type"] if current_status["type"] != "none" else "manual",
                    "defect": True,
                    "source": "manual_label"
                }
                collector.save_negative(frame, meta)
                cv2.putText(frame, "DEFECT SAVED (NEGATIVE)", (50, int(height/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                cv2.imshow("AI Quality Inspector", frame)
                cv2.waitKey(1500) # 1.5 sec explicit wait
                
    except Exception as e:
        logger.error(f"Critical error in main loop: {e}")
    finally:
        # Cleanup
        logger.info("Shutting down...")
        if 'camera' in locals():
            camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
