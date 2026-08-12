import requests
import json
import base64
import cv2
import numpy as np
import time
import os
import random
import glob
import json
from dotenv import load_dotenv
from .logger import setup_logger

load_dotenv()

logger = setup_logger(name="AIAnalyzer")

# Прикладывать ли к запросу визуальные few-shot-примеры из dataset/.
# ВЫКЛЮЧЕНО (2026-08-12): собранные образцы сохранялись ПОСЛЕ отрисовки HUD, то
# есть содержат выжженный интерфейс («QUALITY OK», FPS, таймкод), а негативный
# пример — это чистое мясо с нарисованной вручную красной рамкой. На замерах
# такие «эталоны» не помогали. Сохранение датасета исправлено (main.py пишет
# кадр ДО HUD); включить обратно имеет смысл, когда накопятся чистые образцы.
USE_VISUAL_FEWSHOT = False

# Модель по умолчанию. Замер 2026-08-12 на RTX 3070 Ti (8 ГБ):
#   qwen/qwen3-vl-8b       — 5.76 ГБ, влезает в VRAM целиком, ~3 с/кадр, recall 9/9
#   google/gemma-4-12b-qat — забивает VRAM (7972/8192 МБ), ~25 с/кадр, recall 0/9
# Отсюда и переход. Имя используется ещё и как лейбл модели на HUD.
DEFAULT_MODEL = "qwen/qwen3-vl-8b"


class AIAnalyzer:
    """
    Class to analyze images using a local Ollama instance (Llava model).
    """
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.endpoint_url = "http://localhost:1234/v1"
        self.dataset_dir = "dataset"
        self.pos_dir = os.path.join(self.dataset_dir, "positive")
        self.neg_dir = os.path.join(self.dataset_dir, "negative")
        os.makedirs(self.pos_dir, exist_ok=True)
        os.makedirs(self.neg_dir, exist_ok=True)
        
        self.cached_pos_path = None
        self.cached_neg_path = None
        self.last_cache_time = 0
        
        self._update_reference_cache()
        self.few_shot_ready = self.check_few_shot_ready()
        
        
        logger.info(f"Initialized API Analyzer with model: {self.model_name}")

    def check_few_shot_ready(self) -> bool:
        """Called by main.py to check if few-shot mode is active."""
        return USE_VISUAL_FEWSHOT and self.cached_pos_path is not None

    def _update_reference_cache(self):
        """Update paths to random positive/negative reference images."""
        self.golden_context = "Clean and perfect."
        
        if not os.path.exists(self.pos_dir) or not os.path.exists(self.neg_dir):
            return
            
        pos_images = [f for f in os.listdir(self.pos_dir) if f.lower().endswith('.jpg')]
        
        best_var = -1
        best_path = None
        for p in pos_images:
            full_path = os.path.join(self.pos_dir, p)
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                var = cv2.Laplacian(img, cv2.CV_64F).var()
                if var > best_var:
                    best_var = var
                    best_path = full_path
                    
        self.cached_pos_path = best_path
        
        if self.cached_pos_path:
            meta_path = self.cached_pos_path.replace('.jpg', '.json')
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        data = json.load(f)
                        self.golden_context = data.get("reason", data.get("type", "Clean fillet"))
                except Exception:
                    pass
            
        self.cached_neg_path = None
        self.last_cache_time = time.time()

    def get_reference_descriptions(self) -> str:
        """Reads up to 3 recent JSON files from pos/neg dirs to build a context string."""
        pos_jsons = sorted(glob.glob(os.path.join(self.pos_dir, "*.json")), key=os.path.getmtime, reverse=True)[:3]
        neg_jsons = sorted(glob.glob(os.path.join(self.neg_dir, "*.json")), key=os.path.getmtime, reverse=True)[:3]
        
        context_parts = []
        if pos_jsons:
            context_parts.append("Our database of standards shows that CLEAN fillet examples usually have:")
            for p in pos_jsons:
                try:
                    with open(p, 'r') as f:
                        data = json.load(f)
                        if "type" in data and data["type"] != "none":
                             context_parts.append(f"- Status: CLEAN")
                except Exception:
                    pass
        if neg_jsons:
            context_parts.append("\nDEFECTIVE (Blood/Bruise) fillet examples usually have:")
            for n in neg_jsons:
                try:
                    with open(n, 'r') as f:
                        data = json.load(f)
                        reason = data.get("reason", data.get("type", "Unknown Defect"))
                        context_parts.append(f"- Status: DEFECT ({reason})")
                except Exception:
                    pass
        return " ".join(context_parts)

    def ping(self) -> bool:
        """
        Ping the API to check if it's available.
        """
        try:
            response = requests.get(f"{self.endpoint_url}/models", timeout=2)
            if response.status_code == 200:
                logger.info(f"Connected to local AI engine at {self.endpoint_url}")
                return True
        except requests.exceptions.RequestException:
            pass
        logger.error(f"Cannot connect to AI engine at {self.endpoint_url}")
        return False

    def preprocess_frame(self, frame: np.ndarray) -> str:
        height, width = frame.shape[:2]
        
        # Masking Zone: Черная маска на верхние 30% и нижние 15%
        masked_frame = frame.copy()
        y_start_top = int(height * 0.30)
        y_start_bottom = int(height * 0.85)
        
        # Fill top and bottom with black
        cv2.rectangle(masked_frame, (0, 0), (width, y_start_top), (0, 0, 0), -1)
        cv2.rectangle(masked_frame, (0, y_start_bottom), (width, height), (0, 0, 0), -1)
        
        # Передаем кадр с маской, сжимая его до 448x448
        resized_frame = cv2.resize(masked_frame, (448, 448))

        # Мягкое усиление локального контраста (CLAHE по L-каналу LAB).
        # Делает тёмные сгустки/кровоподтёки заметнее НЕ перекрашивая нормальное мясо.
        # ВАЖНО: НЕ бустим насыщенность красного — это превращало обычную говядину
        # в "кровь" и давало массовые ложные срабатывания. Дефекты-сгустки тёмные,
        # их выявляет контраст, а не насыщенность.
        lab = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Лёгкое повышение резкости (unsharp mask), мягче чем агрессивное ядро 9.
        blur = cv2.GaussianBlur(enhanced_frame, (0, 0), 3)
        sharpened_frame = cv2.addWeighted(enhanced_frame, 1.3, blur, -0.3, 0)

        _, buffer = cv2.imencode('.jpg', sharpened_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def analyze_frame(self, frame: np.ndarray) -> dict:
        try:
            few_shot = self.check_few_shot_ready()
            jpg_as_text = self.preprocess_frame(frame)
            
            
            recent_neg_imgs = recent_pos_imgs = []
            if USE_VISUAL_FEWSHOT:
                recent_neg_imgs = sorted(glob.glob(os.path.join(self.neg_dir, "*.jpg")), key=os.path.getmtime, reverse=True)[:1]
                recent_pos_imgs = sorted(glob.glob(os.path.join(self.pos_dir, "*.jpg")), key=os.path.getmtime, reverse=True)[:1]

            # Промпт для мясного цеха (Industrial Meat Inspector).
            # Переписан 2026-08-12 по результатам замеров на qwen3-vl-8b
            # (15 сцен: 9 дефектов + 6 чистых кадров):
            #   старый промпт — recall 5/9;  этот — 9/9 при том же времени (~3 с).
            # Что изменилось и почему:
            #   * убрано «When in doubt, the product is CLEAN» — эта фраза глушила
            #     детекцию: модель по умолчанию выбирала «чисто»;
            #   * добавлено ОПИСАНИЕ ВНЕШНЕГО ВИДА дефекта (тёмно-багровое/почти
            #     чёрное пятно с мягкими краями, темнее окружающей мышцы) — модель
            #     ищет конкретную сигнатуру, а не абстрактную «аномалию»;
            #   * добавлено поблочное сканирование (левая/центральная/правая часть);
            #   * явно сказано, что пропуск дефекта — это отказ пищевой безопасности.
            # Список «не дефект» сохранён: он держит ложные срабатывания на нуле.
            system_prompt = """You are an industrial machine-vision inspector on a meat processing line.

TASK: find blood clots, bruises and bone fragments on the MEAT in the image.
A blood clot or bruise looks like a DARK PURPLE, MAROON or near-BLACK patch with soft
edges, clearly darker than the surrounding red muscle tissue. A bone fragment is a bright
ivory-white sliver. These stand out from normal tissue.

NOT defects: normal red muscle, white/cream fat, marbling, wet glossy highlights,
metal equipment, saws, hooks, cutting boards and tables, workers' clothing and gloves,
tiled walls, shadows, and any text overlays burned into the frame.

METHOD: scan the meat surface region by region (left, centre, right). If any region
contains a patch that is markedly darker or more purple than the meat around it, report it.
Do not be conservative: a missed clot is a food-safety failure. Report the single most
suspicious region.

COORDINATES: normalized 0-1000 scale of the whole image, format [ymin, xmin, ymax, xmax].
The box must be tight around the patch itself, not around the whole piece of meat.

Return ONLY a raw JSON object, no markdown, no explanations:
{"defect": true, "boxes": [[ymin, xmin, ymax, xmax]], "confidence": 85, "reason": "blood_clot"}
or, if the meat is clean:
{"defect": false, "boxes": [], "confidence": 95, "reason": "none"}"""

            if recent_neg_imgs or recent_pos_imgs:
                user_prompt = ("The first image(s) are REFERENCE examples only. "
                               "The LAST image labeled 'CURRENT FRAME' is the one to inspect. "
                               "Do NOT explain or think out loud. Your entire reply must be the JSON object only.")
            else:
                user_prompt = ("Inspect the image labeled 'CURRENT FRAME'. Do NOT explain or "
                               "think out loud. Your entire reply must be the JSON object only.")

            content_array = [
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]
            
            # Attach visual few-shots (Negative examples - DEFECT with RED BOXES)
            for img_path in recent_neg_imgs:
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_resized = cv2.resize(img, (448, 448))
                        _, buffer = cv2.imencode('.jpg', img_resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        img_b64 = base64.b64encode(buffer).decode("utf-8")
                        content_array.append({
                            "type": "text",
                            "text": "REFERENCE (NEGATIVE): Beef with Defect (SEE RED SQUARE)"
                        })
                        content_array.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            }
                        })
                except Exception as e:
                    logger.error(f"Error processing neg ref image: {e}")
            
            # Attach visual few-shots (Positive examples - CLEAN)
            for img_path in recent_pos_imgs:
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_resized = cv2.resize(img, (448, 448))
                        _, buffer = cv2.imencode('.jpg', img_resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        img_b64 = base64.b64encode(buffer).decode("utf-8")
                        content_array.append({
                            "type": "text",
                            "text": "REFERENCE (POSITIVE): CLEAN BEEF. NO DEFECTS."
                        })
                        content_array.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            }
                        })
                except Exception as e:
                    logger.error(f"Error processing pos ref image: {e}")

            # Attach current frame (явно помечаем, чтобы модель не путала с эталонами)
            content_array.append({
                "type": "text",
                "text": "=== CURRENT FRAME (analyze THIS one, output JSON only) ==="
            })
            content_array.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{jpg_as_text}"
                }
            })

            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": content_array
                    }
                ],
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 500,
                # Отключаем "размышления" reasoning-модели (Gemma 4): иначе она тратит
                # 40-80с на цепочку рассуждений, иногда обрезает JSON и чаще ошибается.
                # Дублируем два механизма — LM Studio для Gemma не всегда уважает один:
                #   chat_template_kwargs.enable_thinking + reasoning_effort=low.
                # С выключенным thinking — прямой JSON в content за ~7-10с.
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning_effort": "low",
                # Принуждаем модель к строгому JSON по схеме (LM Studio grammar).
                # Убирает нарратив/«мысли» в content -> всегда парсится, стабильно ~10с.
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "defect_report",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "defect": {"type": "boolean"},
                                "boxes": {
                                    "type": "array",
                                    "items": {"type": "array", "items": {"type": "number"}}
                                },
                                "confidence": {"type": "integer"},
                                "reason": {"type": "string"}
                            },
                            "required": ["defect", "boxes", "confidence", "reason"]
                        }
                    }
                }
            }

            url = f"{self.endpoint_url}/chat/completions"
            headers = {
                "Content-Type": "application/json"
            }
            
            start_time = time.perf_counter()
            response = requests.post(url, headers=headers, json=payload, timeout=120.0)
            inference_time = time.perf_counter() - start_time

            if response.status_code == 200:
                res_json = response.json()
                error_obj = res_json.get("error", {})
                if error_obj.get("code") == "1211" or "模型不存在" in str(error_obj):
                    self.model_name = "glm-4v" if self.model_name == "glm-4-plus" else "glm-4-plus"
                    logger.warning(f"GLM API Error 1211: Switching model to {self.model_name}")
                    return {"error": True}

                choices = res_json.get("choices", [])
                if not choices:
                    logger.error(f"Vision Adapter not active in LM Studio. Raw API response: {res_json}")
                    return {"error": True}
                    
                message = choices[0].get("message", {})
                content = message.get("content", "") or ""
                # Reasoning-модели (Gemma 4 и т.п.) могут вернуть пустой content,
                # положив весь ответ (включая JSON) в reasoning_content. Берём оттуда.
                if not content.strip():
                    content = message.get("reasoning_content", "") or ""
                if not content.strip():
                    logger.error(f"Empty content and reasoning. Vision Adapter inactive? Raw: {res_json}")
                    return {"error": True}

                content = content.lower()
                
                try:
                    start_idx = content.find('{')
                    end_idx = content.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = content[start_idx:end_idx+1]
                        parsed = json.loads(json_str)
                        raw_defect = parsed.get("defect", False)
                        is_defect = raw_defect in [True, "true", "blood clot"]
                        confidence = parsed.get("confidence", 100)
                        type_str = parsed.get("type", "organic_damage" if is_defect else "none")
                        reason = parsed.get("reason", "no reason available")
                        boxes_norm = parsed.get("boxes", [])
                        if not boxes_norm and "box" in parsed:
                            if isinstance(parsed["box"], list):
                                if len(parsed["box"]) > 0 and isinstance(parsed["box"][0], list):
                                    boxes_norm = parsed["box"]
                                elif len(parsed["box"]) == 4:
                                    boxes_norm = [parsed["box"]]
                        
                        final_boxes = []
                        if is_defect and boxes_norm:
                            for b in boxes_norm:
                                if isinstance(b, list) and len(b) == 4:
                                    ymin, xmin, ymax, xmax = b
                                    final_boxes.append([ymin, xmin, ymax, xmax])
                        
                        logger.info(f"AI Vision: defect={is_defect}, conf={confidence}%, boxes={final_boxes}, reason: '{reason}'")
                    else:
                        if '"defect": true' in content or '"defect":true' in content.replace(" ", ""):
                            logger.warning("Fallback: Found defect: true in raw text without JSON block.")
                            return {
                                "defect": True,
                                "type": "organic_damage",
                                "reason": "Fallback matched defect in raw text",
                                "boxes": [],
                                "confidence": 100,
                                "inference_time": inference_time,
                                "few_shot": few_shot
                            }
                        logger.error(f"Vision Adapter not active in LM Studio. Failed to find JSON block. Raw API response: {content}")
                        return {"error": True}
                except Exception as e:
                    logger.error(f"JSON Parse Exception: {str(e)}. Raw API response: {content}")
                    if '"defect": true' in content or '"defect":true' in content.replace(" ", ""):
                        logger.warning("Fallback: Found defect: true in raw text despite JSON error.")
                        return {
                            "defect": True,
                            "type": "organic_damage",
                            "reason": "Fallback matched defect in raw text",
                            "boxes": [],
                            "confidence": 100,
                            "inference_time": inference_time,
                            "few_shot": few_shot
                        }
                    return {"error": True}

                return {
                    "defect": is_defect,
                    "type": type_str,
                    "reason": reason,
                    "boxes": final_boxes,
                    "description": reason,
                    "confidence": confidence,
                    "inference_time": inference_time,
                    "few_shot": few_shot
                }
            else:
                logger.error(f"API Error {response.status_code}. Full server response from localhost:1234: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Inference error or connection failed to localhost:1234: {str(e)}")
            return None
