"""Инференс обученного детектора свежести + локализация через CAM.

Модель обучается скриптом `train_freshness.py` на реальных фотографиях говядины
с метками из открытых датасетов. Здесь только применение.

Локализация. Классификатор говорит «испорчено», но не говорит где. Для resnet
это чинится без дообучения: архитектура заканчивается global average pooling +
линейным слоем, поэтому карта активаций последнего свёрточного блока, взвешенная
весами нужного класса, даёт Class Activation Map — тепловую карту областей,
которые склонили решение. Из неё берём bbox по порогу.

ВАЖНО: препроцессинг обязан совпадать с обучением (баланс белого -> кроп по мясу
-> Resize/CenterCrop), иначе модель увидит не то, на чём училась.
"""
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from .meat_crop import looks_like_meat, meat_bbox, white_balance

DEFAULT_MODEL_PATH = os.path.join("models", "freshness_resnet18.pt")
CACHE_MAX_SIDE = 512        # должно совпадать с train_freshness.CACHE_MAX_SIDE
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CAM_THRESHOLD = 0.55        # доля от максимума карты, по которой режем bbox


class FreshnessInspector:
    """Обёртка над обученной моделью: вердикт + уверенность + область."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, device=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Нет модели {model_path}. Сначала: python train_freshness.py")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(model_path, map_location=self.device)
        self.classes = ckpt.get("classes", ["fresh", "spoiled"])
        self.img_size = ckpt.get("img_size", 224)
        arch = ckpt.get("arch", "resnet18")

        fn = {"resnet18": models.resnet18, "resnet34": models.resnet34}[arch]
        self.model = fn(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(self.device)

        # веса последнего слоя нужны для CAM
        self.fc_weight = self.model.fc.weight.detach()
        self.features = None
        self.model.layer4.register_forward_hook(self._grab)

        self.eval_tf = transforms.Compose([
            transforms.Resize(int(self.img_size * 1.14)),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        self.model_name = f"{arch}-freshness"

    def _grab(self, module, inp, out):
        self.features = out.detach()

    def _prepare(self, bgr):
        """Повторяет валидационный препроцессинг обучения.

        Возвращает (тензор, функция преобразования координат 224-пространства
        обратно в координаты исходного кадра)."""
        wb = white_balance(bgr)
        box = meat_bbox(wb)
        if box is None:
            box = [0, 0, bgr.shape[1], bgr.shape[0]]
        x0, y0, x1, y1 = box
        crop = wb[y0:y1, x0:x1]

        # Обучение шло по кешу, где кадр после кропа уменьшался до 512 по длинной
        # стороне (INTER_AREA). Инференс обязан повторить этот шаг: без него
        # мелкая текстура остаётся другой, и точность на независимом наборе
        # проседала на 5 п.п. (60% против 65%).
        ch, cw = crop.shape[:2]
        down = CACHE_MAX_SIDE / max(ch, cw)
        if down < 1.0:
            crop = cv2.resize(crop, (max(1, int(cw * down)), max(1, int(ch * down))),
                              interpolation=cv2.INTER_AREA)
        else:
            down = 1.0
        ch, cw = crop.shape[:2]

        # Ресайз делаем ТЕМ ЖЕ кодом, что и обучение (torchvision/PIL, bilinear),
        # а не cv2: разное сглаживание даёт разные предсказания на пограничных
        # кадрах. Геометрию пересчитываем вручную, чтобы вернуть бокс CAM в кадр.
        short = int(self.img_size * 1.14)
        s = short / max(1, min(ch, cw))
        nh, nw = max(1, round(ch * s)), max(1, round(cw * s))
        ox, oy = max(0, (nw - self.img_size) // 2), max(0, (nh - self.img_size) // 2)

        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        t = self.eval_tf(pil).unsqueeze(0).to(self.device)

        def to_frame(px, py):
            # 224-пространство -> кроп после даунскейла -> исходный кадр
            return (int(x0 + (px + ox) / s / down),
                    int(y0 + (py + oy) / s / down))

        return t, to_frame, crop

    @torch.no_grad()
    def predict(self, bgr, check_meat=True):
        """Вердикт по кадру/фото. Возвращает dict с классом, уверенностью и боксом.

        `check_meat=True` — не выдавать вердикт, если в вырезанной области нет
        мяса. Без этого на видео цеха модель уверенно классифицировала красную
        кафельную плитку: классификатор обучен отвечать на любую картинку и
        воздерживаться не умеет, так что «не знаю» приходится добавлять снаружи.
        """
        t, to_frame, crop = self._prepare(bgr)
        if check_meat and not looks_like_meat(crop):
            return {"class_id": None, "label": "no_sample", "spoiled": False,
                    "confidence": 0, "box": None, "valid": False}
        logits = self.model(t)
        probs = F.softmax(logits, dim=1)[0]
        cls = int(probs.argmax())
        conf = float(probs[cls])

        box = None
        if self.features is not None:
            # CAM: карта активаций layer4, взвешенная весами класса
            cam = (self.features[0] * self.fc_weight[cls][:, None, None]).sum(0)
            cam = cam - cam.min()
            if float(cam.max()) > 0:
                cam = cam / cam.max()
                cam_np = cv2.resize(cam.cpu().numpy(), (self.img_size, self.img_size))
                mask = (cam_np >= CAM_THRESHOLD).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    x1f, y1f = to_frame(x, y)
                    x2f, y2f = to_frame(x + w, y + h)
                    box = [x1f, y1f, x2f, y2f]

        return {
            "class_id": cls,
            "label": self.classes[cls],
            "spoiled": cls == 1,
            "confidence": round(conf * 100),
            "box": box,
            "valid": True,
        }
