from ultralytics import YOLO
from torch import nn
import torch
import threading


class ObjectDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.model = YOLO("models/yolo11n.pt")

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def predict(self, x):
        with self.lock:
            results = self.model.predict(
                source=x,
                # imgsz=128,
                half=torch.cuda.is_available(),
                verbose=False,
            )

        if len(results) == 0:
            return None, x

        r = results[0]
        # BGR-order numpy array
        x_plot = r.plot()

        return r, x_plot
