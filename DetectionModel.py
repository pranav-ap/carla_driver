from ultralytics import YOLO
from torch import nn
import torch


class DetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = YOLO("models/yolo11n.pt")

        if torch.cuda.is_available():
            print('Yay! Cuda')
            self.model = self.model.cuda()

    def pred(self, x):
        results = self.model.predict(
            source=x,
            # imgsz=128,
            half=torch.cuda.is_available(),
            verbose=False,
        )

        if len(results) == 0:
            return None, x

        r = results[0]
        x_plot = r.plot()  # BGR-order numpy array

        return r, x_plot
