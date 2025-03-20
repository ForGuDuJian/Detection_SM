import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('') # select your model.pt path
    model.predict(source='',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  # classes=0,
                )