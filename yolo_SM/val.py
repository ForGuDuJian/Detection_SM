import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('xx.pt')
    model.val(data=r'xx.yaml',
              split='val',
              imgsz=640,
              batch=16,
              project='runs/val',
              name='exp',
              )