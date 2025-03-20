import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'')

    model.train(data=r"datasets",

                task='detect',
                cache=False,
                imgsz=640,
                epochs=100,
                single_cls=False,
                batch=4,
                close_mosaic=0,
                workers=0,
                device='0',
                optimizer='SGD',
                project='runs/train',
                name='exp',
                )

