# Custom the Weights for Object Detection using yolov5 or yolov8
This includes how to train labelled images from ROBOFLOW Platform, evaluate the results and export to onnx or tensorflowjs

### 1. Clone the repo and cd into
```shell
cd object-classifier-tng
```

### 2. Set up Python Environemnt 
Assuming using Python3.8 and above. This is setup with Python 3.10.9
```shell
python3 -m venv object-classifier-env
```
#### Activate it
```shell
source object-classifier-env/bin/activate
```
### 3. Install dependencies
```shell
pip install -r requirements-dev.txt
pip install -r requirements.txt 
```

### 4. For Yolov5 (jupyter)
```shell
cd notebook-yolov5 && jupyter lab 
```

### Or for Yolov5 (Vscode)
```shell
cd notebook-yolov5 && code .
```
### 5. For Yolov8 (jupyter)
```shell
cd notebook-yolov8 && jupyter lab 
```

### Or for Yolov8 (Vscode)
```shell
cd notebook-yolov8 && code .
```

