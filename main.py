import os
import uvicorn
import logging
import darknet
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import typing as T
from typing import Optional, Tuple
import matplotlib.pyplot as plt

CONFIG="./cfg/yolov4.cfg"
DATA="./cfg/coco.data"
WEIGHTS="yolov4.weights"
THRESH=0.25
UPLOAD_FOLDER = "./images/"

network, class_names, class_colors = darknet.load_network(
    CONFIG,
    DATA,
    WEIGHTS,
    1
)

origins = [
    "http://192.168.1.158:8080" "http://127.0.0.1:8080",
    "http://localhost:8080",
]

logging.basicConfig(
    filename="predict.log",
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return ("message", "YOLOv4 Server")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    logging.info("Called Predict")
    file_location = f"images/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    print(file_location)

    image, detections = image_detection(
        file_location, network, class_names, class_colors, THRESH
    )
    
    #save_annotations(file_location, image, detections, class_names)
    darknet.print_detections(detections, True)

    # Save the bounding box image
    plt.imsave(f"images/bounding-{file.filename}", image)

    return detections

@app.get("/boundingbox")
async def boundingbox( file_name: str):

    return FileResponse(
        path=f"images/bounding-{file_name}", status_code=200
    )

def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

def image_classification(image, network, class_names):
    
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    darknet.free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])

def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = os.path.splitext(name)[0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))

def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, log_level="info") 
