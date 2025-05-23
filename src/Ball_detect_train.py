from ultralytics import YOLO
from roboflow import Roboflow

# Initialize the Roboflow API
rf = Roboflow(api_key="j4zqqTmqjmKWMxsp8dCD")
project = rf.workspace("asad-ali-wxg3b").project("cricket-ball-detection-8uv1o-md7iv")


# Load a pre-trained YOLOv8 model
model = YOLO('/yolov8n.pt')  # You can choose other variants like yolov8s, yolov8m, yolov8l, yolov8x



# Train the model
results = model.train(
    data='./datasets/data.yaml',
    epochs=30,  # Number of training epochs
    imgsz=640,  # Image size
    batch=16,   # Batch size
    name='ball_detection'  # Name of the training run
)

# from roboflow import Roboflow

# # Initialize Roboflow
# rf = Roboflow(api_key="j4zqqTmqjmKWMxsp8dCD")

# # Specify your Roboflow workspace and project name
# project = rf.workspace("asad-ali-wxg3b").project("cricket-ball-detection-8uv1o-md7iv")
# print(dir(project))

# project = project.version(2)


# # Upload the trained YOLOv8 model (.pt file)
# # project.version("2").model.upload("runs/detect/ball_detection/weights/best.pt")
# # Path to your trained YOLOv8 model
# model_path = "src\\runs\\detect\\ball_detection"

# # Upload the model to Roboflow using the correct method
# # This is a hypothetical method; replace it with the actual method from the documentation
# project.deploy(model_type="yolov8",model_path=model_path)

# print("Deployed")
