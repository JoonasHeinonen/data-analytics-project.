from imageai.Detection import ObjectDetection
import os

path = os.getcwd()

obj_detector = ObjectDetection()
obj_detector.setModelTypeAsRetinaNet()
obj_detector.setModelPath( os.path.join(path, "resnet50_coco_best_v2.0.1.h5"))
obj_detector.loadModel()
detections = obj_detector.detectObjectsFromImage(input_image=os.path.join(path, "apple.png"), output_image_path=os.path.join(path, "result_image.jpg"))