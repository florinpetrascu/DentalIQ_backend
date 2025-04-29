
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw
import cv2
from domain.teeth import Teeth

class TeethDetectorONNX:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.yolo_classes = [
            "1", "2", "3", "4", "5", "6", "7", "8",
            "9", "10", "11", "12", "13", "14", "15", "16",
            "17", "18", "19", "20", "21", "22", "23", "24",
            "25", "26", "27", "28", "29", "30", "31", "32",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"
        ]


    def intersection(self, box1, box2):
        box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
        box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)
        return (x2 - x1) * (y2 - y1)

    def union(self, box1, box2):
        box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
        box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        return box1_area + box2_area - self.intersection(box1, box2)

    def iou(self, box1, box2):
        return self.intersection(box1, box2) / self.union(box1, box2)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # parse segmentation mask
    def get_mask(self, row, box, img_width, img_height):
        # convert mask to image (matrix of pixels)
        mask = row.reshape(160, 160)
        mask = self.sigmoid(mask)
        mask = (mask > 0.5).astype("uint8") * 255
        # crop the object defined by "box" from mask
        x1, y1, x2, y2 = box
        mask_x1 = round(x1 / img_width * 160)
        mask_y1 = round(y1 / img_height * 160)
        mask_x2 = round(x2 / img_width * 160)
        mask_y2 = round(y2 / img_height * 160)
        mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]
        # resize the cropped mask to the size of object
        img_mask = Image.fromarray(mask, "L")
        img_mask = img_mask.resize((round(x2 - x1), round(y2 - y1)))
        mask = np.array(img_mask)
        return mask

    # calculate bounding polygon from mask
    def get_polygon(self, mask):
        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        polygon = [[contour[0][0], contour[0][1]] for contour in contours[0][0]]
        return polygon

    def scale_polygon_to_640(self, polygon, img_width, img_height):
        scale_x = 640 / img_width
        scale_y = 640 / img_height
        scaled_polygon = [(int(x * scale_x), int(y * scale_y)) for (x, y) in polygon]
        return scaled_polygon

    def preprocess(self, img):
        img_width, img_height = img.size
        img = img.convert("RGB")
        img = img.resize((640, 640))

        input = np.array(img)
        input = input.transpose(2, 0, 1)
        input = input.reshape(1, 3, 640, 640).astype("float32")
        input = input / 255.0
        return input , img_width, img_height

    def postprocess(self, img, outputs, img_width, img_height):
        output0 = outputs[0]
        output1 = outputs[1]

        output0 = output0[0].transpose()
        output1 = output1[0]

        boxes = output0[:, 0:56]
        masks = output0[:, 56:]

        output1 = output1.reshape(32, 160 * 160)
        print(masks.shape, output1.shape)

        # Join matrices by matrix multiplication
        masks = masks @ output1
        print(masks.shape)

        # Connect masks to boxes: move all columns from masks matrix to the right of boxes matrix
        boxes = np.hstack([boxes, masks])
        print(boxes.shape)

        # parse and filter all boxes
        objects = []
        for row in boxes:
            xc, yc, w, h = row[:4]
            x1 = (xc - w / 2) / 640 * img_width
            y1 = (yc - h / 2) / 640 * img_height
            x2 = (xc + w / 2) / 640 * img_width
            y2 = (yc + h / 2) / 640 * img_height
            # x1 = xc - w / 2
            # y1 = yc - h / 2
            # x2 = xc + w / 2
            # y2 = yc + h / 2

            prob = row[4:56].max()
            if prob < 0.5:
                continue
            class_id = row[4:56].argmax()
            label = self.yolo_classes[class_id]


            mask = self.get_mask(row[56:25684], (x1, y1, x2, y2), img_width, img_height)
            polygon = self.get_polygon(mask)
            objects.append([x1, y1, x2, y2, label, prob, mask, polygon])

        # apply non-maximum suppression
        objects.sort(key=lambda x: x[5], reverse=True)
        result = []
        while len(objects) > 0:
            result.append(objects[0])
            objects = [object for object in objects if self.iou(object, objects[0]) < 0.7]

        print("len result ",len(result))
        print("result ",result)

        # display bounding polygons of detected objects
        # on the image
        # img = Image.open(img_name)
        draw = ImageDraw.Draw(img, "RGBA")

        teeth_list = []

        for object in result:
            [x1, y1, x2, y2, label, prob, mask, polygon] = object
            # move polygon from (0,0) to the top left point of detected object
            print("poligon nescalat tupluri")
            print(polygon)
            polygon = [(round(x1 + point[0]), round(y1 + point[1])) for point in polygon]

            #polygon = self.scale_polygon_to_640(polygon,img_width,img_height)

            draw.polygon(polygon, fill=(0, 255, 0, 125))

            print("poligon nescalat liste ", polygon)

            teeth = Teeth(name = label, polygon = polygon)

            teeth_list.append(teeth)

            print(teeth)

        #img.show()

        return teeth_list






    def detect_teeths(self, image):

        print("detect_teeths")
        input , img_width, img_height = self.preprocess(image)

        print("input_tensor ",input)

        outputs = self.session.run(None, {"images":input})

        print("outputs ", outputs)
        return self.postprocess(image ,outputs, img_width, img_height), img_width, img_height

