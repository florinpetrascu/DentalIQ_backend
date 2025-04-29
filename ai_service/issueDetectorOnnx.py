
import onnxruntime as ort
import numpy as np

from domain.issue import Issue

from PIL import ImageDraw
class IssueDetectorONNX:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.yolo_classes = [
            'Implant',
            'Prosthetic restoration',
            'Obturation',
            'Endodontic treatment',
            'Carious lesion',
            'Bone resorbtion',
            'Impacted tooth',
            'Apical periodontitis',
            'Root fragment',
            'Furcation lesion',
            'Apical surgery',
            'Root resorption',
            'Orthodontic device',
            'Surgical device'
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
        # Process the outputs

        output0 = outputs[0]
        print("Output0:", output0.shape)

        output0 = output0[0].transpose()

        print("Output0:", output0.shape)
        print(output0[0])

        boxes = output0
        print("Boxes:", boxes.shape)

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


            prob = row[4:].max()
            # print(prob)
            if prob < 0.2:
                continue
            class_id = row[4:].argmax()
            print(class_id)
            label = self.yolo_classes[class_id]

            objects.append([x1, y1, x2, y2, label, prob])

        print("len obj", len(objects))
        # apply non-maximum suppression
        objects.sort(key=lambda x: x[5], reverse=True)
        result = []
        while len(objects) > 0:
            result.append(objects[0])
            objects = [object for object in objects if self.iou(object, objects[0]) < 0.5]

        print("len rez", len(result))

        # display bounding polygons of detected objects
        # on the image
        # Draw bounding boxes
        #image = Image.open(img).convert("RGB")
        draw = ImageDraw.Draw(img)


        for x1, y1, x2, y2, label, prob in result:
            issue = Issue(name = label)

            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), f"{label} {prob:.2f}", fill="red")

        img.show()


        return result

    def detect_issues(self, image):


        print("detect_teeths")
        input, img_width, img_height = self.preprocess(image)

        print("input_tensor ", input)

        outputs = self.session.run(None, {"images": input})

        print("outputs ", outputs)
        return self.postprocess(image, outputs, img_width, img_height)

