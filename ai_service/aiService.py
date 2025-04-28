
from ultralytics import YOLO

from domain.issue import Issue
from domain.teeth import Teeth
from PIL import Image
from shapely.geometry import Polygon, Point , LineString

import numpy as np
import cv2
from ai_service.teethDetectorOnnx import  TeethDetectorONNX

class AiService:
    def __init__(self, teeth_model_path, issue_model_path):
          # Assuming you're using YOLO models
        self.teeth_model = YOLO(teeth_model_path)  # Load the first model
        self.issues_model = YOLO(issue_model_path)  # Load the second model
        self.teethDetector = TeethDetectorONNX('teeth_model.onnx')

        self.issue_labels = {
            0:  'Implant',
            1:  'Prosthetic restoration',
            2:  'Obturation',
            3:  'Endodontic treatment',
            4: ' Carious lesion',
            5:  'Bone resorbtion',
            6:  'Impacted tooth',
            7:  'Apical periodontitis',
            8:  'Root fragment',
            9:  'Furcation lesion',
            10: 'Apical surgery',
            11: 'Root resorption',
            12: 'Orthodontic device',
            13: 'Surgical device'
        }


    def teeth_predict(self, image_path):
        return self.teeth_model.predict(source=image_path)

    def issue_predict(self, image_path):
        return self.issues_model.predict(source=image_path)



    def scale_coordinates(self,coords, original_shape, processed_shape):
        """
        Scale coordinates from processed image size to original image size.
        """
        scale_x = original_shape[1] / processed_shape[1]  # Width scale
        scale_y = original_shape[0] / processed_shape[0]  # Height scale

        scaled_coords = [
            [int(point[0] * scale_x), int(point[1] * scale_y)] for point in coords
        ]
        return scaled_coords


    def resize_image(self, image, target_size=640):
        """
        Redimensionează imaginea astfel încât latura cea mai mare să fie 640, menținând proporțiile.

        Args:
            image (PIL.Image.Image): Imaginea de redimensionat.
            target_size (int): Dimensiunea dorită pentru latura cea mai mare.

        Returns:
            PIL.Image.Image: Imaginea redimensionată.
        """
        original_width, original_height = image.size

        # Calculează factorul de scalare
        if original_width > original_height:
            scale_factor = target_size / original_width
        else:
            scale_factor = target_size / original_height

        # Calculează dimensiunile noi
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        # Redimensionează imaginea
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image

    def load_issues(self, teeth_list, image ):
        # Obține predicțiile pentru bounding boxes
        result = self.issue_predict(image)

        boxes = result[0].boxes
        #result[0].show()
        #self.display_predictions(image,result)
        # Convertim imaginea într-un format compatibil cu OpenCV
        image_cv = np.array(image)
        print("teeth list len = ",len(teeth_list))



        # Iterăm prin fiecare dinte
        for tooth in teeth_list:
            if len(tooth.polygon) > 2:

                tooth_polygon_shapely = Polygon(tooth.polygon)



                for i in range(len(boxes)):
                    try:
                        class_id = int(boxes.cls[i].item())
                        x_min, y_min, x_max, y_max = boxes.xyxy[i].tolist()

                        if class_id == 12 or class_id == 1:

                            # Dimensiunea laturii pătratului
                            square_side = min(x_max - x_min, y_max - y_min)

                            # Centrul pătratului din stânga
                            left_center = (x_min + square_side / 2, (y_min + y_max) / 2)

                            # Centrul pătratului din dreapta
                            right_center = (x_max - square_side / 2, (y_min + y_max) / 2)



                            # Creează linia din cele două puncte
                            line = LineString([left_center, right_center])


                            # Verifică intersecția
                            if tooth_polygon_shapely.intersects(line) or tooth_polygon_shapely.contains(line):
                                #issue = Issue(name=self.issue_labels[class_id])
                                issue = Issue(name=self.issue_labels[class_id])
                                tooth.add_issue(issue)
                                print(f"Issue PRR/SD adăugat: {issue.name}")



                        else:

                            # Calculăm centrul bbox-ului
                            bbox_center_x = (x_min + x_max) / 2
                            bbox_center_y = (y_min + y_max) / 2
                            bbox_center = Point(bbox_center_x, bbox_center_y)

                            # Verificăm dacă centrul bbox-ului este în interiorul poligonului dintelui
                            if tooth_polygon_shapely.contains(bbox_center):
                                #issue = Issue(name=self.issue_labels[class_id])
                                issue = Issue(name=self.issue_labels[class_id])
                                tooth.add_issue(issue)
                                print(f"Issue adăugat: {issue.name}")



                    except Exception as e:
                        print(f"A apărut o eroare: {e}")

            print("Dinte procesat:", tooth.name)
        cv2.imwrite("processed_image.jpg", image_cv)
        print("load_issues finalizat")


        #cv2.imshow("Image with Bounding Boxes and Polygons", image_cv)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    # def load_models(self):
    #
    #     issue_model_url = "https://raw.githubusercontent.com/florinpetrascu/pdm_backend/main/issues_model.pt"
    #     teeth_model_url = "https://raw.githubusercontent.com/florinpetrascu/pdm_backend/main/teeth_model.pt"
    #
    #     issue_model_path = "issues_model.pt"
    #     teeth_model_path = "teeth_model.pt"
    #
    #     if not os.path.exists("issues_model.pt"):
    #         download_file(issue_model_url, "issues_model.pt")
    #         self.model2 = YOLO(issue_model_path)
    #         print("Fisierul issues_model.pt incarcat cu success")
    #     else:
    #         print("Fișierul issues_model.pt există deja. Nu se descarcă din nou.")
    #
    #     if not os.path.exists("teeth_model.pt"):
    #         download_file(teeth_model_url, "teeth_model.pt")
    #         self.model1 = YOLO(teeth_model_path)
    #         print("Fisierul teeth_model.pt incarcat cu success")
    #     else:
    #         print("Fișierul issues_model.pt există deja. Nu se descarcă din nou.")
    #




    def get_teeths(self, image):

        #self.load_models()

        # Run the model on the input image
        # Obține dimensiunea originală a imaginii
        resized_image = self.resize_image(image, target_size=640)

        # Rulează modelul pentru detectarea dinților
        result = self.teeth_predict(image)

        #result[0].show()
        #self.display_predictions(image, result)
        teeth_list = []  # List to store Teeth objects

        # Extract bounding boxes and segmentation masks
        boxes = result[0].boxes
        masks = result[0].masks


        if masks is not None:
            # Iterate over each detected object
            for i in range(len(masks.data)):
                # Get the polygon (contour) of the mask
                mask = masks.data[i].numpy()

                print(mask)


                contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                print(contours)
                # Choose the largest contour as the polygon (in case of multiple)
                if contours:
                    polygon = max(contours, key=cv2.contourArea).squeeze().tolist()

                    print(polygon)

                else:
                    polygon = []

                # Get the class name (label)
                class_id = int(int(boxes.cls[i]))  # Class ID
                #print("classes",boxes.cls)
                #class_name = self.class_names[class_id]  # Map ID to name

                # Create a Teeth object and add it to the list

                #teeth = Teeth(name=class_id, polygon=polygon, issues=[])

                tooth = Teeth(name = class_id,polygon = polygon)

                teeth_list.append(tooth)

        print("get teeths yolo")
        print(teeth_list)
        print("get teeths onnx")
        print("input names")
        print(self.teethDetector.input_name)
        print("output names")
        print(self.teethDetector.output_names)
        self.teethDetector.detect_teeths(image)

        self.load_issues(teeth_list,resized_image)
        for teeth in teeth_list:
            # Elimină duplicatele din teeth.issues pe baza numelui (name)
            unique_issues = {issue.name: issue for issue in teeth.issues}.values()
            teeth.issues = list(unique_issues)

        print('get_teeths_finished')
        return teeth_list


#Theets  [Teeth(name=31, issues=['Prosthetic restoration']), Teeth(name=29, issues=['Prosthetic restoration']), Teeth(name=6, issues=['Prosthetic restoration']), Teeth(name=7, issues=['Endodontic treatment', 'Prosthetic restoration']), Teeth(name=24, issues=[]), Teeth(name=27, issues=[]), Teeth(name=28, issues=[]), Teeth(name=9, issues=[]), Teeth(name=21, issues=[]), Teeth(name=8, issues=[]), Teeth(name=20, issues=[]), Teeth(name=4, issues=['Prosthetic restoration']), Teeth(name=5, issues=['Prosthetic restoration']), Teeth(name=1, issues=['Impacted tooth']), Teeth(name=3, issues=['Prosthetic restoration']), Teeth(name=10, issues=['Prosthetic restoration']), Teeth(name=23, issues=[]), Teeth(name=14, issues=['Prosthetic restoration']), Teeth(name=22, issues=[]), Teeth(name=13, issues=['Prosthetic restoration']), Teeth(name=18, issues=[]), Teeth(name=13, issues=['Prosthetic restoration']), Teeth(name=15, issues=[]), Teeth(name=11, issues=['Prosthetic restoration']), Teeth(name=12, issues=['Prosthetic restoration']), Teeth(name=25, issues=[]), Teeth(name=10, issues=['Prosthetic restoration']), Teeth(name=2, issues=['Prosthetic restoration']), Teeth(name=14, issues=['Prosthetic restoration']), Teeth(name=26, issues=[]), Teeth(name=30, issues=['Prosthetic restoration'])]


