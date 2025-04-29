


from domain.issue import Issue

from PIL import Image
from shapely.geometry import Polygon, Point , LineString

import numpy as np

from ai_service.teethDetectorOnnx import  TeethDetectorONNX
from ai_service.issueDetectorOnnx import IssueDetectorONNX
class AiServiceOnnx:
    def __init__(self, teeth_model_path, issue_model_path):

        self.teethDetector = TeethDetectorONNX(teeth_model_path)
        self.issueDetector = IssueDetectorONNX(issue_model_path)



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



    def load_issues1(self, teeth_list, image):
            # Obține predicțiile pentru bounding boxes
            result = self.issueDetector.detect_issues(image)
            print("result load",result)
            result = np.array(result)

            boxes = result[:,0:4]

            boxes = np.array(boxes, dtype=float).astype(int)
            labels = result[:,4]

            print("boxes ",boxes)
            # result[0].show()
            # self.display_predictions(image,result)
            # Convertim imaginea într-un format compatibil cu OpenCV
            #image_cv = np.array(image)
            print("teeth list len = ", len(teeth_list))

            # Iterăm prin fiecare dinte
            for tooth in teeth_list:
                if len(tooth.polygon) > 2:

                    tooth_polygon_shapely = Polygon(tooth.polygon)

                    for i in range(len(boxes)):
                        try:
                            class_id = labels[i]
                            x_min, y_min, x_max, y_max = boxes[i]

                            if class_id == '12' or class_id == '1':
                                print('class_id ',class_id)
                                # Dimensiunea laturii pătratului
                                square_side = min(x_max - x_min, y_max - y_min)

                                # Centrul pătratului din stânga
                                left_center = (x_min + square_side / 2, (y_min + y_max) / 2)

                                # Centrul pătratului din dreapta
                                right_center = (x_max - square_side / 2, (y_min + y_max) / 2)

                                # Creează linia din cele două puncte
                                line = LineString([left_center, right_center])

                                print('Aici123')

                                # Verifică intersecția
                                if tooth_polygon_shapely.intersects(line) or tooth_polygon_shapely.contains(line):
                                    # issue = Issue(name=self.issue_labels[class_id])
                                    issue = Issue(name=class_id)
                                    tooth.add_issue(issue)
                                    print(f"Issue PRR/SD adăugat: {issue.name}")



                            else:
                                print('class_id else', class_id)
                                # Calculăm centrul bbox-ului
                                bbox_center_x = (x_min + x_max) / 2
                                bbox_center_y = (y_min + y_max) / 2
                                bbox_center = Point(bbox_center_x, bbox_center_y)
                                print(bbox_center)
                                # Verificăm dacă centrul bbox-ului este în interiorul poligonului dintelui
                                if tooth_polygon_shapely.contains(bbox_center):
                                    # issue = Issue(name=self.issue_labels[class_id])
                                    issue = Issue(name=class_id)
                                    tooth.add_issue(issue)
                                    print(f"Issue adăugat onnx: {issue.name}")



                        except Exception as e:
                            print(f"A apărut o eroare: {e}")

                print("Dinte procesat onnx:", tooth.name)




    def scale_polygon_to_640(self, polygon, img_width, img_height):
        scale_x = 640 / img_width
        scale_y = 340 / img_height
        scaled_polygon = [(int(x * scale_x), int(y * scale_y)) for (x, y) in polygon]
        return scaled_polygon


    def get_teeths(self, image):



        img = image.copy()

        teeth_list_onnx, img_width, img_height = self.teethDetector.detect_teeths(image)

        max_x = max(pt[0] for tooth in teeth_list_onnx for pt in tooth.polygon)
        max_y = max(pt[1] for tooth in teeth_list_onnx for pt in tooth.polygon)

        print("maxime onnx ")
        print(max_x)
        print(max_y)


        self.load_issues1(teeth_list_onnx ,img)


        for teeth in teeth_list_onnx:
            teeth.polygon = self.scale_polygon_to_640(teeth.polygon,img_width,img_height)

            teeth.polygon =  [list(point) for point in teeth.polygon]
            # Conversie în listă de tupluri




        print("*************** final **************")

        print("teeth_list_onnx ",teeth_list_onnx[0:4])




        for teeth in teeth_list_onnx:
            # Elimină duplicatele din teeth.issues pe baza numelui (name)
            unique_issues = {issue.name: issue for issue in teeth.issues}.values()
            teeth.issues = list(unique_issues)


        print('get_teeths_finished')
        return teeth_list_onnx


#Theets  [Teeth(name=31, issues=['Prosthetic restoration']), Teeth(name=29, issues=['Prosthetic restoration']), Teeth(name=6, issues=['Prosthetic restoration']), Teeth(name=7, issues=['Endodontic treatment', 'Prosthetic restoration']), Teeth(name=24, issues=[]), Teeth(name=27, issues=[]), Teeth(name=28, issues=[]), Teeth(name=9, issues=[]), Teeth(name=21, issues=[]), Teeth(name=8, issues=[]), Teeth(name=20, issues=[]), Teeth(name=4, issues=['Prosthetic restoration']), Teeth(name=5, issues=['Prosthetic restoration']), Teeth(name=1, issues=['Impacted tooth']), Teeth(name=3, issues=['Prosthetic restoration']), Teeth(name=10, issues=['Prosthetic restoration']), Teeth(name=23, issues=[]), Teeth(name=14, issues=['Prosthetic restoration']), Teeth(name=22, issues=[]), Teeth(name=13, issues=['Prosthetic restoration']), Teeth(name=18, issues=[]), Teeth(name=13, issues=['Prosthetic restoration']), Teeth(name=15, issues=[]), Teeth(name=11, issues=['Prosthetic restoration']), Teeth(name=12, issues=['Prosthetic restoration']), Teeth(name=25, issues=[]), Teeth(name=10, issues=['Prosthetic restoration']), Teeth(name=2, issues=['Prosthetic restoration']), Teeth(name=14, issues=['Prosthetic restoration']), Teeth(name=26, issues=[]), Teeth(name=30, issues=['Prosthetic restoration'])]


