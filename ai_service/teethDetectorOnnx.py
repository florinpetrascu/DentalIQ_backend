
import onnxruntime as ort
import numpy as np
import cv2
from domain.teeth import Teeth
class TeethDetectorONNX:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def preprocess(self, image, target_size=640):
        image = np.array(image)
        # Resize and normalize
        print("preprocess")
        resized = cv2.resize(image, (target_size, target_size))
        print("preprocess1")
        resized = resized.astype(np.float32) / 255.0
        print("preprocess2")
        resized = np.transpose(resized, (2, 0, 1))  # channels first
        print("preprocess3")
        resized = np.expand_dims(resized, axis=0)  # batch dim
        print("preprocess4")
        return resized, image.shape[1], image.shape[0]  # also return original width, height

    def postprocess(self, outputs, orig_w, orig_h, conf_thres=0.2):
        preds = outputs[0]  # (1, 88, 8400)
        mask_protos = outputs[1]  # (1, 32, 160, 160)
        num_classes = 52  # <- ⚡ adaugă aici câte clase ai

        preds = preds[0]  # (88, 8400)
        mask_protos = mask_protos[0]  # (32, 160, 160)

        preds = preds.transpose(1, 0)  # (8400, 88)

        print("preds0 = ",preds[0],"len = ",len(preds[0]))

        print("preds1 = ",preds[1],"len = ",len(preds[1]))
        bbox = preds[:, :4]
        class_probs = preds[:, 4:4 + num_classes]
        mask_coeffs = preds[:, 4 + num_classes:]
        print("bbox len ",len(bbox))
        print("class_probs len", len(class_probs[0]))




        print("bbox ",bbox[0])
        print("class_probs ",class_probs[0] , "len ",len(class_probs[0]))
        print("mask_coeffs ",mask_coeffs[0] , "len ",len(mask_coeffs[0]))

        teeth_list = []

        print("preds shape len = ",preds.shape[0])

        teeth_dict = {}

        # Iterează prin toate predicțiile
        for i in range(preds.shape[0]):

            # Extrage probabilitățile pentru fiecare clasă
            class_probs_i = class_probs[i]

            # Caută probabilitatea maximă și clasa asociată
            max_class_prob = np.max(class_probs_i)
            class_id = np.argmax(class_probs_i)

            # Adaugă un tuplu cu (indice predicție, probabilitate maximă) în dicționar
            if class_id not in teeth_dict:
                teeth_dict[class_id] = [(i, max_class_prob)]  # Inițializează lista cu primul tuplu
            else:
                teeth_dict[class_id].append((i, max_class_prob))  # Adaugă tuplu la lista existentă

        # Filtrează dictionarul pentru a păstra doar tuplul cu probabilitatea maximă pentru fiecare clasă
        for class_id in teeth_dict:
            # Obține lista de tupluri pentru clasa respectivă
            prediction_list = teeth_dict[class_id]

            # Alege tuplul cu probabilitatea maximă
            best_prediction = max(prediction_list, key=lambda x: x[1])

            # Actualizează în dictionar pentru fiecare clasă cu tuplul cu probabilitatea maximă
            teeth_dict[class_id] = best_prediction

        print("teeth dict ", teeth_dict)


        for class_id in list(teeth_dict.keys()):  # Iterează printr-o copie a cheilor pentru a evita erori în timp ce modifici dicționarul

            val = teeth_dict[class_id]

            # Verifică dacă probabilitatea maximă este mai mare sau egală cu 0.3
            if val[1] < conf_thres:
                # Dacă probabilitatea este mai mică decât 0.3, elimină clasa din dicționar
                del teeth_dict[class_id]


        print(sorted(teeth_dict.keys()))

        return teeth_list
        # for i in range(preds.shape[0]):
        #
        #         x1, y1, x2, y2 = bbox[i]
        #         class_probs_i = class_probs[i]
        #         class_id = int(np.argmax(class_probs_i))  # <- ia clasa cu cea mai mare probabilitate
        #
        #
        #         # Caută probabilitatea maximă și clasa asociată
        #         max_class_prob = np.max(class_probs_i)
        #         class_id = np.argmax(class_probs_i)
        #
        #         # Setează un prag de confidență, de exemplu 0.5
        #
        #         if not teeth_dict[str(class_id)]:
        #             teeth_dict[str(class_id)] = [max_class_prob]
        #         else:
        #             teeth_dict[str(class_id)].append(max_class_prob)
        #
        #         #object_confidence = preds[i, 4]  # Probabilitatea obiectului (fără clasa)
        #         #print("obj_conf ",object_confidence)
        #         if max_class_prob > conf_thres:
        #             print("conf = ", max_class_prob)
        #             print("class_id ", class_id)
        #             coeffs = mask_coeffs[i]
        #             # print("coeffs ",coeffs)
        #
        #             mask = np.dot(coeffs, mask_protos.reshape(mask_protos.shape[0], -1))
        #             # print("mask1 ",mask)
        #             mask = mask.reshape(mask_protos.shape[1], mask_protos.shape[2])
        #             mask = cv2.resize(mask, (640, 640))
        #             mask = (mask > 0.5).astype(np.uint8)
        #             mask = cv2.resize(mask, (orig_w, orig_h))
        #
        #             x1_int, y1_int, x2_int, y2_int = map(int, [x1 / 640 * orig_w, y1 / 640 * orig_h, x2 / 640 * orig_w,
        #                                                        y2 / 640 * orig_h])
        #             mask_cropped = mask[y1_int:y2_int, x1_int:x2_int]
        #
        #             contours, _ = cv2.findContours(mask_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #
        #             if contours:
        #                 polygon = max(contours, key=cv2.contourArea).squeeze().tolist()
        #                 if isinstance(polygon[0], list):
        #                     polygon = [[p[0] + x1_int, p[1] + y1_int] for p in polygon]
        #                 else:
        #                     polygon = [polygon[0] + x1_int, polygon[1] + y1_int]
        #             else:
        #                 polygon = []
        #
        #             tooth = Teeth(name=class_id, polygon=polygon)
        #             teeth_list.append(tooth)



    def detect_teeths(self, image):

        print("detect_teeths")
        input_tensor, orig_w, orig_h = self.preprocess(image)

        print("input_tensor ",input_tensor)

        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        print("outputs ", outputs)
        teeth_list = self.postprocess(outputs, orig_w, orig_h)
        #print("teeth_list detector ", teeth_list)
        return teeth_list
