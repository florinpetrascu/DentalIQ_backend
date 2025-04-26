



import json
import re
def get_patient_dto_for_firebase(patient):
    patient_dto = {
        "id": patient.id,
        "firstName": patient.firstName,
        "lastName": patient.lastName,
        "phoneNumber": patient.phoneNumber,
        "image": patient.image,
        "teeths": [
            {
                "id": tooth.id,
                "name": tooth.name,
                "issues": [
                    {
                        "id": issue.id,
                        "name": issue.name,

                    }
                    for issue in tooth.issues
                ],
                "notes": [
                    {
                        "id": note.id,
                        "date": note.date,
                        "text": note.text
                    }
                    for note in tooth.notes
                ],
                "polygon": json.dumps(tooth.polygon)  # aici e cheia

            }
            for tooth in patient.teeth
        ],
    }
    return patient_dto


def get_patient_dto(patient):
    patient_dto = {
        "id": patient.id,
        "first_name": patient.first_name,
        "last_name": patient.last_name,
        "phone_number": patient.phone_number,
        "teeths": [
            {
                "id": tooth.id,
                "name": tooth.name,
                "issues": [
                    {
                        "id": issue.id,
                        "name": issue.name,

                    }
                    for issue in tooth.issues
                ],
                # "notes": [
                #     {
                #         "id": note.id,
                #         "date": note.date,
                #         "text": note.text
                #     }
                #     for note in tooth.notes
                # ],
                "polygon": json.dumps([list(map(int, pair)) for pair in re.findall(r'\{(\d+),(\d+)\}', tooth.polygon)] ) # aici e cheia

            }
            for tooth in patient.teeths
        ],
    }
    return patient_dto

def get_simple_patient_dto(patient):
    patient_dto = {
        "id": patient.id,
        "first_name": patient.first_name,
        "last_name": patient.last_name,
        "phone_number": patient.phone_number,
    }
    return patient_dto
