





def get_patient_dto(patient):
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
                        "text": note.text
                    }
                    for note in tooth.notes
                ],
                "polygon": tooth.polygon

            }
            for tooth in patient.teeths
        ],
    }
    return patient_dto