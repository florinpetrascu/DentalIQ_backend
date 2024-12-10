

class Teeth:
    def __init__(self,name,polygon,issues):
        self.name = name
        self.polygon = polygon
        self.issues = issues

    def addIssue(self,issue):
        self.issues.append(issue)

    def __str__(self):
        return f"Teeth(name={self.name}, issues={self.issues})"
    def __repr__(self):
        return f"Teeth(name={self.name}, issues={self.issues})"

