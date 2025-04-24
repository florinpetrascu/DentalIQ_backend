from domain.issue import Issue

class IssuesRepo:

    def __init__(self, session):
        self.session = session

    # CRUD pentru Issues
    def create_issues(self, name, teeth_id):
        new_issue = Issue(None, name, teeth_id)
        self.session.add(new_issue)
        self.session.commit()

    def get_issues_by_id(self, issue_id):
        return self.session.query(Issue).filter(Issue.id == issue_id).first()

    def update_issues(self, issue_id, name, teeth_id):
        issue = self.get_issues_by_id(issue_id)
        if issue:
            issue.name = name
            issue.teeth_id = teeth_id
            self.session.commit()

    def delete_issues(self, issue_id):
        issue = self.get_issues_by_id(issue_id)
        if issue:
            self.session.delete(issue)
            self.session.commit()
