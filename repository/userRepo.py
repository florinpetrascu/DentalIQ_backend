

from domain.user import User


class UserRepo:

    def __init__(self, session):
        self.session = session


    def find_by_username(self, username):
        return self.session.query(User).filter(User.username == username).first()
