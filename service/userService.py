


from repository.userRepo import UserRepo


class UserService:

    def __init__(self, userRepo: UserRepo):

        self.userRepo = userRepo

    def get_by_username(self, username):
        return self.userRepo.find_by_username(username)