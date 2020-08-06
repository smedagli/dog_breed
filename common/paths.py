import os

def listdir(folder: str) -> list: return [os.path.join(folder, file) for file in os.listdir(folder)]


class Folders():
    def __init__(self):
        self.data = 'data/'
        self.models = 'saved_models/'
