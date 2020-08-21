import os

def listdir(folder: str) -> list: return [os.path.join(folder, file) for file in os.listdir(folder)]

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

class Folders():
    def __init__(self):
        self.data = os.path.join(module_path, 'data')
        self.models = os.path.join(module_path, 'saved_models')
        self.training_data = os.path.join(self.data, 'dogImages', 'train')
        self.test_data = os.path.join(self.data, 'dogImages', 'test')
        self.validation_data = os.path.join(self.data, 'dogImages', 'valid')