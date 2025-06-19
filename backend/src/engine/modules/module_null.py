import torch

class ModuleNull:
    def __init__(self):
        self.name = "Null"

    def run(self, engine, task):
        return "RESULT_SUCCESS"

    def run_with_tensor(self, engine, task, tensor):
        return "RESULT_SUCCESS"
