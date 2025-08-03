

# 保存checkpoint时用到的回调函数
class CheckpointSaveFunctor:
    def __init__(self):
        self.model = None

    def register_model(self, model):
        self.model = model

    def __call__(self, output_dir):
        pass