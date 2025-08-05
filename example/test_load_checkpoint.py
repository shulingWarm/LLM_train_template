import sys
sys.path.insert(0, '/mnt/data/workspace/trainer_template')

import load_embedding_inference

# checkpoint 的路径 
# checkpoint_dir = '/mnt/data/temp/train_output/v0-20250803-213507'
checkpoint_dir = '/mnt/data/temp/train_output/v1-20250803-235012'
# 模型的路径
model_path = '/mnt/data/models/qwen3_save'

# 调用加载embedding并推理的函数
load_embedding_inference.load_embedding_inference(model_path,
    checkpoint_dir)