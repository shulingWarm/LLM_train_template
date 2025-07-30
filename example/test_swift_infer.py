import sys
sys.path.insert(0, '/mnt/data/workspace/trainer_template')

import swift_infer

# 模型路径 
model_path = '/mnt/data/models/Qwen3-8B'
lora_modules='/mnt/data/temp/train_lora_0730/v1-20250730-080119/checkpoint-20'

swift_infer.inference_online(model_path=model_path, lora_modules=[lora_modules])
