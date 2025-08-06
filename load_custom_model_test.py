from models.qwen3 import register_model
from LLMRunner import LLMRunner

register_model()
# 调用LLM Runner
runner = LLMRunner('/mnt/data/models/qwen3_save')
print(runner.inference('三国杀移动版中刘备的技能是什么？'))