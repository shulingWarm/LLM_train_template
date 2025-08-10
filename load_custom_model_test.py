from models.qwen3 import register_model
# from models.ThinkModel import register_model
from LLMRunner import LLMRunner

register_model()

# 读取输入文本
# 使用 with 语句确保自动关闭文件，避免资源泄漏
with open('/mnt/data/temp/test_input3.txt', 'r', encoding='utf-8') as f:  # 请替换为你的文件路径
    content = f.read()  # 一次性读取全部内容到字符串
# content = '三国杀移动版中关羽的技能是什么？'

# 调用LLM Runner
runner = LLMRunner('/mnt/data/models/qwen3_save', model_type='qwen3', enable_thinking=True)
#print(runner.inference(content))
print(runner.direct_generate(content))