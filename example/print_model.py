import sys
sys.path.insert(0, '/mnt/data/workspace/trainer_template')

from LLMPrinter import LLMPrinter

# 加载模型
printer = LLMPrinter('/mnt/data/models/Qwen3-8B')
# 打印tokenizer的类型
printer.printTokenizerType()

# 打印模型的类型
printer.printModelType()

# 修改token里面模型的数据结构
printer.model.resize_token_embeddings(151937)

# 打印模型结构
printer.printArchitecture()