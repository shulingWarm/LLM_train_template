import sys
sys.path.insert(0, '/mnt/data/workspace/trainer_template')

from LLMPrinter import LLMPrinter

import TokenizerExtender

# 加载tokenizer
printer = LLMPrinter('/mnt/data/models/Qwen3-8B')

# 从printer里面获取tokenizer
tokenizer = printer.tokenizer

# 扩展tokenizer里面的指针
TokenizerExtender.add_vocab(tokenizer, 'think_token')