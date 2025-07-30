import sys
sys.path.insert(0, '/mnt/data/workspace/trainer_template')

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
import TokenizerExtender

# 模型加载后直接保存的示例

model_path = '/mnt/data/models/Qwen3-8B'
output_path = '/mnt/data/models/qwen3_save'
# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 扩展模型的tokenizer
TokenizerExtender.add_think_token(model=model, tokenizer=tokenizer, think_token_num=20)

model.save_pretrained(output_path, torch_dtype=torch.float16, device_map='cuda')
tokenizer.save_pretrained(output_path)