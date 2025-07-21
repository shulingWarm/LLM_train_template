import sys
sys.path.insert(0, '/mnt/data/workspace/trainer_template')

from LLMRunner import LLMRunner

runner = LLMRunner('/mnt/data/models/Qwen3-8B',
    model_type='qwen3',
    max_new_token=1024,
    enable_thinking=True
)

while(True):
    prompt = input('输入:')
    print(runner.inference(prompt))