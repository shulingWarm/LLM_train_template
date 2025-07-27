import sys
sys.path.insert(0, '/mnt/data/workspace/trainer_template')

import torch
import swift_sft
from TokenizerPrintShell import TokenizerPrintShell

tokenizer_shell = TokenizerPrintShell()

# 测试启动sft的训练
swift_sft.launch_swift_sft(
    model_path = '/mnt/data/models/Qwen3-8B',
    train_type = 'part_embedding',
    dataset_path = '/mnt/data/temp/dataset_example/train_data.json',
    num_train_epochs = 20,
    output_dir = '/mnt/data/temp/train_output',
    torch_dtype = torch.bfloat16,
    tokenizer_shell = tokenizer_shell
)