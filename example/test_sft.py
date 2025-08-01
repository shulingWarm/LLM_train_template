import sys
sys.path.insert(0, '/mnt/data/workspace/trainer_template')

import torch
import swift_sft
from TokenizerPrintShell import TokenizerPrintShell

# lora训练的模式
def lora_train():
    swift_sft.launch_swift_sft(
        model_path = '/mnt/data/models/Qwen3-8B',
        train_type = 'lora',
        dataset_path = '/mnt/data/temp/dataset_example/train_data.json',
        num_train_epochs = 20,
        output_dir = '/mnt/data/temp/train_lora_0730',
        torch_dtype = torch.bfloat16,
        tokenizer_shell = None,
        train_col_num = 20,
        learning_rate = 1e-4
    )

# 测试启动sft的训练
swift_sft.launch_swift_sft(
    # model_path = '/mnt/data/models/Qwen3-8B',
    model_path = '/mnt/data/models/qwen3_save',
    train_type = 'part_embedding',
    dataset_path = '/mnt/data/temp/dataset_example/train_data_only.json',
    num_train_epochs = 100,
    output_dir = '/mnt/data/temp/train_output',
    torch_dtype = torch.bfloat16,
    tokenizer_shell = None,
    train_col_num = 20,
    learning_rate = 1e-1
)

# lora训练的版本
# lora_train()