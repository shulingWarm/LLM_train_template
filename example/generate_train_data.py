import sys
sys.path.insert(0, '/mnt/data/workspace/trainer_template')

from DetailPromptProvider import DetailPromptProvider
import json

provider = DetailPromptProvider('/mnt/data/models/Qwen3-8B')

# 完整的训练数据的list
train_list = []

def func(line1, line2):
    # 这里替换为你的处理逻辑（行内容包含末尾换行符）
    print(f"Line1: {line1.strip()}, Line2: {line2.strip()}")

with open('/mnt/data/temp/detail_train_data.txt', 'r') as file:
    while True:
        line1 = file.readline().strip()
        if not line1:  # 文件结束
            break
            
        line2 = file.readline().strip()
        # 处理第二行为空的情况（文件行数为奇数）
        if not line2:
            break
        
        temp_list = provider.getTrainPrompt(
            simple_prompt=line1,
            detail_prompt=line2
        )
        temp_dic = {'messages':temp_list}
        train_list.append(temp_dic)

# 将记录的结果保存成json文件
with open('/mnt/data/temp/dataset_example/train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_list, f, ensure_ascii=False, indent=4)