
from transformers import PreTrainedTokenizer
from typing import Union, List

def add_vocab(model, tokenizer: PreTrainedTokenizer, 
              new_tokens: Union[str, List[str]]) -> dict:
    """
    安全添加新词汇到tokenizer而不允许ID指定
    
    参数:
    tokenizer: Transformers库的分词器对象
    new_tokens: 要添加的token字符串或字符串列表
    
    返回:
    包含成功添加的新token及其自动分配ID的字典
    """
    # 统一输入格式为列表
    tokens_to_add = [new_tokens] if isinstance(new_tokens, str) else new_tokens
    
    if not tokens_to_add:
        print("⚠️ 输入token列表为空")
        return {}
    
    # 过滤已存在token
    existing_vocab = tokenizer.get_vocab()
    unique_new_tokens = [token for token in tokens_to_add 
                         if token not in existing_vocab]
    
    # 添加唯一的新token
    added_count = tokenizer.add_tokens(unique_new_tokens)
    
    # 准备结果
    results = {}
    for token in tokens_to_add:
        token_id = tokenizer.convert_tokens_to_ids(token)
        results[token] = token_id
        
        # 输出状态信息
        if token in unique_new_tokens:
            print(f"✅ 添加新token: '{token}' → ID {token_id}")
        else:
            print(f"⏩ 跳过已存在token: '{token}' → ID {token_id}")
    
    # 关键提醒
    print(f"\n✨ 添加完成！共尝试添加 {len(tokens_to_add)} 个token，"
          f"实际添加 {added_count} 个新token")
    
    # 给模型调整embedding的大小
    if (model is not None):
        model.resize_token_embeddings(len(tokenizer))
    return results

# 给模型添加think token
def add_think_token(model, tokenizer, think_token_num):
    # 添加think token
    think_tokens = [f'<think_token{i}>' for i in range(think_token_num)]
    # 获取添加过的token
    token_ids = add_vocab(model, tokenizer, think_tokens)
    return token_ids