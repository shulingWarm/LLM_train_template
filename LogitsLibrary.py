import torch


# logits的shape是[1, token_num, vocab_size]
# 期望两个tensor shape是 logit: [1, token_num, k+1], id: [1, token_num, k+1] dtype=int
# 取vocab_size这个维度里面概率最高的k个概率
# focus_vocab: [1, token_num] dtype=int 每个token位置都有一个特别关注的单词，需要将这个特别关注的单词放在输出logits和id的最后
def topk(logits: torch.Tensor, k: int, focus_vocab: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    提取top-k概率值及其索引，并在末尾添加特别关注的词汇
    
    Args:
        logits: 输入张量，shape=[1, token_num, vocab_size]
        k: 要提取的最高概率项的数量
        focus_vocab: 特别关注的词汇索引，shape=[1, token_num]
    
    Returns:
        topk_values: 最高的k个概率值 + 关注词概率，shape=[1, token_num, k+1]
        topk_indices: 对应的词汇表索引 + 关注词索引，shape=[1, token_num, k+1]
    """
    # 计算softmax得到概率分布
    probs = torch.softmax(logits, dim=-1)
    
    # 1. 获取原始的top-k结果
    topk_values, topk_indices = torch.topk(probs, k, dim=-1)
    
    # 2. 添加特别关注的词汇
    batch_size, token_num, _ = probs.shape
    
    # 获取每个token位置关注词的概率值
    focus_probs = torch.gather(probs, dim=-1, 
                              index=focus_vocab.unsqueeze(-1))
    
    # 扩展关注词索引的维度以匹配输出形状
    focus_indices = focus_vocab.unsqueeze(-1)
    
    # 3. 组合原始top-k和关注词
    combined_values = torch.cat([topk_values, focus_probs], dim=-1)
    combined_indices = torch.cat([topk_indices, focus_indices], dim=-1)
    
    return combined_values, combined_indices

# 打印topk的内容
# 将id置换成字符串，打印每个字符串的概率
# 打印的格式是
# str00:prob00, str01:prob01, str02:prob02, ...
# str10:prob10, str11:prob11, str12:prob12, ...
# ...
def print_topk(topk_prob: torch.Tensor, topk_ids: torch.Tensor, tokenizer):
    """
    格式化打印topk结果
    
    Args:
        topk_prob: 概率值张量，shape=[1, token_num, k+1]
        topk_ids: 索引张量，shape=[1, token_num, k+1]
        tokenizer: 文本分词器，提供id到字符串的转换
    """
    # 移除非必要的批处理维度
    probs = topk_prob[0]  # shape: [token_num, k+1]
    ids = topk_ids[0]     # shape: [token_num, k+1]
    
    # 遍历每个token位置
    for i in range(probs.size(0)):
        # 收集当前token的所有候选词
        candidate_strs = []
        
        # 遍历当前token的所有候选
        for j in range(probs.size(1)):
            token_id = ids[i, j].item()
            
            # 将ID转换为可读字符串
            try:
                token_str = tokenizer.decode([token_id])
                
                # 清理特殊符号和空白
                token_str = token_str.replace("Ġ", " ")  # 处理HuggingFace空格标记
                token_str = token_str.replace("</w>", "")  # 处理SentencePiece词尾
                token_str = token_str.strip()
            except:
                token_str = f"[ID:{token_id}]"
            
            # 格式化为"字符串:概率"对
            prob_val = probs[i, j].item()
            candidate_strs.append(f"{token_str}:{prob_val:.4f}")
        
        # 连接并打印当前token的所有候选
        print(",\t".join(candidate_strs))