from LossFunctor.LossFunctorBase import LossFunctorBase
from transformers.loss.loss_utils import fixed_cross_entropy
import TokenizerExtender
import torch
from torch import nn
import torch.nn.functional as F

def preprocess_focus_tokens(focus_token_list, device='cpu'):
    """
    预处理焦点token列表，转换为高效处理的格式
    
    参数:
        focus_token_list (list): 原始需要抑制的token id列表
        device (str/torch.device): 目标设备 (默认: 'cpu')
    
    返回:
        focus_tensor (tensor): 预处理的token索引向量 [num_focus]
        focus_set (set): 焦点token的集合
    """
    if not focus_token_list:
        return torch.empty(0, dtype=torch.long, device=device), set()
    
    # 确保token唯一性
    unique_tokens = set(focus_token_list)
    
    # 创建排序张量优化索引效率
    sorted_tokens = sorted(unique_tokens)
    focus_tensor = torch.tensor(sorted_tokens, dtype=torch.long, device=device)
    
    return focus_tensor, unique_tokens


def sft_loss(source, target, focus_token_tensor, focus_token_set, ignore_id=-100):
    """
    计算SFT模型的特定token抑制损失 (使用预处理的焦点token数据)
    
    参数:
        source (tensor): [M, N] 预测logits
        target (tensor): [M] 真实token id序列
        focus_token_tensor (tensor): 预处理后的焦点token索引 [num_focus]
        focus_token_set (set): 焦点token的集合
        ignore_id (int, optional): 忽略的token id (默认: -100)
    
    返回:
        loss (tensor): 标量损失值
    """
    # 确保焦点token张量在正确设备上
    if focus_token_tensor.device != source.device:
        focus_token_tensor = focus_token_tensor.to(source.device)
    
    # 步骤1: 创建有效位置掩码 (忽略ignore_id位置)
    valid_mask = (target != ignore_id)
    
    # 若没有有效位置或焦点token为空，返回0损失
    if not valid_mask.any() or not focus_token_set:
        return torch.tensor(0.0, device=source.device)
    
    # 步骤2: 提取有效位置
    valid_logits = source[valid_mask]    # [num_valid, N]
    valid_targets = target[valid_mask]    # [num_valid]
    
    # 步骤3: 创建目标位置是焦点token的掩码
    # 使用预处理的集合进行高效成员检查
    is_target_focus = torch.tensor(
        [t.item() in focus_token_set for t in valid_targets], 
        device=source.device, 
        dtype=torch.bool
    )
    
    # 步骤4: 排除目标位置是焦点token的样本
    suppress_positions = ~is_target_focus
    if not suppress_positions.any():
        return torch.tensor(0.0, device=source.device)
        
    suppress_logits = valid_logits[suppress_positions]  # [num_suppress, N]
    
    # 步骤5: 提取焦点token对应的logits
    focus_logits = suppress_logits[:, focus_token_tensor]  # [num_suppress, num_focus]
    
    # 步骤6: 做softmax运算
    exp_values = F.softmax(focus_logits, dim=-1)
    # clamped_logits = focus_logits.clamp(max=20.0)  # 上限20 (exp(20)=4.85e8)
    
    # # 步骤7: 计算指数损失
    # exp_values = torch.exp(clamped_logits)          # [num_suppress, num_focus]
    loss_per_position = torch.sum(exp_values, dim=1)  # [num_suppress]
    loss = torch.mean(loss_per_position)            # 标量
    
    return loss

class TrainEmbeddingFunctor(LossFunctorBase):
    def __init__(self, train_col_num):
        # 记录训练的col num
        self.train_col_num = train_col_num

    # 注册模型的tokenizer，用来把train_col_num置换成focus_token_list
    def register_tokenizer(self, tokenizer):
        # 获取每种可能的think token
        focus_token = []
        for id_think in range(self.train_col_num):
            temp_token = tokenizer.encode(TokenizerExtender.get_think_token(id_think))
            if (len(temp_token) != 1):
                raise RuntimeError(f'Invalid decode result of id_think={id_think}')
            # 将临时的token添加到list里面
            focus_token.append(temp_token[0])
        # 对focus token做预处理，方便每次计算loss的时候使用
        self.focus_token_tensor, self.focus_token_set = preprocess_focus_tokens(focus_token, 'cuda')
        

    def __call__(self, 
        logits,
        labels,
        vocab_size: int,
        num_items_in_batch = None,
        ignore_index: int = -100,
        shift_labels = None,
        **kwargs
    ):
        # 以下是对logits loss的前置处理
        # 参考的这里: /mnt/data/usrApp/anaconda3/lib/python3.12/site-packages/transformers/loss/loss_utils.py
        # def ForCausalLMLoss
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()

        if shift_labels is None:
            # Shift so that tokens < n predict n
            labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        logits = logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(logits.device)
        # 计算对于非think token的抑制loss
        no_think_loss = sft_loss(source=logits, target=shift_labels, 
            focus_token_tensor=self.focus_token_tensor,
            focus_token_set=self.focus_token_set,
            ignore_id = ignore_index
        )
        # 然后再用传统方法计算一下交叉熵损失
        # 调用transformer里面自带的交叉熵
        cross_entropy = fixed_cross_entropy(
            source=logits, target=shift_labels,
            num_items_in_batch = num_items_in_batch,
            ignore_index=ignore_index,
            **kwargs
        )
        # 目前就简单使用两种loss的叠加吧，后面如果有需要再考虑别的
        return no_think_loss + cross_entropy