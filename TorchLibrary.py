
import torch
import torch.nn as nn

# 其中weight的shape是[M,N]
# 构造出来的linear层的shape是 in_dim=N, out_dim=M
def build_linear_from_tensor(weight):
    M, N = weight.shape[0], weight.shape[1]    # 提取输入维度N和输出维度M
    device = weight.device                      # 获取输入权重所在的设备
    
    # 创建Linear层（无偏置），并移到相同设备
    linear = nn.Linear(N, M, bias=False).to(device)
    
    # 在不追踪梯度的情况下复制权重
    with torch.no_grad():
        linear.weight.copy_(weight)   # 直接复制数值，避免影响梯度计算
        
    return linear

# 从linear层里面叠加think line
# ori_linear的输入维度是M, 输出维度是N
# extend_weight的shape是[K,M]
# 需要用输入的extend_weight把linear拼接成 输入维度是M，输出维度是(N+K)的linear层
# 把extend_weight拼在原有的linear层后面
def extend_linear(ori_linear, extend_weight):
    # 验证原始linear层没有偏置
    if ori_linear.bias is not None:
        raise ValueError("原始线性层应无偏置项，但检测到偏置存在")
    
    # 获取原始权重和设备信息
    original_weight = ori_linear.weight  # 原始权重 [N, M]
    device = original_weight.device
    
    # 确保扩展权重在相同设备上
    extend_weight = extend_weight.to(device)
    
    # 检查维度一致性：扩展权重输入维度应匹配原始线性层输入维度
    if extend_weight.shape[1] != ori_linear.in_features:
        raise ValueError(f"维度不匹配：扩展权重输入维度应为{ori_linear.in_features}，但实际为{extend_weight.shape[1]}")
    
    # 垂直拼接权重矩阵 (N+K, M)
    new_weight = torch.cat([original_weight, extend_weight], dim=0)
    
    # 创建新的线性层：输入维度保持不变，输出维度增加K
    new_linear = nn.Linear(ori_linear.in_features, new_weight.shape[0], bias=False).to(device)
    
    # 设置新权重（不追踪梯度）
    with torch.no_grad():
        new_linear.weight.copy_(new_weight)
    
    return new_linear