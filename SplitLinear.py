import torch
from torch import nn
import torch.nn.functional as F

class SplitLinear(nn.Module):
    def __init__(self, ori_linear: nn.Linear, train_col_num: int):
        """
        封装现有线性层，使输出维度(out_features)的尾部列可训练
        
        Args:
            ori_linear (nn.Linear): 原始线性层
            train_col_num (int): 需设为可训练的列数（从out_features的尾部开始）
        """
        super().__init__()
        
        # 验证线性层是否有偏置
        if ori_linear.bias is not None:
            raise ValueError("SplitLinear 不支持带偏置的线性层")
            
        in_features = ori_linear.in_features
        out_features = ori_linear.out_features
        weight = ori_linear.weight.detach().clone()
        
        # 验证输入列数
        if train_col_num <= 0 or train_col_num > out_features:
            raise ValueError(f"train_col_num 必须在 1 到 {out_features} 之间，但传入了 {train_col_num}")
        
        # 分离固定部分和可训练部分（沿out_features维度）
        self.split_idx = out_features - train_col_num
        
        # 固定部分：前部分行（输出维度中固定的行）
        self.fix_part = weight[:self.split_idx].detach().clone()
        self.fix_part.requires_grad = False
        
        # 可训练部分：尾部行（输出维度中可训练的行）
        self.train_part = nn.Parameter(weight[self.split_idx:].clone())
        
        # 保存维度信息
        self.in_features = in_features
        self.out_features = out_features
        self.train_col_num = train_col_num
        
        # 注册缓冲区确保设备兼容性
        self.register_buffer('fix_buffer', self.fix_part)

    # 存储训练过程中的checkpoint
    def save_checkpoint(self, save_path):
        # 保存可训练的部分
        torch.save(self.train_part.data, save_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：分别计算固定和可训练部分的结果并拼接
        """
        # 计算完整输出
        full_output = x @ self.fix_buffer.T  # 固定部分的计算
        
        # 计算可训练部分的输出
        train_output = x @ self.train_part.T
        
        # 拼接结果
        return torch.cat([full_output, train_output], dim=-1)

    def extra_repr(self) -> str:
        """用于显示模块的额外信息"""
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"trainable_output_cols={self.train_col_num} (rows {self.split_idx} to {self.out_features-1})")