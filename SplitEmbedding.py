import torch
from torch import nn
import torch.nn.functional as F

class SplitEmbedding(nn.Module):
    def __init__(self, ori_embedding, train_col_num: int):
        """
        封装现有嵌入层，仅使其部分列可训练
        
        Args:
            ori_embedding (nn.Embedding): 原始嵌入层
            train_col_num (int): 需设为可训练的列数（从尾部开始）
        """
        super().__init__()
        num_embeddings, embedding_dim = ori_embedding.weight.shape
        
        # 验证输入列数
        if train_col_num <= 0 or train_col_num > embedding_dim:
            raise ValueError(f"train_col_num 必须在 1 到 {embedding_dim} 之间，但传入了 {train_col_num}")
        
        # 从原始嵌入层获取权重
        full_weight = ori_embedding.weight.detach().clone()
        
        # 分离固定部分和可训练部分
        self.split_idx = embedding_dim - train_col_num
        
        # 固定部分：移除尾部可训练列
        self.freeze_part = full_weight[:, :self.split_idx].detach().clone()
        self.freeze_part.requires_grad = False  # 确保无梯度计算
        
        # 可训练部分：仅尾部列
        self.training_part = nn.Parameter(full_weight[:, self.split_idx:].clone())
        
        # 词表大小和维度信息
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # 注册缓冲区以确保兼容设备转移
        self.register_buffer('freeze_buffer', self.freeze_part)

    def forward(self, input_ids):
        """
        前向传播：拼接固定部分和可训练部分
        """
        # 在设备上获取固定部分（兼容CPU/GPU切换）
        freeze_part = self.freeze_buffer
        
        # 高效拼接权重矩阵：固定部分 + 可训练部分
        full_weight = torch.cat([freeze_part, self.training_part], dim=1)
        
        # 执行嵌入查找
        return F.embedding(input_ids, full_weight)

