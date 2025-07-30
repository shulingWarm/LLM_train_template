import torch
from torch import nn

class SplitEmbedding(nn.Module):
    def __init__(self, ori_embedding_layer, train_embedding_num):
        """
        部分参数可训练的embedding层
        
        Args:
            ori_embedding_layer (nn.Embedding): 原始Embedding层对象
            train_embedding_num (int): 需要训练的token数量（从末尾开始计数）
        """
        super().__init__()
        # 提取Embedding层的权重和维度
        ori_embedding = ori_embedding_layer.weight.data
        vocab_size, embed_dim = ori_embedding.shape
        
        # 参数检查
        assert train_embedding_num <= vocab_size, \
            f"train_embedding_num({train_embedding_num}) 必须小于等于总词表大小({vocab_size})"
        
        # 切分嵌入矩阵
        self.fixed_embedding_num = vocab_size - train_embedding_num
        self.train_embedding_num = train_embedding_num
        self.embedding_dim = embed_dim
        self.padding_idx = getattr(ori_embedding_layer, 'padding_idx', None)
        
        # 固定部分（不可训练）
        fixed_embeddings = ori_embedding[:self.fixed_embedding_num].detach().clone()
        self.register_buffer('fixed_embedding', fixed_embeddings)
        
        # 可训练部分
        train_embeddings = ori_embedding[-train_embedding_num:].detach().clone()
        self.train_embedding = nn.Parameter(train_embeddings)
    
    def forward(self, input_ids):
        """
        Args:
            input_ids (Tensor): 输入token ID，任意形状
        Returns:
            embeddings (Tensor): 对应embedding向量
        """
        # 1. 计算固定部分索引和可训练部分索引
        fixed_mask = input_ids < self.fixed_embedding_num
        train_mask = ~fixed_mask
        
        # 2. 初始化输出张量
        output = torch.empty(
            *input_ids.shape, self.embedding_dim, 
            dtype=self.fixed_embedding.dtype,
            device=input_ids.device
        )
        
        # 3. 处理固定部分
        if torch.any(fixed_mask):
            fixed_ids = input_ids[fixed_mask]
            # 确保索引安全 (0 到 fixed_embedding_num-1)
            fixed_ids = fixed_ids.clamp(min=0, max=self.fixed_embedding_num-1)
            output[fixed_mask] = self.fixed_embedding[fixed_ids]
        
        # 4. 处理可训练部分
        if torch.any(train_mask):
            # 计算在可训练矩阵中的局部索引
            train_ids = input_ids[train_mask] - self.fixed_embedding_num
            # 确保索引在合法范围 (0 ~ train_embedding_num-1)
            train_ids = train_ids.clamp(min=0, max=self.train_embedding_num-1)
            output[train_mask] = self.train_embedding[train_ids]
        
        # 处理padding索引（如果存在）
        if self.padding_idx is not None:
            padding_mask = input_ids == self.padding_idx
            if padding_mask.any():
                output[padding_mask] = 0.0
                
        return output

    def extra_repr(self):
        return f"总词表大小: {self.fixed_embedding_num + self.train_embedding_num}, " \
               f"固定参数: {self.fixed_embedding_num}, " \
               f"可训练参数: {self.train_embedding_num}, " \
               f"嵌入维度: {self.embedding_dim}" + \
               (f", padding_idx: {self.padding_idx}" if self.padding_idx is not None else "")