import torch
import torch.nn as nn

class ThinkEmbedding:
    # 传入原始的embedding
    def __init__(self, ori_embedding):
        # 记录原始的embedding
        self.ori_embedding = ori_embedding
        # 初始化think embedding，初始化为none
        self.think_embedding = None

    def forward(self, ids):
        """
        ids: 输入ID张量，值范围在[0, total_vocab-1]
        """
        # 如果think embedding不为none，就直接正常返回
        if (self.think_embedding is None):
            return self.ori_embedding(ids)
        # 创建布尔掩码判断ID属于哪个嵌入层
        mask_B = ids >= self.ori_embedding.num_embeddings
        ids_A = ids.clone()
        ids_B = ids.clone() - self.ori_embedding.num_embeddings
        
        # 获取有效索引 (避免无效索引导致的报错)
        valid_mask_B = mask_B & (ids_B < self.think_embedding.num_embeddings)
        valid_mask_A = ~mask_B & (ids_A < self.ori_embedding.num_embeddings)
        
        # 初始化输出张量
        output = torch.zeros(
            *ids.shape, 
            self.embedding_dim,
            dtype=self.ori_embedding.weight.dtype,
            device=self.ori_embedding.weight.device
        )
        
        # 从A层获取嵌入（ID < 10）
        if valid_mask_A.any():
            output[valid_mask_A] = self.ori_embedding(ids_A[valid_mask_A])
        
        # 从B层获取嵌入（10 <= ID < 20）
        if valid_mask_B.any():
            output[valid_mask_B] = self.think_embedding(ids_B[valid_mask_B])
        
        return output

class CombinedEmbedding(nn.Module):
    def __init__(self, ori_embedding, think_embedding):
        super().__init__()
        self.ori_embedding = ori_embedding
        self.think_embedding = think_embedding
        
        # 验证嵌入维度是否相同
        if ori_embedding.embedding_dim != think_embedding.embedding_dim:
            raise ValueError("两个嵌入层的维度必须相同")
            
        self.embedding_dim = ori_embedding.embedding_dim
        self.total_vocab = ori_embedding.num_embeddings + think_embedding.num_embeddings

    def forward(self, ids):
        """
        ids: 输入ID张量，值范围在[0, total_vocab-1]
        """
        # 创建布尔掩码判断ID属于哪个嵌入层
        mask_B = ids >= self.ori_embedding.num_embeddings
        ids_A = ids.clone()
        ids_B = ids.clone() - self.ori_embedding.num_embeddings
        
        # 获取有效索引 (避免无效索引导致的报错)
        valid_mask_B = mask_B & (ids_B < self.think_embedding.num_embeddings)
        valid_mask_A = ~mask_B & (ids_A < self.ori_embedding.num_embeddings)
        
        # 初始化输出张量
        output = torch.zeros(
            *ids.shape, 
            self.embedding_dim,
            dtype=self.ori_embedding.weight.dtype,
            device=self.ori_embedding.weight.device
        )
        
        # 从A层获取嵌入（ID < 10）
        if valid_mask_A.any():
            output[valid_mask_A] = self.ori_embedding(ids_A[valid_mask_A])
        
        # 从B层获取嵌入（10 <= ID < 20）
        if valid_mask_B.any():
            output[valid_mask_B] = self.think_embedding(ids_B[valid_mask_B])
        
        return output