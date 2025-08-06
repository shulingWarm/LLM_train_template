from CheckpointSaveFunctor.CheckpointSaveFunctor import CheckpointSaveFunctor

import SplitModelFunctor
from SplitEmbedding import SplitEmbedding
from SplitLinear import SplitLinear
import os
import torch
import traceback

# 加载checkpoint
def load_embedding_checkpoint(checkpoint_dir):
    # 两个路径分别加载embedding和输出linear
    embedding = torch.load(os.path.join(checkpoint_dir, 'embedding.pt'))
    out_head = torch.load(os.path.join(checkpoint_dir, 'out_head.pt'))
    return embedding, out_head

class SaveSplitEmbedding(CheckpointSaveFunctor):
    
    def __call__(self, output_dir):
        # 模型必须是有效模型
        if (self.model is None):
            raise RuntimeError('Model is None')
        # 取出模型里面的embedding层
        embedding = SplitModelFunctor.get_model_embedding(self.model)
        out_head = SplitModelFunctor.get_model_out_head(self.model)

        # 需要确认类型符合
        if not isinstance(embedding, SplitEmbedding):
            raise RuntimeError(f'Error type {type(embedding)} not SplitEmbedding')
        if not isinstance(out_head, SplitLinear):
            raise RuntimeError(f'Error type {type(out_head)} not SplitLienar')
        # 存储embedding
        embedding.save_checkpoint(os.path.join(output_dir, 'embedding.pt'))
        # 存储 linear
        out_head.save_checkpoint(os.path.join(output_dir, 'out_head.pt'))