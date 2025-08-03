from SplitEmbedding import SplitEmbedding
from SplitLinear import SplitLinear
import TokenizerExtender

# 取出模型里面的 embedding
def get_model_embedding(model):
    return model.model.embed_tokens

def set_model_embedding(model, new_embedding):
    model.model.embed_tokens = new_embedding

# 获得模型的输出linear层
def get_model_out_head(model):
    return model.lm_head

def set_model_out_head(model, new_out_head):
    model.lm_head = new_out_head

# 传入整个LLM模型，然后替换模型里面的embedding
class SplitModelFunctor:
    def __init__(self, train_col_num):
        self.train_col_num = train_col_num
    
    # 传入外部模型，将模型处理后再返回
    def __call__(self, model, tokenizer):
        # 取出模型里面的embedding 目前是针对Qwen3的hard coding
        embedding_layer = model.model.embed_tokens
        linear_layer = model.lm_head
        # 调用embedding的拆分逻辑
        set_model_embedding(model, SplitEmbedding(embedding_layer, self.train_col_num))
        set_model_out_head(model, SplitLinear(linear_layer, self.train_col_num))
        return model