from SplitEmbedding import SplitEmbedding
from SplitLinear import SplitLinear
import TokenizerExtender

# 传入整个LLM模型，然后替换模型里面的embedding
class SplitModelFunctor:
    def __init__(self, train_col_num):
        self.train_col_num = train_col_num
    
    # 传入外部模型，将模型处理后再返回
    def __call__(self, model, tokenizer):
        # 给模型的tokenizer扩展出对应的列数
        TokenizerExtender.add_think_token(model ,tokenizer, self.train_col_num)
        # 取出模型里面的embedding 目前是针对Qwen3的hard coding
        embedding_layer = model.model.embed_tokens
        linear_layer = model.lm_head
        # 调用embedding的拆分逻辑
        model.model.embed_tokens = SplitEmbedding(embedding_layer, self.train_col_num)
        model.model.lm_head = SplitLinear(linear_layer, self.train_col_num)
        return model