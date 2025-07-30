

# 这东西是用于在这个地方被调用的
# /mnt/data/usrApp/anaconda3/lib/python3.12/site-packages/swift/llm/template/base.py
# token_list = self._tokenize(context)
# 替换它所在的for循环的整体位置
class SwiftTokenizerFunctor:
    def __init__(self):
        # 将tokenizer初始化为None
        self.tokenizer = None

    # 注册用于生成input id的tokenizer
    def register_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    # 获取词表大小
    def get_tokenizer_size(self):
        return self.tokenizer.vocab_size

    # 原生的tokenize的函数
    def _tokenize(self, context, **tokenizer_kwargs):
        # 如果tokenizer还没有被初始化过，需要报个错
        if (self.tokenizer is None):
            raise RuntimeError('class SwiftTokenzerFunctor should be initialzed firstly.')
        return self.tokenizer(
            context, return_attention_mask=False, add_special_tokens=False, **tokenizer_kwargs)['input_ids']

    # 注册 loss scale的信息
    def register_loss_scale(self, loss_scale):
        # 记录loss scale
        self.loss_scale = loss_scale

    # 根据context list和loss scale list生成input id
    def generate_input_id(self, 
        context_list,
        loss_scale_list
    ):
        raise NotImplementedError()