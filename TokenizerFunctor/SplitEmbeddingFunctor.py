from TokenizerFunctor.SwiftTokenizerFunctor import SwiftTokenizerFunctor
import TokenizerExtender
import ListLibrary

class SplitEmbeddingFunctor(SwiftTokenizerFunctor):
    def __init__(self, train_col_num):
        super().__init__()
        self.train_col_num = train_col_num
        self.train_token_list = []
        self.think_symbol_token = None

    # 注册tokenizer时添加词表
    def register_tokenizer(self, tokenizer):
        super().register_tokenizer(tokenizer)
        if (self.tokenizer is None):
            raise RuntimeError('SplitEmbeddingFunctor should provide tokenizer')
        # 词表的大小
        vocab_size = tokenizer.vocab_size
        # 根据tokenizer的size直接倒着数作为train token
        for id_token in range(self.train_col_num):
            token_str = f'<think_token{id_token}>'
            temp_token_id = tokenizer.encode(token_str)
            self.train_token_list.append(temp_token_id[0])
        # 记录tokenizer的思考token
        input_ids = self.tokenizer.encode('<think>')
        # 需要确认input id只有一个
        if(len(input_ids) != 1):
            raise RuntimeError(f'input length {input_ids} not 1')
        self.think_symbol_token = input_ids[0]

    # 对需要插入的token list的处理
    # 方便子类函数做一些修改用的
    def insert_think_token(self, ori_list):
        return ListLibrary.insert_list(list1 = ori_list,
                        list2 = self.train_token_list, target_value=self.think_symbol_token,
                        hit_offset = 1)

    # 根据context list和loss scale list生成input id
    def generate_input_id(self, 
        context_list,
        loss_scale_list
    ):
        # 下面这就是原来的input id的原生实现
        """return: input_ids, labels, tokenizer_kwargs"""
        input_ids: List[int] = []
        labels: List[int] = []
        loss_scale: List[float] = []
        tokenizer_kwargs = {}
        if loss_scale_list is None:
            loss_scale_list = [0.] * len(context_list)
        if self.loss_scale.keep_loss_scale:
            ignore_loss_scale = False
        else:
            ignore_loss_scale = all(loss_scale in {0, 1} for loss_scale in loss_scale_list)
        for i, (context, loss_weight) in enumerate(zip(context_list, loss_scale_list)):
            if isinstance(context, str):
                # tokenizer_kwargs is the returned tokenizer_kwargs,
                # while curr_tokenizer_kwargs is the tokenizer_kwargs for the current context.
                token_list = self._tokenize(context)
                # 如果是第2个句子，那就在里面叠加两个token
                if(i == 1):
                    print('训练token替换', __file__)
                    token_list = self.insert_think_token(ori_list=token_list)
                    print(token_list)
            else:
                token_list = context
            input_ids += token_list
            if loss_scale_list[i] > 0.0:
                labels += token_list
            else:
                labels += [-100] * len(token_list)
            if not ignore_loss_scale:
                loss_scale.extend([loss_weight] * len(token_list))
        if ignore_loss_scale:
            loss_scale = None
        return input_ids, labels, loss_scale, tokenizer_kwargs