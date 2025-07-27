from TokenizerFunctor.SwiftTokenizerFunctor import SwiftTokenizerFunctor

class SplitEmbeddingFunctor:
    def __init__(self, train_id_list):
        super().__init__()
        self.train_id_list = train_id_list

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
        # 打印context 和 loss_weight的内容，用于查看后续操作
        
        for i, (context, loss_weight) in enumerate(zip(context_list, loss_scale_list)):
            if isinstance(context, str):
                # tokenizer_kwargs is the returned tokenizer_kwargs,
                # while curr_tokenizer_kwargs is the tokenizer_kwargs for the current context.
                token_list = self._tokenize(context)
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