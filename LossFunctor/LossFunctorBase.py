

class LossFunctorBase:
    def __init__(self):
        pass

    def register_tokenizer(self, tokenizer):
        pass

    def __call__(self, 
        logits,
        labels,
        vocab_size: int,
        num_items_in_batch = None,
        ignore_index: int = -100,
        shift_labels = None,
        **kwargs
    ):
        pass