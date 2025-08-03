from TokenizerFunctor.SplitEmbeddingFunctor import SplitEmbeddingFunctor
import ListLibrary

class SeparateInsertThinkToken(SplitEmbeddingFunctor):
    def __init__(self, train_col_num):
        super().__init__(train_col_num)

    # 对需要插入的token list的处理
    # 方便子类函数做一些修改用的
    def insert_think_token(self, ori_list):
        # 在list里面查找某个id第一次出现的位置
        target_id = ListLibrary.find_in_list(ori_list, self.think_symbol_token)
        if(target_id >= 0):
            return ListLibrary.separate_insert_token(list1=ori_list,
                list2 = self.train_token_list, begin_offset=target_id+1)
        return ori_list