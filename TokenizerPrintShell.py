import traceback

# tokenizer打印相关的shell
# 主要是为了看一下tokenizer是什么时候调用的
class TokenizerPrintShell:
    def __init__(self):
        pass

    # 初始化tokenizer shell
    def init_shell(self, tokenizer):
        self.tokenizer = tokenizer
        return self

    # 承接tokenizer原本的处理函数
    def __call__(self, prompt):
        # 打印调用栈
        traceback.print_stack()
        # 打印调用内容
        print(prompt)
        exit()
        return self.tokenizer(prompt)
