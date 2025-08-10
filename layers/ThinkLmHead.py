import torch
import TorchLibrary

# 传入原始的lm head
# 把think lm head弄成空的
class ThinkLmHead(torch.nn.Module):
    def __init__(self, ori_lm_head):
        super().__init__()
        # 需要确保lm_head是linear
        if not isinstance(ori_lm_head, torch.nn.Linear):
            raise RuntimeError(f'ori_lm_head {type(ori_lm_head)} not Linear')
        # 记录原始的linear
        self.ori_lm_head = ori_lm_head
        # 初始化 think linear
        self.think_linear = None

    # 添加新的think行
    def add_think_line(self, think_line):
        # 判断think linear是否已经初始化过
        if(self.think_linear is None):
            # 从think line里面新建linear层
            self.think_linear = TorchLibrary.build_linear_from_tensor(think_line)
        else:
            # 其他情况下需要重构linear层
            self.think_linear = TorchLibrary.extend_linear(self.think_linear, think_line)

    def forward(self, x):
        # 原始的输出数据
        ori_output = self.ori_lm_head(x)
        # 本地的linear输出
        if (self.think_linear is not None):
            think_output = self.think_linear(x)
            # 把两个输出数据拼起来
            ori_output = torch.cat([ori_output, think_output], dim=-1)
        # 返回linear的结果
        return ori_output