from LLMRunner import LLMRunner

# 这是一个训练数据生成器
# 输入一个提示词再输入一个详细提示词，最后把详细提示词的输出和简单提示词的输入拼起来
class DetailPromptProvider:
    def __init__(self, model):
        # 将模型封装成runner
        self.llm_runner = LLMRunner(model)

    # 根据简单提示词和详细提示词得到生成数据
    def getTrainPrompt(self, simple_prompt, detail_prompt):
        # 获得简单提示词下的输出
        outputs = self.llm_runner.batchInference([simple_prompt, detail_prompt])
        # 将简单提示词和复杂提示词的输出拼接起来
        return simple_prompt + outputs[1]
        