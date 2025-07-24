import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

class LLMPrinter:
    def __init__(self, model_path):
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        ).to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 打印tokenizer的数据类型
    def printTokenizerType(self):
        print(type(self.tokenizer))

    # 打印tokenizer里面的词表
    def printTokenizerVocab(self):
        print(self.tokenizer.get_vocab())

    # 打印model的数据类型
    def printModelType(self):
        print(type(self.model))

    # 打印模型结构
    def printArchitecture(self):
        print(self.model)