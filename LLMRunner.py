import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from typing import Union, List, Optional

class StopOnTokens(StoppingCriteria):
    """自定义停止条件，遇到特定标记时停止生成（支持batch处理）"""
    def __init__(self, tokenizer, stop_list):
        # 打印一下tokenzer里面词表的最大值
        print('tokenizer.vocab_size',tokenizer.vocab_size)
        # 过滤空标记并确保停止标记有效
        self.stop_token_ids = []
        for x in stop_list:
            token_ids = tokenizer.encode(x, add_special_tokens=False)
            if token_ids:  # 跳过空列表
                self.stop_token_ids.append(token_ids)
                # # 检查标记是否在词汇表范围内
                # if hasattr(tokenizer, "vocab_size") and any(id_ >= tokenizer.vocab_size for id_ in token_ids):
                #     raise RuntimeError(f"警告: 停止词 '{x}' 包含超出词汇表范围的标记: {token_ids}")
                # else:
                #     self.stop_token_ids.append(token_ids)
        
        # 预转换停止标记为设备无关的张量
        self.stop_tensors = [torch.tensor(ids, dtype=torch.long) for ids in self.stop_token_ids]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 遍历所有预定义的停止标记序列
        for stop_tensor in self.stop_tensors:
            if stop_tensor.numel() == 0:  # 跳过空张量
                continue
                
            stop_len = len(stop_tensor)
            # 检查batch中每个样本
            for i in range(input_ids.size(0)):
                if input_ids.size(1) < stop_len:
                    continue
                    
                # 获取当前样本的最后N个token
                sequence_end = input_ids[i, -stop_len:].to(stop_tensor.device)
                
                # 比较是否匹配停止标记
                if torch.all(sequence_end == stop_tensor.to(sequence_end.device)):
                    return True
        return False

class LLMRunner:
    def __init__(self, 
                 model: Union[str, AutoModelForCausalLM],
                 model_type: str = "qwen",
                 max_new_token: int = 512,
                 device: str = "auto",
                 enable_thinking: bool = False,
                 tokenizer = None):
        """
        初始化Qwen模型推理引擎
        
        Args:
            model: 模型路径或已加载的模型实例
            model_type: 模型类型 (qwen2.5/qwen3)
            max_new_token: 最大生成token数
            device: 运行设备 (auto/cpu/cuda)
            enable_thinking: 是否启用思考模式(Qwen3专用)
        """
        # 设备配置
        self.device = device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载模型和分词器
        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto" if self.device == "cuda" else None
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            # 设置停止条件
            stop_list = ['\nHuman:', '\n```\n'] if "instruct" in model else ['<|endoftext|>']
        else:
            if (tokenizer is None):
                raise RuntimeError('tokenizer cannot be None when model is not path')
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
            # 设置停止条件
            stop_list = ['<|endoftext|>']
        
        self.model.eval()
        self.max_new_token = max_new_token
        self.model_type = model_type
        self.enable_thinking = enable_thinking
        if(hasattr(self.model, 'register_tokenizer')):
            self.model.register_tokenizer(self.tokenizer)
        
        self.stopping_criteria = StoppingCriteriaList([
            StopOnTokens(self.tokenizer, stop_list)
        ])

    def setMaxNewToken(self, max_new_token: int):
        """设置最大生成token数"""
        self.max_new_token = max_new_token

    def _prepare_inputs(self, prompt: str) -> torch.Tensor:
        """预处理输入并转换为模型需要的格式"""
        # Qwen3特殊处理
        if "qwen3" in self.model_type and self.enable_thinking:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            return self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # 标准处理
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def _process_output(self, output_ids: torch.Tensor) -> str:
        """处理模型输出并解码为文本"""
        # Qwen3思考模式特殊解析
        if "qwen3" in self.model_type and self.enable_thinking:
            output_ids = output_ids[0].tolist()
            try:
                # index = len(output_ids) - output_ids[::-1].index(151668)  # 151668是Qwen3思考标记
                # thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True)
                # content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True)
                content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                return content.split('assistant', 1)[-1].lstrip()
            except ValueError:
                pass
        
        # 标准解码
        return self.tokenizer.decode(
            output_ids[0], 
            skip_special_tokens=True
        ).strip()

    def direct_generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_token,
                stopping_criteria=self.stopping_criteria,
                temperature=0.1,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self._process_output(outputs)

    def inference(self, prompt: str) -> str:
        """单次推理接口"""
        inputs = self._prepare_inputs(prompt)
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_token,
                stopping_criteria=self.stopping_criteria,
                temperature=0.1,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self._process_output(outputs)

    def batchInference(self, prompt_list: List[str]) -> List[str]:
        """批量推理接口"""
        # 批量编码
        inputs = self.tokenizer(
            prompt_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=1024
        ).to(self.device)
        
        input_lengths = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_token,
                stopping_criteria=self.stopping_criteria,
                temperature=0.1,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 分别解码每个结果
        results = []
        for i in range(len(prompt_list)):
            output_ids = outputs[i][input_lengths:]
            results.append(
                self._process_output(output_ids)
                # self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            )
        
        return results

if __name__ == '__main__':
    # 初始化一个runner
    runner = LLMRunner('/mnt/data/models/Qwen3-8B')
    # 测试推理模型
    print(runner.inference('从公元100年开始，说出每个公元整百年对应的中国朝代。'))