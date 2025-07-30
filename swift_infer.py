from swift.llm.infer import SwiftInfer
from swift.llm.argument import InferArguments


# 执行模型的online推理过程
def inference_online(
    model_path,
    infer_backend='pt',
    stream=True,
    max_new_tokens=2048,
    lora_modules=None
):
    # 新建推理用的args
    infer_args = InferArguments(
        model=model_path,
        infer_backend=infer_backend,
        stream=stream,
        max_new_tokens=max_new_tokens,
        lora_modules=lora_modules
    )
    # 通过args调用在线推理
    infer_tool = SwiftInfer(infer_args)
    # 调用主函数
    infer_tool.main()