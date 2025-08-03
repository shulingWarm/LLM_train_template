import torch
import CheckpointSaveFunctor.SaveSplitEmbedding as SaveSplitEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer
import SplitModelFunctor
from LLMRunner import LLMRunner

# 输入模型路径和embedding路径，用于把训练过的部分embedding记录到词表里面
def load_embedding_inference(model_path, embedding_checkpoint_path):
    # 加载checkpoint
    embedding_train, out_head_train = SaveSplitEmbedding.load_embedding_checkpoint(embedding_checkpoint_path)
    # 打印两个 tensor 的 shape
    print(embedding_train.shape)
    print(out_head_train.shape)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    ).to('cuda')
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 从model里面获取embedding和linear头
    embedding = SplitModelFunctor.get_model_embedding(model)
    out_head = SplitModelFunctor.get_model_out_head(model)
    embedding.requires_grad_(False)
    out_head.requires_grad_(False)

    # 临时测试的解码信息
    # token_list = [198,151644,77091,198, 151667, 198, 99692, 3837]
    # for each_token in token_list:
    #     temp_str = tokenizer.decode(each_token, skip_special_tokens=False)
    #     print(each_token, temp_str)
    # exit()

    # 训练的col num
    train_col_num = embedding_train.shape[0]
    if (train_col_num != out_head_train.shape[0]):
        raise RuntimeError('embedding and out_head not match')

    # 现在怎样从embedding里面取出 weight
    embedding_weight = embedding.weight
    print(type(embedding_weight))
    print(embedding_weight.shape)

    linear_weight = out_head.weight
    print(linear_weight.shape)

    embedding_weight[-train_col_num:, :] = embedding_train.clone()
    linear_weight[-train_col_num:, :] = out_head_train.clone()

    # 加载好 checkpoint之后就可以重新开始推理了
    llm_runner = LLMRunner(model = model, model_type='qwen3', 
        tokenizer = tokenizer, enable_thinking=True)
    # 调用 runner做一下推理
    print(llm_runner.inference("男性和女性在生理上的主要区别是什么？"))
    