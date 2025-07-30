from swift.llm.train import SwiftSft
from swift.llm.argument import TrainArguments
from SplitModelFunctor import SplitModelFunctor
from TokenizerFunctor.OriginTokenizerFunctor import OriginTokenizerFunctor
from TokenizerFunctor.SplitEmbeddingFunctor import SplitEmbeddingFunctor

# 通过代码启动sft的训练过程
def launch_swift_sft(model_path,
    train_type, #训练的数据类型
    dataset_path, # 数据集的路径
    torch_dtype, # 训练过程中的torch数据类型
    num_train_epochs, # 训练的epoch个数
    output_dir, # 输出路径
    tokenizer_shell = None, # tokenizer的封装处理
    system = 'You are a helpful assistant.',
    per_device_train_batch_size = 1,
    per_device_eval_batch_size = 1,
    learning_rate = 1e-4,
    lora_rank = 8, # lora相关的配置
    lora_alpha = 32,
    target_modules = ['all-linear'],
    gradient_accumulation_steps = 16,
    eval_steps = 50,
    save_steps = 50,
    save_total_limit = 2,
    logging_steps = 5,
    max_length = 2048,
    warmup_ratio = 0.05,
    dataloader_num_workers = 4,
    train_col_num = 2, # 如果是embedding的部分训练，需要训练几列
    model_author = 'zSeal',
    model_name = 'zSeal-robot'
):
    if isinstance(dataset_path, str):
        dataset_path = [dataset_path]
    # 如果使用的是部分embedding的训练形式，需要添加一个回调
    parameter_config_callback = None
    if train_type == 'part_embedding':
        # 指定functor
        parameter_config_callback = SplitModelFunctor(train_col_num)
    # 新建train args
    train_args = TrainArguments(
        model = model_path,
        train_type = train_type,
        dataset = dataset_path,
        model_type='qwen3',
        torch_dtype = torch_dtype,
        num_train_epochs = num_train_epochs,
        per_device_train_batch_size = per_device_train_batch_size,
        per_device_eval_batch_size = per_device_eval_batch_size,
        learning_rate = learning_rate,
        lora_rank = lora_rank,
        lora_alpha = lora_alpha,
        target_modules = target_modules,
        gradient_accumulation_steps = gradient_accumulation_steps,
        eval_steps = eval_steps,
        save_steps = save_steps,
        save_total_limit = save_total_limit,
        logging_steps = logging_steps,
        max_length = max_length,
        output_dir = output_dir,
        system = system,
        warmup_ratio = warmup_ratio,
        dataloader_num_workers = dataloader_num_workers,
        model_author = model_author,
        model_name = model_name,
        load_from_cache_file = False
    )

    train_args.parameter_config_callback = parameter_config_callback
    # 记录tokenizer的shell
    train_args.tokenizer_shell = tokenizer_shell
    # 如果训练类型是part_embedding，就给每个训练句子强行带上这个训练token
    if(train_type == 'part_embedding'):
        train_args.tokenizer_functor = SplitEmbeddingFunctor(train_col_num = train_col_num)
    else:
        # 记录tokenizer的functor
        train_args.tokenizer_functor = None

    # 新建sft的实体
    sft_instance = SwiftSft(train_args)
    # 调用实体里面的main
    sft_instance.main()