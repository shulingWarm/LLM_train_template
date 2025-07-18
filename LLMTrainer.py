import time
import random
from tqdm import tqdm
import torch
from DataProvder import DataProvder

class LLMTrainer:
    def __init__(self, model, data_provider: DataProvder, tokenizer, optimizer, 
                 scheduler=None, device='cuda', batch_size=16, epochs=3, 
                 max_length=512, logging_interval=100):
        """
        完整的LLM训练器
        
        参数:
            model: 要训练的模型
            data_provider: 数据提供器
            tokenizer: 文本分词器
            optimizer: 优化器
            scheduler: 学习率调度器（可选）
            device: 训练设备 ('cuda' 或 'cpu')
            batch_size: 批次大小
            epochs: 训练轮数
            max_length: 文本最大长度
            logging_interval: 日志间隔（多少步记录一次）
        """
        self.model = model.to(device)
        self.data_provider = data_provider
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_length = max_length
        self.logging_interval = logging_interval
        self.losses = []
        
        # 验证设备
        if device == 'cuda' and not torch.cuda.is_available():
            print("警告: CUDA不可用，将使用CPU训练")
            self.device = 'cpu'

    def collate_fn(self, batch):
        """处理批数据，进行分词和填充"""
        inputs = [item['input'] for item in batch]
        targets = [item['target'] for item in batch]
        
        # 为输入和目标分词
        model_inputs = self.tokenizer(
            inputs, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        labels = self.tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )["input_ids"]
        
        # 将标签中填充token的ID设为-100，这样损失函数会忽略它们
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": model_inputs["input_ids"].to(self.device),
            "attention_mask": model_inputs["attention_mask"].to(self.device),
            "labels": labels.to(self.device)
        }
    
    def get_batches(self):
        """生成批数据"""
        total_data = self.data_provider.getDataNum()
        indices = list(range(total_data))
        random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = []
            
            for idx in batch_indices:
                input_text, target_text = self.data_provider.getDataByIndex(idx)
                batch_data.append({
                    "input": input_text,
                    "target": target_text
                })
            
            yield self.collate_fn(batch_data)

    def train(self):
        """开始训练循环"""
        total_steps = (self.data_provider.getDataNum() // self.batch_size) * self.epochs
        global_step = 0
        start_time = time.time()
        
        print(f"开始训练，总数据量: {self.data_provider.getDataNum()}")
        print(f"批大小: {self.batch_size}, 总轮数: {self.epochs}")
        print(f"估计总步数: {total_steps}")

        self.model.train()
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            print(f"\n=== 开始第 {epoch+1}/{self.epochs} 轮训练 ===")
            
            # 创建进度条
            total_batches = self.data_provider.getDataNum() // self.batch_size
            progress_bar = tqdm(total=total_batches, desc=f"轮次 {epoch+1}")
            
            for batch in self.get_batches():
                # 前向传播
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                # 记录损失
                self.losses.append(loss.item())
                global_step += 1
                
                # 定期打印日志
                if global_step % self.logging_interval == 0:
                    avg_loss = sum(self.losses[-self.logging_interval:]) / self.logging_interval
                    print(f"\n步骤 {global_step}/{total_steps} - 损失: {avg_loss:.4f}")
                    if self.scheduler:
                        lr = self.scheduler.get_last_lr()[0]
                        print(f"学习率: {lr:.2e}")
                
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            progress_bar.close()
            epoch_time = time.time() - epoch_start
            print(f"轮次 {epoch+1} 完成, 耗时: {epoch_time:.2f}秒")
        
        total_time = time.time() - start_time
        print(f"\n训练完成! 总耗时: {total_time:.2f}秒")
        print(f"平均损失: {sum(self.losses) / len(self.losses):.4f}")
        return self.model