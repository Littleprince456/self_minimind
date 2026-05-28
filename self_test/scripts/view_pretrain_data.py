import sys
import os

# 将根目录加入路径以便导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataset.lm_dataset import PretrainDataset

class SafePretrainDataset(PretrainDataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # 逐行读取并捕获 JSON 解析错误，绕过 pyarrow 的严格检查
        # 使用 errors='ignore' 忽略 UTF-8 解码错误
        with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    import json
                    self.samples.append(json.loads(line))
                except Exception as e:
                    print(f"警告：忽略第 {line_num} 行的损坏数据 - {e}")
                    continue

def main():
    # 配置路径
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dataset/pretrain_t2t_mini.jsonl'))
    tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../model'))
    
    # 初始化 tokenizer 和 dataset
    print("正在加载 Tokenizer 和 Dataset...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"加载 Tokenizer 失败: {e}")
        return

    # 初始化 Dataset，设置较小的 max_length 方便观察
    # 使用自定义的 SafePretrainDataset 来兼容损坏的 JSONL 行
    dataset = SafePretrainDataset(data_path, tokenizer, max_length=128)
    
    # 模拟训练时的 DataLoader，设置 batch_size 为 32
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 获取第一个 batch
    print("\n获取第一个 Batch...")
    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        print(f"\n==================== Batch {batch_idx + 1} ====================")
        print(f"Batch 输入张量形状 (input_ids): {input_ids.shape} -> (batch_size, max_seq_len)")
        print(f"Batch 标签张量形状 (labels): {labels.shape}")
        
        # 挑选这个 batch 中的前两条数据打印详细内容
        for i in range(2):
            print(f"\n--- Batch 内的第 {i+1} 条数据 ---")
            
            # 1. 打印原始的 Token ID
            ids = input_ids[i].tolist()
            print(f"【Input IDs (截断显示前20个)】:\n{ids[:20]} ...")
            
            # 2. 将 ID 解码回人类可读的文本
            # 预训练数据通常是连续的文本，解码出来就是一句或多句话
            decoded_text = tokenizer.decode(ids, skip_special_tokens=False)
            print(f"【解码后的文本内容】:\n{decoded_text}")
            
            # 3. 打印对应的 Label
            # 预训练是自回归任务 (Next Token Prediction)，所以 Label 理论上应该和 input_ids 类似
            # 可能会发生偏移或者对 padding 进行忽略（通常设为 -100）
            lbls = labels[i].tolist()
            print(f"【Labels (截断显示前20个)】:\n{lbls[:20]} ...")
            
        break # 我们只看第一个 batch 即可

if __name__ == "__main__":
    main()
