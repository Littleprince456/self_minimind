import os
import sys
import json
import argparse
from transformers import AutoTokenizer

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from dataset.lm_dataset import pre_processing_chat, post_processing_chat

def view_sft_data(args):
    data_path = args.data_path
    
    if not os.path.exists(data_path):
        print(f"⚠️ 未找到 SFT 数据文件: {data_path}")
        print("请检查路径是否正确，或者先下载数据文件放到该目录下。")
        return

    # 加载 Tokenizer（用于展示 Chat Template 处理后的格式）
    tokenizer = None
    if os.path.exists(args.tokenizer_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
            print(f"✅ 成功加载 Tokenizer: {args.tokenizer_path}\n")
        except Exception as e:
            print(f"⚠️ 加载 Tokenizer 失败: {e}\n")
    else:
        print(f"⚠️ Tokenizer 路径不存在: {args.tokenizer_path}，将仅展示原始 JSON 数据。\n")

    print(f"====== 开始读取 SFT 数据: {data_path} ======\n")
    
    count = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                print("⚠️ 格式损坏的行，跳过...")
                continue
                
            print(f"[{count + 1}] 原始数据 (Raw JSON):")
            print(json.dumps(sample, ensure_ascii=False, indent=2))
            
            # 如果成功加载了 Tokenizer，展示模型实际接收到的 Prompt
            if tokenizer and 'conversations' in sample:
                conversations = sample['conversations']
                
                # SFTDataset 中的前处理
                processed_conv = pre_processing_chat(conversations)
                
                # 尝试解析 tools
                messages = []
                tools = None
                for message in processed_conv:
                    message = dict(message)
                    if message.get("role") == "system" and message.get("tools"):
                        tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
                    if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                        message["tool_calls"] = json.loads(message["tool_calls"])
                    messages.append(message)
                
                try:
                    # 应用 chat template
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                        tools=tools
                    )
                    
                    # SFTDataset 中的后处理
                    prompt = post_processing_chat(prompt)
                    
                    print("\n[模型实际输入的 Prompt (Chat Template 后)]:")
                    print("-" * 50)
                    print(prompt)
                    print("-" * 50)
                except Exception as e:
                    print(f"\n⚠️ 应用 Chat Template 失败: {e}")
                    
            print("\n" + "="*80 + "\n")
            
            count += 1
            if count >= args.num_samples:
                break
                
    if count == 0:
        print("数据文件为空！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查看 SFT 数据集样本")
    parser.add_argument('--data_path', type=str, default=os.path.join(PROJECT_ROOT, 'dataset/sft_t2t_mini.jsonl'), help="SFT 数据文件路径")
    parser.add_argument('--tokenizer_path', type=str, default=os.path.join(PROJECT_ROOT, 'model'), help="Tokenizer 路径")
    parser.add_argument('--num_samples', type=int, default=3, help="要打印的样本数量")
    
    args = parser.parse_args()
    view_sft_data(args)
