import os
import sys
import time
import argparse
import torch
from transformers import AutoTokenizer

# 将项目根目录加入 sys.path，以便导入 model 等模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params

def check_pretrain_model(args):
    device = args.device
    
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    print(f"Initializing MoE Causal LM with hidden_size={args.hidden_size}...")
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=True,
        inference_rope_scaling=False
    )
    model = MiniMindForCausalLM(config)
    
    if os.path.exists(args.model_path):
        print(f"Loading weights from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path, map_location=device), strict=True)
    else:
        print(f"⚠️ 警告: 未找到模型文件 {args.model_path}")
        print("⚠️ 继续使用随机初始化的权重，以测试脚本运行逻辑。\n")
    
    model = model.half().eval().to(device)
    get_model_params(model, model.config)
    
    prompts = [
        '为什么天空是蓝色的',
        '解释什么是机器学习',
    ]
    
    input_mode = input('[0] 自动测试\n[1] 手动输入\n')
    try:
        input_mode = int(input_mode)
    except ValueError:
        input_mode = 0
        
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    
    # 准备日志文件
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'check_pretrain.log')
    
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n{'='*40}\n")
        log_file.write(f"Testing start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for prompt in prompt_iter:
            if not prompt.strip():
                continue
                
            setup_seed(42)
            if input_mode == 0: 
                print(f'\n💬: {prompt}')
            
            # 预训练模型通常只拼接 bos_token，不做复杂的 Chat Template 格式化
            text_input = tokenizer.bos_token + prompt
            
            # 1. 打印 Tokenization 结果
            inputs = tokenizer(text_input, return_tensors="pt")
            input_ids = inputs["input_ids"][0].tolist()
            tokens = [tokenizer.decode([token_id]) for token_id in input_ids]
            
            tokenization_info = (
                f"\n--- Tokenization Info ---\n"
                f"Raw Input: {text_input}\n"
                f"Input IDs: {input_ids}\n"
                f"Tokens: {tokens}\n"
                f"-------------------------\n"
            )
            print(tokenization_info)
            log_file.write(f"💬 Prompt: {prompt}\n")
            log_file.write(tokenization_info)
            
            inputs = inputs.to(device)
            
            print('🧠: ', end='', flush=True)
            st = time.time()
            
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    repetition_penalty=1
                )
            
            # 2. 打印生成文本，包含 <SOS> <EOS> 等特殊 token（设置 skip_special_tokens=False）
            full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
            print(full_output)
            log_file.write(f"🧠 Generated (Raw): {full_output}\n")
            
            # 3. 打印生成的 Token 详情
            output_ids = generated_ids[0].tolist()
            new_ids = output_ids[len(input_ids):]
            new_tokens = [tokenizer.decode([token_id]) for token_id in new_ids]
            
            output_details = (
                f"\n--- Output Details ---\n"
                f"Generated Output IDs: {new_ids}\n"
                f"Generated Tokens: {new_tokens}\n"
                f"----------------------\n"
            )
            print(output_details)
            log_file.write(output_details)
            
            gen_tokens = len(new_ids)
            speed_info = f'[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n'
            print(speed_info)
            log_file.write(speed_info + "\n")

    print(f"\n日志已保存至: {log_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Pretrained MoE Model")
    parser.add_argument('--tokenizer_path', default=os.path.join(PROJECT_ROOT, 'model'), type=str, help="Tokenizer path")
    parser.add_argument('--model_path', default=os.path.join(PROJECT_ROOT, 'trainer/out/pretrain_768_moe.pth'), type=str, help="Model checkpoint path")
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_new_tokens', default=512, type=int)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.95, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    
    args = parser.parse_args()
    check_pretrain_model(args)
