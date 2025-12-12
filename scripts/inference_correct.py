# scripts/inference_correct.py
"""
正确的推理脚本 - 处理词汇表大小不匹配问题
"""

import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

class CorrectMethodNamingInference:
    """正确处理词汇表大小的推理器"""
    
    def __init__(self, model_dir):
        """初始化推理引擎"""
        self.model_dir = model_dir
        
        print(f"Loading model from {model_dir}...")
        
        # 先加载tokenizer来获取词汇表大小
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # 设置填充token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 获取tokenizer的词汇表大小
        vocab_size = len(self.tokenizer)
        print(f"Tokenizer vocab size: {vocab_size}")
        
        # 加载模型时忽略不匹配的大小
        try:
            # 方法1：尝试加载LoRA适配器
            from peft import PeftModel, PeftConfig
            
            # 先加载基础模型
            print("Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                "unsloth/Qwen2.5-Coder-0.5B",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # 加载LoRA适配器
            print("Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(base_model, model_dir)
            
            # 合并权重用于推理
            print("Merging weights for inference...")
            self.model = self.model.merge_and_unload()
            
        except Exception as e1:
            print(f"LoRA loading failed: {e1}")
            print("Trying direct load with ignore mismatches...")
            
            try:
                # 方法2：直接加载，忽略大小不匹配
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                    ignore_mismatched_sizes=True  # 关键：忽略大小不匹配
                )
            except Exception as e2:
                print(f"Direct load failed: {e2}")
                print("Trying simple load...")
                
                # 方法3：最简单的方式
                self.model = AutoModelForCausalLM.from_pretrained(model_dir)
                
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
        
        self.model.eval()
        print("✅ Model loaded successfully")
    
    def _find_method_name_position(self, method_body):
        """查找方法名位置"""
        lines = method_body.strip().split('\n')
        if not lines:
            return None, None
        
        java_types = {
            'void', 'int', 'String', 'boolean', 'float', 'double', 'long',
            'char', 'byte', 'short', 'List', 'Map', 'Set', 'ArrayList',
            'HashMap', 'HashSet', 'Object', 'public', 'private', 'protected',
            'static', 'final', 'abstract', 'synchronized', 'native'
        }
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('//') or line_stripped.startswith('/*'):
                continue
            
            if '(' in line and ')' in line:
                before_paren = line.split('(')[0].strip()
                words = before_paren.split()
                
                if len(words) >= 2:
                    for word in reversed(words):
                        clean_word = word.strip('*&<>[]')
                        if clean_word and clean_word not in java_types:
                            start_idx = line.rfind(clean_word)
                            if start_idx != -1:
                                return i, start_idx
        
        return None, None
    
    def create_fim_input(self, method_body):
        """创建FIM格式输入"""
        result = self._find_method_name_position(method_body)
        if result is None:
            return None
        
        line_idx, char_idx = result
        lines = method_body.strip().split('\n')
        
        if line_idx >= len(lines):
            return None
        
        signature_line = lines[line_idx]
        
        # 找到要掩码的词
        before_paren = signature_line.split('(')[0]
        words = before_paren.split()
        java_types = {'void', 'int', 'String', 'boolean', 'float', 'double', 'long'}
        
        method_name = None
        for word in reversed(words):
            clean_word = word.strip('*&<>[]')
            if clean_word and clean_word not in java_types:
                method_name = clean_word
                break
        
        if method_name is None:
            return None
        
        # 创建掩码行
        masked_line = signature_line.replace(method_name, "<MASK>", 1)
        lines[line_idx] = masked_line
        masked_body = '\n'.join(lines)
        
        # 创建FIM格式
        mask_pos = masked_body.find("<MASK>")
        if mask_pos == -1:
            return None
        
        prefix = masked_body[:mask_pos]
        suffix = masked_body[mask_pos + 6:]  # "<MASK>"的长度
        
        fim_input = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
        return fim_input
    
    def predict_method_name(self, method_body):
        """预测方法名"""
        fim_input = self.create_fim_input(method_body)
        if not fim_input:
            return ""
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                fim_input,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # 移动到模型设备
            device = self.model.device if hasattr(self.model, 'device') else 'cpu'
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # 解码
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # 提取方法名
            start_marker = "<|fim_middle|>"
            end_marker = "<|endoftext|>"
            
            start_idx = generated.find(start_marker)
            if start_idx != -1:
                start_idx += len(start_marker)
                end_idx = generated.find(end_marker, start_idx)
                if end_idx != -1:
                    predicted = generated[start_idx:end_idx].strip()
                    predicted = predicted.split('<')[0].strip()
                    return predicted
            
            return ""
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return ""
    
    def evaluate(self, test_data, max_samples=100):
        """评估模型"""
        if max_samples:
            test_data = test_data[:max_samples]
        
        correct = 0
        results = []
        
        print(f"Evaluating {len(test_data)} samples...")
        
        for i, item in enumerate(test_data):
            true_name = item.get('name', '')
            method_body = item.get('body', '')
            
            if not true_name or not method_body:
                results.append({
                    "index": i,
                    "true_name": true_name,
                    "predicted_name": "",
                    "correct": False,
                    "error": "Missing data"
                })
                continue
            
            try:
                predicted_name = self.predict_method_name(method_body)
                
                # 宽松匹配（忽略大小写和下划线）
                true_clean = true_name.lower().replace('_', '')
                pred_clean = predicted_name.lower().replace('_', '')
                
                match = (true_clean == pred_clean)
                
                if match:
                    correct += 1
                
                results.append({
                    "index": i,
                    "true_name": true_name,
                    "predicted_name": predicted_name,
                    "correct": match
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "true_name": true_name,
                    "predicted_name": "",
                    "correct": False,
                    "error": str(e)
                })
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(test_data)} samples...")
        
        accuracy = correct / len(test_data) * 100 if test_data else 0
        
        return accuracy, results

def load_test_data(test_path, max_samples=None):
    """加载测试数据"""
    test_data = []
    try:
        with open(test_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                data = json.loads(line.strip())
                if 'name' in data and 'body' in data:
                    test_data.append({
                        'name': data['name'],
                        'body': data['body']
                    })
        print(f"Loaded {len(test_data)} test samples")
        return test_data
    except Exception as e:
        print(f"Error loading test data: {e}")
        return []

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Java method naming model')
    parser.add_argument('--model-dir', required=True, help='Path to trained model')
    parser.add_argument('--test-data', required=True, help='Path to test data')
    parser.add_argument('--max-samples', type=int, default=100, help='Max samples to evaluate')
    parser.add_argument('--output', default='evaluation_results.json', help='Output file')
    
    args = parser.parse_args()
    
    # 初始化推理器
    inference = CorrectMethodNamingInference(args.model_dir)
    
    # 加载测试数据
    test_data = load_test_data(args.test_data, args.max_samples)
    
    # 评估
    accuracy, results = inference.evaluate(test_data, args.max_samples)
    
    # 保存结果
    output_data = {
        "accuracy": accuracy,
        "evaluated_samples": len(results),
        "results": results[:20]
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {args.output}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
