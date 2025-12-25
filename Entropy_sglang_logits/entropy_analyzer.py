import json
import re
import numpy as np
import requests
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PhaseEntropy:
    """存储某个Phase的熵统计结果"""
    phase_name: str
    entropies: List[float] = field(default_factory=list)
    token_count: int = 0
    
    @property
    def average_entropy(self) -> float:
        if not self.entropies:
            return 0.0
        return float(np.mean(self.entropies))
    
    @property
    def std_entropy(self) -> float:
        if len(self.entropies) < 2:
            return 0.0
        return float(np.std(self.entropies))
    
    def add_entropies(self, entropies: List[float]):
        self.entropies.extend(entropies)
        self.token_count += len(entropies)


class SGLangEntropyAnalyzer:
    """使用 sglang API 进行熵统计分析"""
    
    def __init__(self, base_url: str = "http://localhost:6001"):
        self.base_url = base_url.rstrip('/')
        self.model_name = self._get_model_name()
        
        # Phase标记正则表达式
        self.phase_patterns = {
            'think': re.compile(r'<think>(.*?)</think>', re.DOTALL),
            'tool_call': re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL),
            'tool_response': re.compile(r'<tool_response>(.*?)</tool_response>', re.DOTALL),
            'answer': re.compile(r'<answer>(.*?)</answer>', re.DOTALL),
        }
        
        logger.info(f"初始化完成，模型: {self.model_name}")
    
    def _get_model_name(self) -> str:
        """获取模型名称"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    return data['data'][0]['id']
        except Exception as e:
            logger.warning(f"获取模型名称失败: {e}")
        return "default"
    
    def calculate_entropy_from_top_logprobs(self, top_logprobs_dict: Dict[str, float]) -> float:
        """从 top_logprobs 字典计算熵"""
        if not top_logprobs_dict:
            return 0.0
        
        logprobs = list(top_logprobs_dict.values())
        probs = np.exp(np.array(logprobs, dtype=np.float64))
        
        prob_sum = probs.sum()
        if prob_sum <= 0:
            return 0.0
        probs = probs / prob_sum
        
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return float(entropy)
    
    def get_completion_with_logprobs(self, prompt: str, max_tokens: int = 1024) -> Optional[Dict]:
        """获取 completion 和 logprobs"""
        url = f"{self.base_url}/v1/completions"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "logprobs": 10,
        }
        
        try:
            response = requests.post(url, json=payload, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                return self._parse_response(result)
            else:
                logger.warning(f"API返回错误 {response.status_code}: {response.text[:200]}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("请求超时")
            return None
        except Exception as e:
            logger.error(f"请求失败: {e}")
            return None
    
    def _parse_response(self, result: Dict) -> Optional[Dict]:
        """解析 API 响应"""
        if 'choices' not in result or len(result['choices']) == 0:
            return None
        
        choice = result['choices'][0]
        logprobs_data = choice.get('logprobs', {})
        
        return {
            'text': choice.get('text', ''),
            'tokens': logprobs_data.get('tokens', []),
            'token_logprobs': logprobs_data.get('token_logprobs', []),
            'top_logprobs': logprobs_data.get('top_logprobs', []),
        }
    
    def compute_token_entropies(self, top_logprobs: List[Dict[str, float]]) -> List[float]:
        """计算每个 token 位置的熵"""
        entropies = []
        for top_lp in top_logprobs:
            if top_lp is None or not isinstance(top_lp, dict):
                entropies.append(0.0)
                continue
            entropy = self.calculate_entropy_from_top_logprobs(top_lp)
            entropies.append(entropy)
        return entropies
    
    def extract_phases_with_token_ranges(
        self,
        content: str,
        tokens: List[str]
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        提取各个 phase 对应的 token 索引范围
        返回: {phase_name: [(start_token_idx, end_token_idx), ...]}
        """
        phases = {phase: [] for phase in self.phase_patterns.keys()}
        
        if not tokens:
            return phases
        
        # 重建 token 的字符位置映射
        token_char_positions = []  # [(char_start, char_end), ...]
        current_pos = 0
        
        for token in tokens:
            token_start = current_pos
            token_end = current_pos + len(token)
            token_char_positions.append((token_start, token_end))
            current_pos = token_end
        
        # 对每个 phase pattern，找到其在 content 中的字符范围
        for phase_name, pattern in self.phase_patterns.items():
            for match in pattern.finditer(content):
                # 标签内部内容的字符位置
                tag_open = f'<{phase_name}>'
                tag_close = f'</{phase_name}>'
                inner_char_start = match.start() + len(tag_open)
                inner_char_end = match.end() - len(tag_close)
                
                # 找到对应的 token 范围
                start_token_idx = None
                end_token_idx = None
                
                for i, (t_start, t_end) in enumerate(token_char_positions):
                    # token 与 phase 内容有交集
                    if t_end > inner_char_start and t_start < inner_char_end:
                        if start_token_idx is None:
                            start_token_idx = i
                        end_token_idx = i + 1
                
                if start_token_idx is not None:
                    phases[phase_name].append((start_token_idx, end_token_idx))
        
        return phases
    
    def build_prompt_for_message(self, messages: List[Dict], target_msg_idx: int) -> str:
        """
        构建用于让模型续写第 target_msg_idx 个 message 的 prompt
        """
        prompt_parts = []
        
        for i, msg in enumerate(messages):
            if i >= target_msg_idx:
                break
            
            role = msg.get('role', '')
            content = msg.get('content', '')
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        # 添加目标 message 的开头
        target_role = messages[target_msg_idx].get('role', 'assistant')
        prompt_parts.append(f"<|im_start|>{target_role}\n")
        
        return "\n".join(prompt_parts)
    
    def analyze_single_sample(self, messages: List[Dict]) -> Dict[str, List[float]]:
        """分析单个样本中所有 assistant 生成的内容"""
        all_phase_entropies = {phase: [] for phase in self.phase_patterns.keys()}
        
        for msg_idx, msg in enumerate(messages):
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if not content or len(content) < 5:
                continue
            
            # 检测这个 message 中有哪些 phase
            detected_phases = []
            for phase_name, pattern in self.phase_patterns.items():
                if pattern.search(content):
                    detected_phases.append(phase_name)
            
            if not detected_phases:
                logger.debug(f"Message {msg_idx} (role={role}): 未检测到任何 phase 标签，跳过")
                continue
            
            logger.info(f"Message {msg_idx} (role={role}): 检测到 phases: {detected_phases}")
            logger.debug(f"  Content preview: {content[:100]}...")
            
            # 构建 prompt
            prompt = self.build_prompt_for_message(messages, msg_idx)
            
            # 估算需要生成的 token 数
            estimated_tokens = int(len(content) * 1.5)
            max_gen_tokens = min(max(estimated_tokens, 512), 8192)
            
            # 获取 logprobs
            result = self.get_completion_with_logprobs(prompt, max_tokens=max_gen_tokens)
            
            if result is None:
                logger.warning(f"获取 logprobs 失败，跳过 message {msg_idx}")
                continue
            
            tokens = result.get('tokens', [])
            top_logprobs = result.get('top_logprobs', [])
            
            if not tokens or not top_logprobs:
                logger.warning(f"Message {msg_idx}: tokens 或 logprobs 为空")
                continue
            
            # 计算每个 token 的熵
            token_entropies = self.compute_token_entropies(top_logprobs)
            
            # 重建生成的文本
            generated_text = ''.join(tokens)
            
            logger.info(f"Message {msg_idx}: 生成了 {len(tokens)} 个 tokens")
            logger.debug(f"  原始 content 长度: {len(content)}")
            logger.debug(f"  生成文本长度: {len(generated_text)}")
            logger.debug(f"  生成文本预览: {generated_text[:150]}...")
            
            # 检查生成的文本中是否包含预期的 phase
            gen_detected = []
            for phase_name, pattern in self.phase_patterns.items():
                if pattern.search(generated_text):
                    gen_detected.append(phase_name)
            logger.info(f"  生成文本中检测到的 phases: {gen_detected}")
            
            # 在生成的文本中定位 phases
            phase_token_ranges = self.extract_phases_with_token_ranges(generated_text, tokens)
            
            # 收集每个 phase 的熵
            for phase_name, ranges in phase_token_ranges.items():
                for start_idx, end_idx in ranges:
                    phase_entropies = token_entropies[start_idx:end_idx]
                    all_phase_entropies[phase_name].extend(phase_entropies)
                    
                    if phase_entropies:
                        logger.info(
                            f"  Phase '{phase_name}': tokens[{start_idx}:{end_idx}] = {end_idx - start_idx} tokens, "
                            f"avg_entropy={np.mean(phase_entropies):.4f}"
                        )
        
        return all_phase_entropies
    
    def analyze_dataset(self, data: List[Dict], max_samples: int = None) -> Dict[str, PhaseEntropy]:
        """分析整个数据集"""
        phase_stats = {
            'think': PhaseEntropy('Think Phase (思考阶段)'),
            'tool_call': PhaseEntropy('Action Phase (工具调用)'),
            'tool_response': PhaseEntropy('Tool Response (工具响应)'),
            'answer': PhaseEntropy('Answer Phase (回答阶段)'),
        }
        
        samples = data[:max_samples] if max_samples else data
        logger.info(f"开始分析 {len(samples)} 个样本...")
        
        successful = 0
        for sample in tqdm(samples, desc="分析进度"):
            messages = sample.get('messages', [])
            if not messages:
                continue
            
            try:
                sample_entropies = self.analyze_single_sample(messages)
                
                has_data = False
                for phase_name, entropies in sample_entropies.items():
                    if entropies:
                        phase_stats[phase_name].add_entropies(entropies)
                        has_data = True
                
                if has_data:
                    successful += 1
                        
            except Exception as e:
                logger.warning(f"样本分析出错: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
        
        logger.info(f"成功分析 {successful}/{len(samples)} 个样本")
        return phase_stats
    
    def print_results(self, phase_stats: Dict[str, PhaseEntropy]):
        """打印结果"""
        print("\n" + "=" * 70)
        print("            熵统计分析结果 (Entropy Analysis Results)")
        print("=" * 70)
        
        for phase_name, stats in phase_stats.items():
            print(f"\n📊 【{stats.phase_name}】")
            print(f"   ├─ Token数量:      {stats.token_count:,}")
            print(f"   ├─ 平均熵 (Avg):   {stats.average_entropy:.4f}")
            print(f"   ├─ 标准差 (Std):   {stats.std_entropy:.4f}")
            if stats.entropies:
                print(f"   ├─ 最小熵 (Min):   {min(stats.entropies):.4f}")
                print(f"   ├─ 最大熵 (Max):   {max(stats.entropies):.4f}")
                print(f"   └─ 中位数 (Med):   {float(np.median(stats.entropies)):.4f}")
            else:
                print(f"   └─ (无数据)")
        
        print("\n" + "=" * 70)
    
    def save_results(self, phase_stats: Dict[str, PhaseEntropy], output_path: str):
        """保存结果到 JSON"""
        results = {
            'summary': {},
            'details': {}
        }
        
        for phase_name, stats in phase_stats.items():
            results['details'][phase_name] = {
                'phase_name': stats.phase_name,
                'token_count': stats.token_count,
                'average_entropy': stats.average_entropy,
                'std_entropy': stats.std_entropy,
            }
            results['summary'][phase_name] = {
                'avg_entropy': stats.average_entropy,
                'token_count': stats.token_count
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已保存到: {output_path}")


def diagnose_data(input_path: str):
    """诊断数据文件中的 phase 分布"""
    print("\n" + "=" * 70)
    print("🔍 数据诊断")
    print("=" * 70)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n📂 文件: {input_path}")
    print(f"📊 样本总数: {len(data)}")
    
    phase_patterns = {
        'think': re.compile(r'<think>(.*?)</think>', re.DOTALL),
        'tool_call': re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL),
        'tool_response': re.compile(r'<tool_response>(.*?)</tool_response>', re.DOTALL),
        'answer': re.compile(r'<answer>(.*?)</answer>', re.DOTALL),
    }
    
    # 统计各 role 和 phase 的分布
    role_counts = {}
    phase_by_role = {phase: {} for phase in phase_patterns.keys()}
    
    for sample_idx, sample in enumerate(data[:10]):  # 只看前10个样本
        messages = sample.get('messages', [])
        
        print(f"\n--- Sample {sample_idx} ({len(messages)} messages) ---")
        
        for msg_idx, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            
            role_counts[role] = role_counts.get(role, 0) + 1
            
            detected = []
            for phase_name, pattern in phase_patterns.items():
                matches = pattern.findall(content)
                if matches:
                    detected.append(f"{phase_name}({len(matches)})")
                    phase_by_role[phase_name][role] = phase_by_role[phase_name].get(role, 0) + len(matches)
            
            if detected or role == 'assistant':
                content_preview = content[:60].replace('\n', '↵') if content else "(empty)"
                print(f"  [{msg_idx}] role={role:10s} phases={detected or 'none':30s} | {content_preview}...")
    
    print("\n" + "-" * 70)
    print("📈 Phase 分布按 Role 统计 (前10个样本):")
    for phase_name, role_dist in phase_by_role.items():
        print(f"  {phase_name:15s}: {role_dist}")
    
    print("\n💡 如果 tool_response 只出现在非 assistant role，需要调整代码逻辑")
    print("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Agent 熵统计分析工具')
    parser.add_argument('--input', '-i', type=str, help='输入 JSON 文件路径')
    parser.add_argument('--output', '-o', type=str, default='entropy_results.json', help='输出文件')
    parser.add_argument('--base-url', type=str, default='http://localhost:6001', help='sglang 服务地址')
    parser.add_argument('--limit', type=int, default=None, help='限制样本数量')
    parser.add_argument('--test', action='store_true', help='测试 API 连接')
    parser.add_argument('--diagnose', action='store_true', help='诊断数据文件中的 phase 分布')
    parser.add_argument('--debug', action='store_true', help='显示调试信息')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.diagnose and args.input:
        diagnose_data(args.input)
        return
    
    if args.test:
        analyzer = SGLangEntropyAnalyzer(args.base_url)
        result = analyzer.get_completion_with_logprobs("Hello", max_tokens=5)
        if result:
            print("✅ API 连接成功")
            print(f"   生成: {result['text']}")
        else:
            print("❌ API 连接失败")
        return
    
    if not args.input:
        print("用法:")
        print("  诊断数据:  python script.py --diagnose -i data.json")
        print("  运行分析:  python script.py -i data.json --debug --limit 2")
        return
    
    # 运行分析
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    analyzer = SGLangEntropyAnalyzer(args.base_url)
    phase_stats = analyzer.analyze_dataset(data, max_samples=args.limit)
    
    analyzer.print_results(phase_stats)
    analyzer.save_results(phase_stats, args.output)


if __name__ == '__main__':
    main()