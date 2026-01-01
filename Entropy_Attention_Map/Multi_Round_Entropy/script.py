import json
import re
import numpy as np
import requests
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import BrokenBarHCollection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


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
        
        # Phase 颜色配置
        self.phase_colors = {
            'think': '#FF6B6B',        # 红色
            'tool_call': '#4ECDC4',    # 青色
            'tool_response': '#45B7D1', # 蓝色
            'answer': '#96CEB4',        # 绿色
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
    
    def analyze_single_case(self, messages: List[Dict]) -> Dict:
        """
        分析单个 case，返回详细的 token 熵信息
        返回: {
            'tokens': List[str],
            'entropies': List[float],
            'phase_ranges': Dict[str, List[Tuple[int, int]]],
            'generated_text': str
        }
        """
        all_tokens = []
        all_entropies = []
        all_phase_ranges = {phase: [] for phase in self.phase_patterns.keys()}
        all_generated_text = ""
        
        token_offset = 0  # 用于累计 token 位置偏移
        
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
            
            # 在生成的文本中定位 phases
            phase_token_ranges = self.extract_phases_with_token_ranges(generated_text, tokens)
            
            # 累加到全局结果中，调整 token 范围的偏移
            all_tokens.extend(tokens)
            all_entropies.extend(token_entropies)
            all_generated_text += generated_text
            
            for phase_name, ranges in phase_token_ranges.items():
                for start_idx, end_idx in ranges:
                    # 加上偏移量
                    all_phase_ranges[phase_name].append((start_idx + token_offset, end_idx + token_offset))
            
            token_offset += len(tokens)
        
        return {
            'tokens': all_tokens,
            'entropies': all_entropies,
            'phase_ranges': all_phase_ranges,
            'generated_text': all_generated_text
        }
    
    def plot_entropy_curve(
        self, 
        case_data: Dict, 
        case_idx: int = 0,
        save_path: str = None,
        figsize: Tuple[int, int] = (16, 8),
        show_tokens: bool = False
    ):
        """
        绘制单个 case 的 token 熵变化曲线
        
        Args:
            case_data: analyze_single_case 返回的数据
            case_idx: case 索引（用于标题）
            save_path: 保存路径，如果为 None 则显示
            figsize: 图片大小
            show_tokens: 是否在 x 轴显示 token 文本
        """
        tokens = case_data['tokens']
        entropies = case_data['entropies']
        phase_ranges = case_data['phase_ranges']
        
        if not tokens or not entropies:
            logger.warning("没有数据可以绘制")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制熵曲线
        x = np.arange(len(entropies))
        ax.plot(x, entropies, 'b-', linewidth=0.8, alpha=0.7, label='Token Entropy')
        ax.scatter(x, entropies, s=10, c='blue', alpha=0.5)
        
        # 绘制 phase 背景区域
        y_min, y_max = ax.get_ylim()
        if y_max <= y_min:
            y_max = max(entropies) * 1.1 if entropies else 1.0
            y_min = 0
        
        legend_patches = []
        for phase_name, ranges in phase_ranges.items():
            if not ranges:
                continue
            
            color = self.phase_colors.get(phase_name, '#CCCCCC')
            
            for start_idx, end_idx in ranges:
                ax.axvspan(start_idx - 0.5, end_idx - 0.5, 
                          alpha=0.3, color=color, 
                          label=f'{phase_name}' if start_idx == ranges[0][0] else None)
                
                # 在区域顶部添加标签
                mid_x = (start_idx + end_idx) / 2
                ax.text(mid_x, y_max * 0.95, phase_name, 
                       ha='center', va='top', fontsize=9, 
                       color=color, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            # 添加到图例
            patch = mpatches.Patch(color=color, alpha=0.3, label=phase_name)
            legend_patches.append(patch)
        
        # 设置标签和标题
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Entropy', fontsize=12)
        ax.set_title(f'Token Entropy Curve - Case {case_idx}\n(Total Tokens: {len(tokens)})', fontsize=14)
        
        # 添加图例
        if legend_patches:
            ax.legend(handles=legend_patches, loc='upper right', fontsize=10)
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 设置 x 轴范围
        ax.set_xlim(-0.5, len(entropies) - 0.5)
        
        # 如果需要显示 token 文本
        if show_tokens and len(tokens) <= 100:
            # 显示部分 token 文本（避免太密集）
            step = max(1, len(tokens) // 50)
            tick_positions = range(0, len(tokens), step)
            tick_labels = [tokens[i][:10].replace('\n', '↵') for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
        
        # 添加统计信息
        stats_text = (
            f"Mean Entropy: {np.mean(entropies):.4f}\n"
            f"Std Entropy: {np.std(entropies):.4f}\n"
            f"Max Entropy: {np.max(entropies):.4f}\n"
            f"Min Entropy: {np.min(entropies):.4f}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 添加各 phase 的统计信息
        phase_stats_lines = []
        for phase_name, ranges in phase_ranges.items():
            if not ranges:
                continue
            phase_entropies = []
            for start_idx, end_idx in ranges:
                phase_entropies.extend(entropies[start_idx:end_idx])
            if phase_entropies:
                avg_e = np.mean(phase_entropies)
                phase_stats_lines.append(f"{phase_name}: avg={avg_e:.4f} (n={len(phase_entropies)})")
        
        if phase_stats_lines:
            phase_stats_text = "Phase Stats:\n" + "\n".join(phase_stats_lines)
            ax.text(0.02, 0.75, phase_stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"图表已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_and_plot_case(
        self, 
        data: List[Dict], 
        case_idx: int = 0,
        save_path: str = None,
        figsize: Tuple[int, int] = (16, 8),
        show_tokens: bool = False
    ) -> Dict:
        """
        分析并绘制单个 case 的熵曲线
        
        Args:
            data: 数据集
            case_idx: 要分析的 case 索引
            save_path: 图片保存路径
            figsize: 图片大小
            show_tokens: 是否显示 token 文本
            
        Returns:
            case_data: 分析结果
        """
        if case_idx >= len(data):
            logger.error(f"Case 索引 {case_idx} 超出范围 (共 {len(data)} 个样本)")
            return {}
        
        sample = data[case_idx]
        messages = sample.get('messages', [])
        
        if not messages:
            logger.error(f"Case {case_idx} 没有 messages")
            return {}
        
        logger.info(f"开始分析 Case {case_idx}...")
        case_data = self.analyze_single_case(messages)
        
        if case_data['tokens']:
            self.plot_entropy_curve(case_data, case_idx, save_path, figsize, show_tokens)
        else:
            logger.warning(f"Case {case_idx} 没有有效数据")
        
        return case_data
    
    def interactive_case_viewer(self, data: List[Dict], start_idx: int = 0):
        """
        交互式查看多个 case
        """
        current_idx = start_idx
        
        while True:
            print(f"\n{'='*50}")
            print(f"当前 Case: {current_idx} / {len(data) - 1}")
            print(f"{'='*50}")
            
            # 分析并显示当前 case
            case_data = self.analyze_and_plot_case(data, current_idx)
            
            if case_data.get('tokens'):
                print(f"\n📊 Token 数量: {len(case_data['tokens'])}")
                print(f"📈 平均熵: {np.mean(case_data['entropies']):.4f}")
                
                for phase_name, ranges in case_data['phase_ranges'].items():
                    if ranges:
                        total_tokens = sum(end - start for start, end in ranges)
                        print(f"  • {phase_name}: {len(ranges)} 个区块, 共 {total_tokens} tokens")
            
            # 用户输入
            print("\n操作: [n]下一个 [p]上一个 [g]跳转 [s]保存 [q]退出")
            user_input = input("请输入: ").strip().lower()
            
            if user_input == 'n':
                current_idx = min(current_idx + 1, len(data) - 1)
            elif user_input == 'p':
                current_idx = max(current_idx - 1, 0)
            elif user_input == 'g':
                try:
                    new_idx = int(input("输入 case 索引: "))
                    if 0 <= new_idx < len(data):
                        current_idx = new_idx
                    else:
                        print(f"索引超出范围 [0, {len(data) - 1}]")
                except ValueError:
                    print("请输入有效数字")
            elif user_input == 's':
                save_path = input("输入保存路径 (默认: entropy_case_{idx}.png): ").strip()
                if not save_path:
                    save_path = f"entropy_case_{current_idx}.png"
                self.plot_entropy_curve(case_data, current_idx, save_path)
            elif user_input == 'q':
                break
    
    def analyze_dataset(self, data: List[Dict], max_samples: int = None) -> Dict[str, PhaseEntropy]:
        """分析整个数据集"""
        from tqdm import tqdm
        
        phase_stats = {
            'think': PhaseEntropy('Think Phase (思考阶段)'),
            'tool_call': PhaseEntropy('Action Phase (工具调用)'),
            'tool_response': PhaseEntropy('Tool Response (工具响应)'),
            'answer': PhaseEntropy('Answer Phase (回答阶段)'),
        }
        
        samples = data[:max_samples] if max_samples else data
        logger.info(f"开始分析 {len(samples)} 个样本...")
        
        successful = 0
        for idx, sample in enumerate(tqdm(samples, desc="分析进度")):
            messages = sample.get('messages', [])
            if not messages:
                continue
            
            try:
                case_data = self.analyze_single_case(messages)
                
                if not case_data['tokens']:
                    continue
                
                entropies = case_data['entropies']
                phase_ranges = case_data['phase_ranges']
                
                has_data = False
                for phase_name, ranges in phase_ranges.items():
                    for start_idx, end_idx in ranges:
                        phase_entropies = entropies[start_idx:end_idx]
                        if phase_entropies:
                            phase_stats[phase_name].add_entropies(phase_entropies)
                            has_data = True
                
                if has_data:
                    successful += 1
                        
            except Exception as e:
                logger.warning(f"样本 {idx} 分析出错: {e}")
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
    
    parser = argparse.ArgumentParser(description='Agent 熵统计分析工具 - Case by Case 可视化')
    parser.add_argument('--input', '-i', type=str, help='输入 JSON 文件路径')
    parser.add_argument('--output', '-o', type=str, default='entropy_results.json', help='输出文件')
    parser.add_argument('--base-url', type=str, default='http://localhost:6001', help='sglang 服务地址')
    parser.add_argument('--limit', type=int, default=None, help='限制样本数量')
    parser.add_argument('--test', action='store_true', help='测试 API 连接')
    parser.add_argument('--diagnose', action='store_true', help='诊断数据文件中的 phase 分布')
    parser.add_argument('--debug', action='store_true', help='显示调试信息')
    
    # 新增的 case-by-case 可视化参数
    parser.add_argument('--case', '-c', type=int, default=0, help='要分析的单个 case 索引 (默认: 0)')
    parser.add_argument('--plot', '-p', action='store_true', help='绘制单个 case 的熵曲线')
    parser.add_argument('--save-plot', type=str, default=None, help='保存熵曲线图片的路径')
    parser.add_argument('--interactive', action='store_true', help='交互式查看多个 case')
    parser.add_argument('--show-tokens', action='store_true', help='在图表中显示 token 文本')
    parser.add_argument('--figsize', type=str, default='16,8', help='图片大小，格式: "宽,高"')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 解析 figsize
    try:
        figsize = tuple(map(int, args.figsize.split(',')))
    except:
        figsize = (16, 8)
    
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
        print("  诊断数据:       python script.py --diagnose -i data.json")
        print("  绘制单个 case:  python script.py -i data.json --plot --case 0")
        print("  保存图片:       python script.py -i data.json --plot --case 0 --save-plot output.png")
        print("  交互式查看:     python script.py -i data.json --interactive")
        print("  批量分析:       python script.py -i data.json --limit 10")
        return
    
    # 加载数据
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    analyzer = SGLangEntropyAnalyzer(args.base_url)
    
    # Case-by-case 可视化模式
    if args.plot:
        case_data = analyzer.analyze_and_plot_case(
            data, 
            case_idx=args.case,
            save_path=args.save_plot,
            figsize=figsize,
            show_tokens=args.show_tokens
        )
        
        if case_data.get('tokens'):
            print(f"\n📊 Case {args.case} 分析完成")
            print(f"   Token 数量: {len(case_data['tokens'])}")
            print(f"   平均熵: {np.mean(case_data['entropies']):.4f}")
            
            for phase_name, ranges in case_data['phase_ranges'].items():
                if ranges:
                    total_tokens = sum(end - start for start, end in ranges)
                    phase_entropies = []
                    for start_idx, end_idx in ranges:
                        phase_entropies.extend(case_data['entropies'][start_idx:end_idx])
                    if phase_entropies:
                        print(f"   • {phase_name}: {len(ranges)} 个区块, {total_tokens} tokens, avg_entropy={np.mean(phase_entropies):.4f}")
        return
    
    # 交互式模式
    if args.interactive:
        analyzer.interactive_case_viewer(data, start_idx=args.case)
        return
    
    # 批量分析模式
    phase_stats = analyzer.analyze_dataset(data, max_samples=args.limit)
    analyzer.print_results(phase_stats)
    analyzer.save_results(phase_stats, args.output)


if __name__ == '__main__':
    main()