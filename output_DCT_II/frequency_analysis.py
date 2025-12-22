"""
基于频域分析的大模型思维与工具调用模式研究
完整实验代码实现
"""

import json
import re
import torch
import torch.nn.functional as F
import numpy as np
from scipy.fftpack import dct
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """实验配置参数"""
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"  # 模型名称
    target_layers: List[int] = None  # 目标提取层 (后初始化)
    target_length: int = 128  # 重采样目标长度
    batch_size: int = 1  # 推理批次大小
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./frequency_analysis_results"
    
    def __post_init__(self):
        if self.target_layers is None:
            # 默认提取中后层 (根据模型调整)
            self.target_layers = [24, 25, 26, 27, 28]


class SemanticSegment:
    """语义片段数据结构"""
    def __init__(self, segment_type: str, content: str, start_pos: int, end_pos: int):
        self.segment_type = segment_type  # 'think', 'tool_call', 'tool_response', 'answer'
        self.content = content
        self.start_pos = start_pos  # token起始位置
        self.end_pos = end_pos  # token结束位置
        self.attention_output: Optional[torch.Tensor] = None  # [Seq_Len, Hidden_Dim]
        self.ffn_output: Optional[torch.Tensor] = None  # [Seq_Len, Hidden_Dim]


class HiddenStateExtractor:
    """隐藏状态提取器 - 用于捕获指定层的Attention和FFN输出"""
    
    def __init__(self, model, target_layers: List[int]):
        self.model = model
        self.target_layers = target_layers
        self.attention_outputs = {}
        self.ffn_outputs = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向钩子以捕获中间层输出"""
        for layer_idx in self.target_layers:
            layer = self.model.model.layers[layer_idx]
            
            # 注册Attention输出钩子
            def get_attn_hook(idx):
                def hook(module, input, output):
                    # output[0] 是attention output
                    if isinstance(output, tuple):
                        self.attention_outputs[idx] = output[0].detach()
                    else:
                        self.attention_outputs[idx] = output.detach()
                return hook
            
            # 注册FFN输出钩子 (MLP层)
            def get_ffn_hook(idx):
                def hook(module, input, output):
                    self.ffn_outputs[idx] = output.detach()
                return hook
            
            # 对于LLaMA架构: self_attn 和 mlp
            attn_hook = layer.self_attn.register_forward_hook(get_attn_hook(layer_idx))
            ffn_hook = layer.mlp.register_forward_hook(get_ffn_hook(layer_idx))
            
            self.hooks.append(attn_hook)
            self.hooks.append(ffn_hook)
    
    def clear(self):
        """清空缓存"""
        self.attention_outputs.clear()
        self.ffn_outputs.clear()
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_outputs(self) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """获取捕获的输出"""
        return self.attention_outputs.copy(), self.ffn_outputs.copy()


class SemanticParser:
    """语义解析器 - 解析assistant消息中的语义片段"""
    
    # 定义语义标签的正则表达式
    PATTERNS = {
        'think': re.compile(r'<think>(.*?)</think>', re.DOTALL),
        'tool_call': re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL),
        'tool_response': re.compile(r'<tool_response>(.*?)</tool_response>', re.DOTALL),
        'answer': re.compile(r'<answer>(.*?)</answer>', re.DOTALL),
    }
    
    @classmethod
    def parse_message(cls, content: str) -> List[Tuple[str, str, int, int]]:
        """
        解析消息内容，提取语义片段
        返回: [(segment_type, content, char_start, char_end), ...]
        """
        segments = []
        
        for seg_type, pattern in cls.PATTERNS.items():
            for match in pattern.finditer(content):
                segments.append((
                    seg_type,
                    match.group(1),
                    match.start(),
                    match.end()
                ))
        
        # 按位置排序
        segments.sort(key=lambda x: x[2])
        return segments
    
    @classmethod
    def extract_assistant_content(cls, messages: List[Dict]) -> str:
        """从messages中提取所有assistant的内容"""
        assistant_contents = []
        for msg in messages:
            if msg.get('role') == 'assistant':
                assistant_contents.append(msg.get('content', ''))
        return ''.join(assistant_contents)


class TokenPositionMapper:
    """Token位置映射器 - 将字符位置映射到token位置"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def get_token_positions(self, text: str, char_positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        将字符位置映射到token位置
        Args:
            text: 完整文本
            char_positions: [(char_start, char_end), ...]
        Returns:
            [(token_start, token_end), ...]
        """
        # 对完整文本进行tokenize，获取offset_mapping
        encoding = self.tokenizer(
            text, 
            return_offsets_mapping=True, 
            add_special_tokens=False,
            return_tensors='pt'
        )
        
        offset_mapping = encoding['offset_mapping'][0].tolist()  # [(start, end), ...]
        
        token_positions = []
        for char_start, char_end in char_positions:
            token_start = None
            token_end = None
            
            for idx, (tok_start, tok_end) in enumerate(offset_mapping):
                # 找到包含char_start的token
                if token_start is None and tok_start <= char_start < tok_end:
                    token_start = idx
                elif token_start is None and char_start < tok_start:
                    token_start = idx
                
                # 找到包含char_end的token
                if tok_start < char_end <= tok_end:
                    token_end = idx + 1
                    break
                elif char_end <= tok_start:
                    token_end = idx
                    break
            
            if token_end is None:
                token_end = len(offset_mapping)
            if token_start is None:
                token_start = 0
                
            token_positions.append((token_start, token_end))
        
        return token_positions


class DCTAnalyzer:
    """离散余弦变换分析器"""
    
    def __init__(self, target_length: int = 128):
        self.target_length = target_length
    
    def resample(self, tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        将张量重采样到指定长度
        Args:
            tensor: [Seq_Len, Hidden_Dim]
            target_len: 目标长度
        Returns:
            [target_len, Hidden_Dim]
        """
        if tensor.size(0) == target_len:
            return tensor
        
        # [Seq_Len, Hidden_Dim] -> [1, Hidden_Dim, Seq_Len]
        tensor = tensor.permute(1, 0).unsqueeze(0)
        
        # 线性插值重采样
        resampled = F.interpolate(
            tensor.float(), 
            size=target_len, 
            mode='linear', 
            align_corners=False
        )
        
        # [1, Hidden_Dim, target_len] -> [target_len, Hidden_Dim]
        return resampled.squeeze(0).permute(1, 0)
    
    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Z-Score归一化
        Args:
            tensor: [Seq_Len, Hidden_Dim]
        Returns:
            归一化后的张量
        """
        # 沿序列方向归一化
        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True) + 1e-6
        return (tensor - mean) / std
    
    def compute_dct(self, tensor: torch.Tensor) -> np.ndarray:
        """
        计算DCT-II变换
        Args:
            tensor: [Seq_Len, Hidden_Dim]
        Returns:
            DCT系数 [Seq_Len, Hidden_Dim]
        """
        # 转换为numpy并计算DCT
        tensor_np = tensor.cpu().numpy()
        # 沿序列方向(axis=0)做DCT
        dct_result = dct(tensor_np, type=2, axis=0, norm='ortho')
        return dct_result
    
    def compute_power_spectrum(self, dct_coeffs: np.ndarray) -> np.ndarray:
        """
        计算功率谱
        Args:
            dct_coeffs: DCT系数 [Seq_Len, Hidden_Dim]
        Returns:
            功率谱 [Seq_Len, Hidden_Dim]
        """
        return dct_coeffs ** 2
    
    def process_segment(self, tensor: torch.Tensor) -> np.ndarray:
        """
        完整处理流程：重采样 -> 归一化 -> DCT -> 功率谱
        Args:
            tensor: [Seq_Len, Hidden_Dim]
        Returns:
            功率谱 [target_len, Hidden_Dim]
        """
        if tensor.size(0) < 2:
            logger.warning(f"Segment too short: {tensor.size(0)} tokens, skipping")
            return None
        
        # 1. 重采样
        resampled = self.resample(tensor, self.target_length)
        
        # 2. 归一化
        normalized = self.normalize(resampled)
        
        # 3. DCT变换
        dct_coeffs = self.compute_dct(normalized)
        
        # 4. 功率谱
        power_spectrum = self.compute_power_spectrum(dct_coeffs)
        
        return power_spectrum


class FrequencyAnalysisPipeline:
    """频域分析完整流水线"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 初始化组件
        logger.info(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        # 初始化提取器和分析器
        self.extractor = HiddenStateExtractor(self.model, config.target_layers)
        self.parser = SemanticParser()
        self.mapper = TokenPositionMapper(self.tokenizer)
        self.dct_analyzer = DCTAnalyzer(config.target_length)
        
        # 存储结果
        self.results = {
            'think': {'attention': [], 'ffn': []},
            'tool_call': {'attention': [], 'ffn': []},
            'tool_response': {'attention': [], 'ffn': []},
            'answer': {'attention': [], 'ffn': []}
        }
    
    def load_data(self, json_path: str) -> List[Dict]:
        """加载JSON数据"""
        logger.info(f"Loading data from: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples")
        return data
    
    def process_sample(self, sample: Dict) -> Dict[str, List[np.ndarray]]:
        """
        处理单个样本
        Returns:
            {segment_type: [power_spectrum, ...], ...}
        """
        messages = sample.get('messages', [])
        
        # 1. 提取assistant内容并解析语义片段
        assistant_content = self.parser.extract_assistant_content(messages)
        if not assistant_content:
            return {}
        
        segments = self.parser.parse_message(assistant_content)
        if not segments:
            return {}
        
        # 2. 构建完整输入 (包含所有messages)
        # 使用chat template构建输入
        full_input = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # 3. Tokenize
        inputs = self.tokenizer(
            full_input, 
            return_tensors='pt',
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        input_ids = inputs['input_ids'].to(self.device)
        
        # 4. 计算assistant内容在完整输入中的位置偏移
        # 找到assistant内容在full_input中的起始位置
        assistant_start_in_full = full_input.find(assistant_content)
        if assistant_start_in_full == -1:
            # 如果找不到完整匹配，尝试找第一个assistant消息
            for msg in messages:
                if msg.get('role') == 'assistant':
                    first_assistant = msg.get('content', '')
                    assistant_start_in_full = full_input.find(first_assistant)
                    break
        
        # 5. 模型推理
        self.extractor.clear()
        with torch.no_grad():
            _ = self.model(input_ids)
        
        attention_outputs, ffn_outputs = self.extractor.get_outputs()
        
        # 6. 提取各片段的特征
        sample_results = {}
        
        for seg_type, content, char_start, char_end in segments:
            # 调整字符位置到完整输入中的位置
            adjusted_char_start = assistant_start_in_full + char_start
            adjusted_char_end = assistant_start_in_full + char_end
            
            # 映射到token位置
            token_positions = self.mapper.get_token_positions(
                full_input, 
                [(adjusted_char_start, adjusted_char_end)]
            )
            
            if not token_positions:
                continue
            
            token_start, token_end = token_positions[0]
            
            # 确保位置有效
            seq_len = input_ids.size(1)
            token_start = max(0, min(token_start, seq_len - 1))
            token_end = max(token_start + 1, min(token_end, seq_len))
            
            if token_end - token_start < 2:
                continue
            
            # 7. 对每个目标层进行频域分析
            for layer_idx in self.config.target_layers:
                if layer_idx not in attention_outputs:
                    continue
                
                # 提取片段的hidden states
                # attention_outputs[layer_idx]: [1, Seq_Len, Hidden_Dim]
                attn_segment = attention_outputs[layer_idx][0, token_start:token_end, :]
                ffn_segment = ffn_outputs[layer_idx][0, token_start:token_end, :]
                
                # 8. DCT分析
                attn_power = self.dct_analyzer.process_segment(attn_segment)
                ffn_power = self.dct_analyzer.process_segment(ffn_segment)
                
                if attn_power is not None:
                    if seg_type not in sample_results:
                        sample_results[seg_type] = {'attention': [], 'ffn': []}
                    sample_results[seg_type]['attention'].append(attn_power)
                    sample_results[seg_type]['ffn'].append(ffn_power)
        
        return sample_results
    
    def run(self, json_path: str):
        """运行完整分析流水线"""
        data = self.load_data(json_path)
        
        for sample in tqdm(data, desc="Processing samples"):
            try:
                sample_results = self.process_sample(sample)
                
                # 聚合结果
                for seg_type, outputs in sample_results.items():
                    self.results[seg_type]['attention'].extend(outputs['attention'])
                    self.results[seg_type]['ffn'].extend(outputs['ffn'])
                    
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                continue
        
        # 清理hooks
        self.extractor.remove_hooks()
        
        return self.compute_statistics()
    
    def compute_statistics(self) -> Dict[str, Dict[str, np.ndarray]]:
        """计算各类片段的平均频谱"""
        statistics = {}
        
        for seg_type, outputs in self.results.items():
            if not outputs['attention']:
                continue
            
            # 计算平均频谱
            # attention spectra: [N, target_len, Hidden_Dim]
            attn_spectra = np.stack(outputs['attention'], axis=0)
            ffn_spectra = np.stack(outputs['ffn'], axis=0)
            
            # 平均: [target_len, Hidden_Dim] -> 再对Hidden_Dim平均 -> [target_len]
            avg_attn = attn_spectra.mean(axis=0).mean(axis=1)  # [target_len]
            avg_ffn = ffn_spectra.mean(axis=0).mean(axis=1)
            
            # 标准差用于置信区间
            std_attn = attn_spectra.mean(axis=2).std(axis=0)  # [target_len]
            std_ffn = ffn_spectra.mean(axis=2).std(axis=0)
            
            statistics[seg_type] = {
                'attention_mean': avg_attn,
                'attention_std': std_attn,
                'ffn_mean': avg_ffn,
                'ffn_std': std_ffn,
                'sample_count': len(outputs['attention'])
            }
            
            logger.info(f"{seg_type}: {len(outputs['attention'])} segments analyzed")
        
        return statistics
    
    def visualize(self, statistics: Dict, save_path: str = None):
        """可视化频谱对比"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        colors = {
            'think': '#4a90e2',
            'tool_call': '#50c878',
            'tool_response': '#ff6b6b',
            'answer': '#9b59b6'
        }
        
        freq_axis = np.arange(self.config.target_length)
        
        # Attention Output频谱
        ax1 = axes[0]
        for seg_type, data in statistics.items():
            mean = data['attention_mean']
            std = data['attention_std']
            ax1.plot(freq_axis, mean, label=f'<{seg_type}>', color=colors.get(seg_type, 'gray'), linewidth=2)
            ax1.fill_between(freq_axis, mean - std, mean + std, alpha=0.2, color=colors.get(seg_type, 'gray'))
        
        ax1.set_xlabel('Frequency Index', fontsize=12)
        ax1.set_ylabel('Power', fontsize=12)
        ax1.set_title('Attention Output - Average Power Spectrum', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # FFN Output频谱
        ax2 = axes[1]
        for seg_type, data in statistics.items():
            mean = data['ffn_mean']
            std = data['ffn_std']
            ax2.plot(freq_axis, mean, label=f'<{seg_type}>', color=colors.get(seg_type, 'gray'), linewidth=2)
            ax2.fill_between(freq_axis, mean - std, mean + std, alpha=0.2, color=colors.get(seg_type, 'gray'))
        
        ax2.set_xlabel('Frequency Index', fontsize=12)
        ax2.set_ylabel('Power', fontsize=12)
        ax2.set_title('FFN Output - Average Power Spectrum', fontsize=14)
        ax2.legend(loc='upper right')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to: {save_path}")
        
        plt.show()
        
        return fig


class FrequencyBandAnalyzer:
    """频段分析器 - 分析低频/中频/高频能量分布"""
    
    def __init__(self, target_length: int = 128):
        self.target_length = target_length
        # 定义频段边界
        self.bands = {
            'low': (0, target_length // 4),           # 0-25%: 低频
            'mid': (target_length // 4, target_length // 2),  # 25-50%: 中频
            'high': (target_length // 2, target_length)       # 50-100%: 高频
        }
    
    def compute_band_energy(self, power_spectrum: np.ndarray) -> Dict[str, float]:
        """
        计算各频段能量
        Args:
            power_spectrum: [target_len] 或 [target_len, Hidden_Dim]
        """
        if power_spectrum.ndim > 1:
            power_spectrum = power_spectrum.mean(axis=1)
        
        total_energy = power_spectrum.sum()
        band_energy = {}
        
        for band_name, (start, end) in self.bands.items():
            energy = power_spectrum[start:end].sum()
            band_energy[band_name] = energy / total_energy if total_energy > 0 else 0
        
        return band_energy
    
    def analyze_statistics(self, statistics: Dict) -> Dict[str, Dict[str, Dict[str, float]]]:
        """分析统计结果的频段能量分布"""
        band_analysis = {}
        
        for seg_type, data in statistics.items():
            band_analysis[seg_type] = {
                'attention': self.compute_band_energy(data['attention_mean']),
                'ffn': self.compute_band_energy(data['ffn_mean'])
            }
        
        return band_analysis


def create_sample_data():
    """创建示例数据用于测试"""
    sample_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant with tool access."},
                {"role": "user", "content": "What is the weather in Beijing today?"},
                {"role": "assistant", "content": "<think>The user wants to know the weather in Beijing. I need to use the weather tool to get this information.</think><tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Beijing\"}}</tool_call>"},
                {"role": "assistant", "content": "<tool_response>{\"temperature\": 25, \"condition\": \"sunny\", \"humidity\": 40}</tool_response>"},
                {"role": "assistant", "content": "<think>The weather data shows it's sunny with 25°C temperature. Let me formulate a nice response.</think><answer>The weather in Beijing today is sunny with a temperature of 25°C and 40% humidity. It's a great day to be outside!</answer>"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Calculate 15 * 23 + 47"},
                {"role": "assistant", "content": "<think>I need to calculate this step by step. First, 15 * 23 = 345. Then 345 + 47 = 392.</think><tool_call>{\"name\": \"calculator\", \"arguments\": {\"expression\": \"15 * 23 + 47\"}}</tool_call>"},
                {"role": "assistant", "content": "<tool_response>{\"result\": 392}</tool_response>"},
                {"role": "assistant", "content": "<think>The calculation confirms my mental math. The answer is 392.</think><answer>The result of 15 * 23 + 47 is 392.</answer>"}
            ]
        }
    ]
    
    return sample_data


def main():
    """主函数"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Frequency Analysis of LLM Hidden States')
    parser.add_argument('--data_path', type=str, default=None, help='Path to JSON data file')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Model name or path')
    parser.add_argument('--target_length', type=int, default=128, help='Target resampling length')
    parser.add_argument('--layers', type=str, default='24,25,26,27,28', help='Target layers (comma-separated)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--use_sample', action='store_true', help='Use sample data for testing')
    
    args = parser.parse_args()
    
    # 配置
    config = ExperimentConfig(
        model_name=args.model_name,
        target_layers=[int(l) for l in args.layers.split(',')],
        target_length=args.target_length,
        output_dir=args.output_dir
    )
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 如果使用示例数据
    if args.use_sample or args.data_path is None:
        sample_data = create_sample_data()
        sample_path = os.path.join(config.output_dir, 'sample_data.json')
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        args.data_path = sample_path
        logger.info("Using sample data for demonstration")
    
    # 初始化并运行分析
    pipeline = FrequencyAnalysisPipeline(config)
    statistics = pipeline.run(args.data_path)
    
    # 频段分析
    band_analyzer = FrequencyBandAnalyzer(config.target_length)
    band_analysis = band_analyzer.analyze_statistics(statistics)
    
    # 打印结果
    print("\n" + "="*60)
    print("FREQUENCY BAND ENERGY DISTRIBUTION")
    print("="*60)
    
    for seg_type, analysis in band_analysis.items():
        print(f"\n<{seg_type}>:")
        print(f"  Attention - Low: {analysis['attention']['low']:.3f}, "
              f"Mid: {analysis['attention']['mid']:.3f}, "
              f"High: {analysis['attention']['high']:.3f}")
        print(f"  FFN       - Low: {analysis['ffn']['low']:.3f}, "
              f"Mid: {analysis['ffn']['mid']:.3f}, "
              f"High: {analysis['ffn']['high']:.3f}")
    
    # 可视化
    fig_path = os.path.join(config.output_dir, 'frequency_spectrum_comparison.png')
    pipeline.visualize(statistics, save_path=fig_path)
    
    # 保存统计结果
    results_path = os.path.join(config.output_dir, 'statistics.json')
    serializable_stats = {}
    for seg_type, data in statistics.items():
        serializable_stats[seg_type] = {
            'attention_mean': data['attention_mean'].tolist(),
            'attention_std': data['attention_std'].tolist(),
            'ffn_mean': data['ffn_mean'].tolist(),
            'ffn_std': data['ffn_std'].tolist(),
            'sample_count': data['sample_count']
        }
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_stats, f, indent=2)
    
    logger.info(f"Results saved to: {config.output_dir}")
    
    return statistics, band_analysis


if __name__ == "__main__":
    main()