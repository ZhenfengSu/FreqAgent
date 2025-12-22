"""
基于频域分析的大模型思维与工具调用模式研究
修复版本：优化内存使用，支持大规模数据处理
"""

import json
import re
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import gc
import os

warnings.filterwarnings('ignore')


# ============================================================================
# 1. 数据结构定义
# ============================================================================

@dataclass
class SemanticSegment:
    """语义片段数据结构"""
    segment_type: str
    start_pos: int
    end_pos: int
    text: str


# ============================================================================
# 2. 数据加载与解析
# ============================================================================

class DataLoader:
    """数据加载器"""
    
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.data = self._load_json()
    
    def _load_json(self) -> List[Dict]:
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_assistant_contents(self) -> List[str]:
        all_samples = []
        for item in self.data:
            messages = item.get('messages', [])
            assistant_contents = []
            for msg in messages:
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    assistant_contents.append(content)
            full_content = ''.join(assistant_contents)
            all_samples.append(full_content)
        return all_samples


# ============================================================================
# 3. 语义切片器
# ============================================================================

class SemanticSlicer:
    """语义切片器"""
    
    PATTERNS = {
        'think': (r'<think>', r'</think>'),
        'tool_call': (r'<tool_call>', r'</tool_call>'),
        'tool_response': (r'<tool_response>', r'</tool_response>'),
    }
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def find_segments_in_text(self, text: str) -> List[Tuple[str, int, int, str]]:
        segments = []
        for seg_type, (open_tag, close_tag) in self.PATTERNS.items():
            pattern = f'{re.escape(open_tag)}(.*?){re.escape(close_tag)}'
            for match in re.finditer(pattern, text, re.DOTALL):
                content = match.group(1)
                char_start = match.start()
                char_end = match.end()
                segments.append((seg_type, char_start, char_end, content))
        segments.sort(key=lambda x: x[1])
        return segments
    
    def char_to_token_positions(
        self, 
        text: str, 
        char_segments: List[Tuple[str, int, int, str]],
        input_ids: torch.Tensor
    ) -> List[SemanticSegment]:
        encoding = self.tokenizer(text, return_offsets_mapping=True, return_tensors='pt')
        offset_mapping = encoding['offset_mapping'][0]
        
        token_segments = []
        for seg_type, char_start, char_end, content in char_segments:
            token_start = None
            token_end = None
            
            for idx, (tok_start, tok_end) in enumerate(offset_mapping.tolist()):
                if tok_start == tok_end == 0:
                    continue
                if token_start is None and tok_end > char_start:
                    token_start = idx
                if tok_start < char_end:
                    token_end = idx + 1
            
            if token_start is not None and token_end is not None and token_end > token_start:
                token_segments.append(SemanticSegment(
                    segment_type=seg_type,
                    start_pos=token_start,
                    end_pos=token_end,
                    text=content
                ))
        
        return token_segments
    
    def slice_by_special_tokens(self, input_ids: torch.Tensor, text: str) -> List[SemanticSegment]:
        char_segments = self.find_segments_in_text(text)
        if not char_segments:
            return []
        return self.char_to_token_positions(text, char_segments, input_ids)


# ============================================================================
# 4. 频域分析器（内存优化版）
# ============================================================================

class FrequencyAnalyzer:
    """频域分析器 - 内存优化版"""
    
    def __init__(self, target_length: int = 128):
        self.target_length = target_length
    
    def dct_fast(self, x: np.ndarray) -> np.ndarray:
        from scipy.fftpack import dct
        return dct(x, type=2, norm='ortho')
    
    def resample_segment(self, segment: torch.Tensor) -> torch.Tensor:
        """将变长片段重采样到固定长度"""
        original_shape = segment.shape
        seq_len = original_shape[0]
        
        if seq_len == self.target_length:
            return segment
        
        if seq_len < 2:
            # 序列太短，返回零张量
            return torch.zeros(self.target_length, *original_shape[1:])
        
        if len(original_shape) == 3:
            segment_flat = segment.reshape(seq_len, -1).permute(1, 0)
            features = segment_flat.shape[0]
        else:
            segment_flat = segment.permute(1, 0)
            features = segment_flat.shape[0]
        
        segment_3d = segment_flat.unsqueeze(0).float()
        
        resampled = F.interpolate(
            segment_3d, 
            size=self.target_length, 
            mode='linear', 
            align_corners=False
        )
        
        resampled = resampled.squeeze(0).permute(1, 0)
        
        if len(original_shape) == 3:
            num_heads = original_shape[1]
            head_dim = original_shape[2]
            resampled = resampled.reshape(self.target_length, num_heads, head_dim)
        
        return resampled
    
    def normalize_segment(self, segment: torch.Tensor) -> torch.Tensor:
        """Z-Score归一化"""
        mean = segment.mean(dim=0, keepdim=True)
        std = segment.std(dim=0, keepdim=True) + 1e-8
        return (segment - mean) / std
    
    def compute_power_spectrum_reduced(self, segment: np.ndarray) -> np.ndarray:
        """
        计算功率谱 - 直接返回降维后的结果
        
        Args:
            segment: [target_length, num_heads, head_dim]
        
        Returns:
            power_curve: [target_length] - 已经对 heads 和 dim 取平均
        """
        target_len, num_heads, head_dim = segment.shape
        
        # 累积功率谱，直接求和而不是存储完整矩阵
        power_sum = np.zeros(target_len)
        count = num_heads * head_dim
        
        for h in range(num_heads):
            for d in range(head_dim):
                signal = segment[:, h, d]
                dct_coeffs = self.dct_fast(signal)
                power_sum += dct_coeffs ** 2
        
        return power_sum / count
    
    def analyze_segment_reduced(
        self, 
        kv_cache: torch.Tensor,  # [num_layers, seq_len, num_heads, head_dim]
        start_pos: int,
        end_pos: int
    ) -> Optional[np.ndarray]:
        """
        分析单个语义片段 - 返回降维后的结果
        
        Returns:
            power_curve: [target_length] 或 None（如果片段无效）
        """
        seq_len = kv_cache.shape[1]
        
        # 边界检查
        if start_pos >= seq_len or end_pos > seq_len or start_pos >= end_pos:
            return None
        
        if end_pos - start_pos < 2:
            return None
        
        segment = kv_cache[:, start_pos:end_pos, :, :]
        num_layers = segment.shape[0]
        
        # 累积各层的功率谱
        layer_power_sum = np.zeros(self.target_length)
        
        for layer_idx in range(num_layers):
            layer_segment = segment[layer_idx]  # [seg_len, heads, dim]
            
            # 重采样
            resampled = self.resample_segment(layer_segment)
            
            # 归一化
            normalized = self.normalize_segment(resampled)
            
            # 计算功率谱（已降维）
            power_curve = self.compute_power_spectrum_reduced(normalized.numpy())
            
            layer_power_sum += power_curve
        
        # 对层取平均
        return layer_power_sum / num_layers


# ============================================================================
# 5. 在线统计累积器
# ============================================================================

class OnlineStatistics:
    """
    在线统计累积器 - 使用 Welford's 算法计算均值和方差
    无需存储所有样本，节省内存
    """
    
    def __init__(self, shape: Tuple[int, ...]):
        self.n = 0
        self.mean = np.zeros(shape)
        self.M2 = np.zeros(shape)  # 用于计算方差
    
    def update(self, x: np.ndarray):
        """添加新样本"""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
    
    def get_mean(self) -> np.ndarray:
        """获取当前均值"""
        return self.mean
    
    def get_variance(self) -> np.ndarray:
        """获取当前方差"""
        if self.n < 2:
            return np.zeros_like(self.mean)
        return self.M2 / (self.n - 1)
    
    def get_std(self) -> np.ndarray:
        """获取当前标准差"""
        return np.sqrt(self.get_variance())
    
    def get_count(self) -> int:
        """获取样本数量"""
        return self.n


# ============================================================================
# 6. KV Cache 提取器（内存优化版）
# ============================================================================

class KVCacheExtractor:
    """KV Cache 提取器 - 内存优化版"""
    
    def __init__(
        self, 
        model_name_or_path: str,
        device: str = 'cuda',
        torch_dtype: torch.dtype = torch.float16,
        target_layers: Optional[List[int]] = None
    ):
        self.device = device
        self.torch_dtype = torch_dtype
        
        print(f"Loading tokenizer from {model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading model from {model_name_or_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map='auto',
            trust_remote_code=True
        )
        self.model.eval()
        
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.head_dim = self.model.config.hidden_size // self.num_heads
        
        if target_layers is None:
            start_layer = int(self.num_layers * 0.75)
            self.target_layers = list(range(start_layer, self.num_layers))
        else:
            self.target_layers = target_layers
        
        print(f"Model: {self.num_layers} layers, {self.num_heads} heads, {self.head_dim} head_dim")
        print(f"Target layers: {self.target_layers}")
        
        self.slicer = SemanticSlicer(self.tokenizer)
    
    @torch.no_grad()
    def extract_and_analyze(
        self, 
        text: str, 
        freq_analyzer: FrequencyAnalyzer
    ) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        """
        提取 KV Cache 并立即进行分析，不保存原始 KV Cache
        
        Returns:
            {segment_type: {'key': power_curve, 'value': power_curve}}
            或 None（如果处理失败）
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                text, 
                return_tensors='pt',
                truncation=True,
                max_length=32768
            ).to(self.device)
            
            input_ids = inputs['input_ids'][0]
            
            # 前向传播
            outputs = self.model(
                **inputs,
                output_hidden_states=False,
                use_cache=True,
                return_dict=True
            )
            
            past_kv = outputs.past_key_values
            
            # 只提取目标层的 KV Cache
            key_list = []
            value_list = []
            
            for layer_idx in self.target_layers:
                key = past_kv[layer_idx][0][0].permute(1, 0, 2).cpu().float()
                value = past_kv[layer_idx][1][0].permute(1, 0, 2).cpu().float()
                key_list.append(key)
                value_list.append(value)
            
            key_cache = torch.stack(key_list, dim=0)
            value_cache = torch.stack(value_list, dim=0)
            
            # 立即释放 GPU 内存
            del outputs, past_kv, inputs
            torch.cuda.empty_cache()
            
            # 语义切片
            segments = self.slicer.slice_by_special_tokens(input_ids.cpu(), text)
            
            if not segments:
                del key_cache, value_cache
                return None
            
            # 分析每个片段
            results = {}
            for segment in segments:
                seg_type = segment.segment_type
                
                # 分析 Key Cache
                key_power = freq_analyzer.analyze_segment_reduced(
                    key_cache, segment.start_pos, segment.end_pos
                )
                
                # 分析 Value Cache
                value_power = freq_analyzer.analyze_segment_reduced(
                    value_cache, segment.start_pos, segment.end_pos
                )
                
                if key_power is not None and value_power is not None:
                    if seg_type not in results:
                        results[seg_type] = {'key': [], 'value': []}
                    results[seg_type]['key'].append(key_power)
                    results[seg_type]['value'].append(value_power)
            
            # 释放内存
            del key_cache, value_cache
            
            return results
            
        except Exception as e:
            print(f"Error: {e}")
            torch.cuda.empty_cache()
            return None


# ============================================================================
# 7. 主分析流程（内存优化版）
# ============================================================================

class KVFrequencyAnalysisPipeline:
    """完整的分析流水线 - 内存优化版"""
    
    def __init__(
        self,
        model_name_or_path: str,
        json_path: str,
        target_length: int = 128,
        device: str = 'cuda',
        target_layers: Optional[List[int]] = None,
        checkpoint_interval: int = 100,
        checkpoint_dir: str = './checkpoints'
    ):
        self.json_path = json_path
        self.target_length = target_length
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print("Initializing data loader...")
        self.data_loader = DataLoader(json_path)
        
        print("Initializing KV extractor...")
        self.kv_extractor = KVCacheExtractor(
            model_name_or_path=model_name_or_path,
            device=device,
            target_layers=target_layers
        )
        
        print("Initializing frequency analyzer...")
        self.freq_analyzer = FrequencyAnalyzer(target_length=target_length)
        
        # 使用在线统计累积器替代列表存储
        self.stats = {
            'think': {'key': OnlineStatistics((target_length,)), 
                     'value': OnlineStatistics((target_length,))},
            'tool_call': {'key': OnlineStatistics((target_length,)), 
                         'value': OnlineStatistics((target_length,))},
            'tool_response': {'key': OnlineStatistics((target_length,)), 
                             'value': OnlineStatistics((target_length,))},
        }
        
        self.processed_count = 0
        self.segment_counts = {k: 0 for k in self.stats.keys()}
    
    def run(
        self, 
        max_samples: Optional[int] = None,
        resume_from: Optional[int] = None
    ) -> Dict:
        """运行分析"""
        assistant_contents = self.data_loader.extract_assistant_contents()
        
        if max_samples:
            assistant_contents = assistant_contents[:max_samples]
        
        start_idx = resume_from if resume_from else 0
        
        print(f"\nProcessing {len(assistant_contents)} samples (starting from {start_idx})...")
        
        for idx in tqdm(range(start_idx, len(assistant_contents)), desc="Processing"):
            text = assistant_contents[idx]
            
            if not text.strip():
                continue
            
            # 提取并分析
            results = self.kv_extractor.extract_and_analyze(text, self.freq_analyzer)
            
            if results is None:
                continue
            
            # 更新在线统计
            for seg_type, cache_data in results.items():
                if seg_type not in self.stats:
                    continue
                
                for cache_type in ['key', 'value']:
                    for power_curve in cache_data.get(cache_type, []):
                        self.stats[seg_type][cache_type].update(power_curve)
                        if cache_type == 'key':
                            self.segment_counts[seg_type] += 1
            
            self.processed_count += 1
            
            # 定期保存 checkpoint
            if self.processed_count % self.checkpoint_interval == 0:
                self._save_checkpoint(idx)
                gc.collect()
                torch.cuda.empty_cache()
        
        # 最终保存
        self._save_checkpoint(len(assistant_contents) - 1, final=True)
        
        return self._get_aggregated_results()
    
    def _save_checkpoint(self, current_idx: int, final: bool = False):
        """保存检查点"""
        checkpoint = {
            'current_idx': current_idx,
            'processed_count': self.processed_count,
            'segment_counts': self.segment_counts,
            'stats': {}
        }
        
        for seg_type, cache_stats in self.stats.items():
            checkpoint['stats'][seg_type] = {}
            for cache_type, online_stat in cache_stats.items():
                checkpoint['stats'][seg_type][cache_type] = {
                    'n': online_stat.n,
                    'mean': online_stat.mean,
                    'M2': online_stat.M2
                }
        
        suffix = 'final' if final else f'idx_{current_idx}'
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{suffix}.npz')
        
        # 使用 numpy 保存
        save_dict = {
            'current_idx': current_idx,
            'processed_count': self.processed_count,
        }
        
        for seg_type in self.stats:
            for cache_type in ['key', 'value']:
                prefix = f'{seg_type}_{cache_type}'
                stat = self.stats[seg_type][cache_type]
                save_dict[f'{prefix}_n'] = stat.n
                save_dict[f'{prefix}_mean'] = stat.mean
                save_dict[f'{prefix}_M2'] = stat.M2
        
        for seg_type, count in self.segment_counts.items():
            save_dict[f'count_{seg_type}'] = count
        
        np.savez(checkpoint_path, **save_dict)
        print(f"\nCheckpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """加载检查点，返回恢复的索引"""
        data = np.load(checkpoint_path)
        
        self.processed_count = int(data['processed_count'])
        current_idx = int(data['current_idx'])
        
        for seg_type in self.stats:
            for cache_type in ['key', 'value']:
                prefix = f'{seg_type}_{cache_type}'
                self.stats[seg_type][cache_type].n = int(data[f'{prefix}_n'])
                self.stats[seg_type][cache_type].mean = data[f'{prefix}_mean']
                self.stats[seg_type][cache_type].M2 = data[f'{prefix}_M2']
        
        for seg_type in self.segment_counts:
            self.segment_counts[seg_type] = int(data[f'count_{seg_type}'])
        
        print(f"Loaded checkpoint from index {current_idx}")
        print(f"Processed samples: {self.processed_count}")
        print(f"Segment counts: {self.segment_counts}")
        
        return current_idx + 1
    
    def _get_aggregated_results(self) -> Dict:
        """获取聚合结果"""
        aggregated = {}
        
        print("\n" + "=" * 50)
        print("Final Statistics:")
        print("=" * 50)
        
        for seg_type in self.stats:
            aggregated[seg_type] = {}
            
            for cache_type in ['key', 'value']:
                stat = self.stats[seg_type][cache_type]
                
                if stat.n > 0:
                    aggregated[seg_type][cache_type] = stat.get_mean()
                    print(f"{seg_type} - {cache_type}: {stat.n} samples")
                else:
                    aggregated[seg_type][cache_type] = np.zeros(self.target_length)
                    print(f"{seg_type} - {cache_type}: No samples")
        
        return aggregated
    
    def plot_results(
        self, 
        aggregated: Dict, 
        save_path: Optional[str] = None,
        show_plot: bool = True
    ):
        """可视化结果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = {
            'think': '#4a90e2',
            'tool_call': '#50c878',
            'tool_response': '#ff6b6b',
        }
        
        freq_axis = np.arange(self.target_length)
        
        # 左上：Key Cache 全频谱
        ax1 = axes[0, 0]
        for seg_type, color in colors.items():
            if seg_type in aggregated and np.any(aggregated[seg_type]['key']):
                ax1.plot(freq_axis, aggregated[seg_type]['key'], 
                        label=f'<{seg_type}>', color=color, linewidth=2)
        ax1.set_xlabel('Frequency Index')
        ax1.set_ylabel('Power Spectral Density')
        ax1.set_title('Key Cache - Full Power Spectrum')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 右上：Value Cache 全频谱
        ax2 = axes[0, 1]
        for seg_type, color in colors.items():
            if seg_type in aggregated and np.any(aggregated[seg_type]['value']):
                ax2.plot(freq_axis, aggregated[seg_type]['value'], 
                        label=f'<{seg_type}>', color=color, linewidth=2)
        ax2.set_xlabel('Frequency Index')
        ax2.set_ylabel('Power Spectral Density')
        ax2.set_title('Value Cache - Full Power Spectrum')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 左下：低频区域
        ax3 = axes[1, 0]
        low_freq_end = min(20, self.target_length)
        for seg_type, color in colors.items():
            if seg_type in aggregated and np.any(aggregated[seg_type]['key']):
                ax3.plot(freq_axis[:low_freq_end], 
                        aggregated[seg_type]['key'][:low_freq_end],
                        label=f'<{seg_type}>', color=color, linewidth=2, marker='o', markersize=4)
        ax3.set_xlabel('Frequency Index')
        ax3.set_ylabel('Power Spectral Density')
        ax3.set_title('Key Cache - Low Frequency (0-20)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 右下：频段能量对比
        ax4 = axes[1, 1]
        bands = {
            'DC': (0, 1),
            'VLow': (1, 5),
            'Low': (5, 20),
            'Mid': (20, 50),
            'High': (50, self.target_length)
        }
        
        x = np.arange(len(bands))
        width = 0.2
        
        for i, (seg_type, color) in enumerate(colors.items()):
            if seg_type in aggregated:
                spectrum = aggregated[seg_type]['key']
                total = np.sum(spectrum) + 1e-10
                ratios = []
                for band_name, (start, end) in bands.items():
                    end = min(end, len(spectrum))
                    ratios.append(np.sum(spectrum[start:end]) / total)
                ax4.bar(x + i*width, ratios, width, label=f'<{seg_type}>', color=color)
        
        ax4.set_xlabel('Frequency Band')
        ax4.set_ylabel('Energy Ratio')
        ax4.set_title('Key Cache - Band Energy Distribution')
        ax4.set_xticks(x + width * 1.5)
        ax4.set_xticklabels(list(bands.keys()))
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def compute_frequency_band_energy(self, aggregated: Dict) -> Dict:
        """计算频段能量"""
        bands = {
            'DC': (0, 1),
            'Very Low': (1, 5),
            'Low': (5, 20),
            'Mid': (20, 50),
            'High': (50, 100),
            'Very High': (100, self.target_length)
        }
        
        band_energy = {}
        
        for seg_type in aggregated:
            band_energy[seg_type] = {}
            
            for cache_type in ['key', 'value']:
                spectrum = aggregated[seg_type][cache_type]
                
                if not np.any(spectrum):
                    continue
                
                total_energy = np.sum(spectrum)
                band_energy[seg_type][cache_type] = {}
                
                for band_name, (start, end) in bands.items():
                    end = min(end, len(spectrum))
                    band_power = np.sum(spectrum[start:end])
                    band_ratio = band_power / total_energy if total_energy > 0 else 0
                    band_energy[seg_type][cache_type][band_name] = {
                        'power': float(band_power),
                        'ratio': float(band_ratio)
                    }
        
        return band_energy
    
    def generate_report(self, aggregated: Dict, band_energy: Dict) -> str:
        """生成报告"""
        report = []
        report.append("=" * 60)
        report.append("KV Cache 频域分析报告")
        report.append("=" * 60)
        report.append("")
        
        report.append("### 数据统计")
        report.append("-" * 40)
        report.append(f"总处理样本数: {self.processed_count}")
        for seg_type, count in self.segment_counts.items():
            report.append(f"  {seg_type}: {count} 个片段")
        
        for seg_type in ['think', 'tool_call', 'tool_response']:
            if seg_type not in aggregated:
                continue
            
            report.append(f"\n### <{seg_type}> 片段分析")
            report.append("-" * 40)
            
            if seg_type in band_energy and 'key' in band_energy[seg_type]:
                report.append("\n**Key Cache 频段能量分布:**")
                for band_name, data in band_energy[seg_type]['key'].items():
                    report.append(f"  {band_name:12s}: {data['ratio']*100:6.2f}%")
            
            if seg_type in band_energy and 'value' in band_energy[seg_type]:
                report.append("\n**Value Cache 频段能量分布:**")
                for band_name, data in band_energy[seg_type]['value'].items():
                    report.append(f"  {band_name:12s}: {data['ratio']*100:6.2f}%")
        
        return "\n".join(report)


# ============================================================================
# 8. 测试数据生成
# ============================================================================

def create_test_data(output_path: str = 'test_agent_data.json', num_samples: int = 10):
    """创建测试数据"""
    samples = []
    
    for i in range(num_samples):
        sample = {
            "id": i,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"What is the weather in city {i}?"},
                {"role": "assistant", "content": f"<think>The user wants weather info for city {i}. Let me use the weather API.</think><tool_call>{{\"name\": \"get_weather\", \"city\": \"city_{i}\"}}</tool_call>"},
                {"role": "assistant", "content": f"<tool_response>{{\"temp\": {20+i}, \"condition\": \"sunny\"}}</tool_response>"},
                {"role": "assistant", "content": f"<think>Got the weather data.</think><answer>Weather in city {i}: {20+i}°C and sunny.</answer>"}
            ]
        }
        samples.append(sample)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)
    
    print(f"Test data saved to {output_path}")


# ============================================================================
# 9. 主函数
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='KV Cache Frequency Analysis (Memory Optimized)')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--data', type=str, default='agent_conversations.json')
    parser.add_argument('--target_length', type=int, default=128)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--checkpoint_interval', type=int, default=100)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--create_test_data', action='store_true')
    parser.add_argument('--output_dir', type=str, default='.')
    
    args = parser.parse_args()
    
    if args.create_test_data:
        create_test_data(args.data)
        return
    
    if not os.path.exists(args.data):
        print(f"Data file not found. Creating test data...")
        create_test_data(args.data)
    
    # 创建流水线
    pipeline = KVFrequencyAnalysisPipeline(
        model_name_or_path=args.model,
        json_path=args.data,
        target_length=args.target_length,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # 恢复检查点
    resume_from = None
    if args.resume and os.path.exists(args.resume):
        resume_from = pipeline.load_checkpoint(args.resume)
    
    # 运行分析
    aggregated = pipeline.run(max_samples=args.max_samples, resume_from=resume_from)
    
    # 生成报告
    band_energy = pipeline.compute_frequency_band_energy(aggregated)
    report = pipeline.generate_report(aggregated, band_energy)
    print(report)
    
    # 保存结果
    output_prefix = os.path.join(args.output_dir, 'frequency_analysis')
    
    with open(f'{output_prefix}_report.txt', 'w') as f:
        f.write(report)
    
    pipeline.plot_results(aggregated, save_path=f'{output_prefix}.png')
    
    # 保存数据
    save_dict = {f"{k}_{t}": v[t] for k, v in aggregated.items() for t in ['key', 'value']}
    np.savez(f'{output_prefix}_data.npz', **save_dict)
    
    print(f"\nAnalysis complete!")
    print(f"Outputs saved to {args.output_dir}/")


if __name__ == '__main__':
    main()