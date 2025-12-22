"""
基于频域分析的大模型思维与工具调用模式研究
KV Cache Frequency Domain Analysis for LLM Thinking and Tool-Calling Patterns
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
warnings.filterwarnings('ignore')


# ============================================================================
# 1. 数据结构定义
# ============================================================================

@dataclass
class SemanticSegment:
    """语义片段数据结构"""
    segment_type: str  # 'think', 'tool_call', 'tool_response', 'answer'
    start_pos: int     # 在token序列中的起始位置
    end_pos: int       # 在token序列中的结束位置
    text: str          # 原始文本内容


@dataclass
class SampleKVData:
    """单个样本的KV数据"""
    sample_id: int
    segments: List[SemanticSegment]
    key_cache: torch.Tensor    # [num_layers, seq_len, num_heads, head_dim]
    value_cache: torch.Tensor  # [num_layers, seq_len, num_heads, head_dim]


# ============================================================================
# 2. 数据加载与解析
# ============================================================================

class DataLoader:
    """数据加载器"""
    
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.data = self._load_json()
    
    def _load_json(self) -> List[Dict]:
        """加载JSON数据"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_assistant_contents(self) -> List[str]:
        """
        提取所有assistant角色的内容，合并为完整对话
        返回每个样本的完整assistant内容
        """
        all_samples = []
        
        for item in self.data:
            messages = item.get('messages', [])
            assistant_contents = []
            
            for msg in messages:
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    assistant_contents.append(content)
            
            # 将同一样本的所有assistant内容合并
            full_content = ''.join(assistant_contents)
            all_samples.append(full_content)
        
        return all_samples
    
    def get_full_conversations(self) -> List[str]:
        """
        获取完整对话（用于模型推理）
        """
        all_conversations = []
        
        for item in self.data:
            messages = item.get('messages', [])
            # 构建完整对话文本
            conversation_parts = []
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                conversation_parts.append(f"<|{role}|>\n{content}")
            
            full_conversation = '\n'.join(conversation_parts)
            all_conversations.append(full_conversation)
        
        return all_conversations


# ============================================================================
# 3. 语义切片器
# ============================================================================

class SemanticSlicer:
    """语义切片器 - 基于特殊标记定位不同语义区域"""
    
    # 定义语义标记的正则模式
    PATTERNS = {
        'think': (r'<think>', r'</think>'),
        'tool_call': (r'<tool_call>', r'</tool_call>'),
        'tool_response': (r'<tool_response>', r'</tool_response>'),
        'answer': (r'<answer>', r'</answer>'),
    }
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def find_segments_in_text(self, text: str) -> List[Tuple[str, int, int, str]]:
        """
        在文本中找到所有语义片段的字符位置
        返回: [(segment_type, char_start, char_end, content), ...]
        """
        segments = []
        
        for seg_type, (open_tag, close_tag) in self.PATTERNS.items():
            # 构建匹配模式
            pattern = f'{open_tag}(.*?){close_tag}'
            
            for match in re.finditer(pattern, text, re.DOTALL):
                content = match.group(1)
                # 记录整个标签（包括开闭标签）的位置
                char_start = match.start()
                char_end = match.end()
                segments.append((seg_type, char_start, char_end, content))
        
        # 按位置排序
        segments.sort(key=lambda x: x[1])
        return segments
    
    def char_to_token_positions(
        self, 
        text: str, 
        char_segments: List[Tuple[str, int, int, str]],
        input_ids: torch.Tensor
    ) -> List[SemanticSegment]:
        """
        将字符位置转换为token位置
        """
        # 获取每个token对应的字符范围
        encoding = self.tokenizer(text, return_offsets_mapping=True, return_tensors='pt')
        offset_mapping = encoding['offset_mapping'][0]  # [seq_len, 2]
        
        token_segments = []
        
        for seg_type, char_start, char_end, content in char_segments:
            # 找到对应的token范围
            token_start = None
            token_end = None
            
            for idx, (tok_start, tok_end) in enumerate(offset_mapping.tolist()):
                if tok_start == tok_end == 0:  # 特殊token
                    continue
                
                # 找起始token
                if token_start is None and tok_end > char_start:
                    token_start = idx
                
                # 找结束token
                if tok_start < char_end:
                    token_end = idx + 1
            
            if token_start is not None and token_end is not None:
                token_segments.append(SemanticSegment(
                    segment_type=seg_type,
                    start_pos=token_start,
                    end_pos=token_end,
                    text=content
                ))
        
        return token_segments
    
    def slice_by_special_tokens(
        self, 
        input_ids: torch.Tensor, 
        text: str
    ) -> List[SemanticSegment]:
        """
        主切片方法：结合文本解析和token位置映射
        """
        # 在文本中找到语义片段
        char_segments = self.find_segments_in_text(text)
        
        if not char_segments:
            return []
        
        # 转换为token位置
        token_segments = self.char_to_token_positions(text, char_segments, input_ids)
        
        return token_segments


# ============================================================================
# 4. KV Cache 提取器
# ============================================================================

class KVCacheExtractor:
    """KV Cache 提取器"""
    
    def __init__(
        self, 
        model_name_or_path: str,
        device: str = 'cuda',
        torch_dtype: torch.dtype = torch.float16,
        target_layers: Optional[List[int]] = None
    ):
        """
        初始化模型和tokenizer
        
        Args:
            model_name_or_path: 模型路径或HuggingFace模型名
            device: 设备
            torch_dtype: 数据类型
            target_layers: 目标层索引列表，如果为None则自动选择中后层
        """
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
        
        # 获取模型配置
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.head_dim = self.model.config.hidden_size // self.num_heads
        
        # 设置目标层（中后层）
        if target_layers is None:
            # 默认选择后1/4到后1/8的层
            start_layer = int(self.num_layers * 0.75)
            end_layer = self.num_layers
            self.target_layers = list(range(start_layer, end_layer))
        else:
            self.target_layers = target_layers
        
        print(f"Model config: {self.num_layers} layers, {self.num_heads} heads, {self.head_dim} head_dim")
        print(f"Target layers for analysis: {self.target_layers}")
        
        # 初始化语义切片器
        self.slicer = SemanticSlicer(self.tokenizer)
    
    @torch.no_grad()
    def extract_kv_cache(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        提取单个文本的KV Cache
        
        Returns:
            input_ids: [seq_len]
            key_cache: [num_target_layers, seq_len, num_heads, head_dim]
            value_cache: [num_target_layers, seq_len, num_heads, head_dim]
        """
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors='pt',
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        input_ids = inputs['input_ids'][0]  # [seq_len]
        
        # 前向传播获取KV Cache
        outputs = self.model(
            **inputs,
            output_hidden_states=False,
            use_cache=True,
            return_dict=True
        )
        
        # 提取past_key_values
        # past_key_values是一个tuple，每层是(key, value)
        # key/value shape: [batch, num_heads, seq_len, head_dim]
        past_kv = outputs.past_key_values
        
        key_list = []
        value_list = []
        
        for layer_idx in self.target_layers:
            key = past_kv[layer_idx][0][0]  # [num_heads, seq_len, head_dim]
            value = past_kv[layer_idx][1][0]  # [num_heads, seq_len, head_dim]
            
            # 转换为 [seq_len, num_heads, head_dim]
            key = key.permute(1, 0, 2).cpu().float()
            value = value.permute(1, 0, 2).cpu().float()
            
            key_list.append(key)
            value_list.append(value)
        
        # Stack: [num_target_layers, seq_len, num_heads, head_dim]
        key_cache = torch.stack(key_list, dim=0)
        value_cache = torch.stack(value_list, dim=0)
        
        return input_ids.cpu(), key_cache, value_cache
    
    def process_sample(self, text: str, sample_id: int) -> Optional[SampleKVData]:
        """
        处理单个样本：提取KV Cache并进行语义切片
        """
        try:
            # 提取KV Cache
            input_ids, key_cache, value_cache = self.extract_kv_cache(text)
            
            # 语义切片
            segments = self.slicer.slice_by_special_tokens(input_ids, text)
            
            if not segments:
                print(f"Warning: Sample {sample_id} has no valid segments")
                return None
            
            return SampleKVData(
                sample_id=sample_id,
                segments=segments,
                key_cache=key_cache,
                value_cache=value_cache
            )
        
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            return None


# ============================================================================
# 5. 频域分析器
# ============================================================================

class FrequencyAnalyzer:
    """频域分析器"""
    
    def __init__(self, target_length: int = 128):
        """
        Args:
            target_length: 重采样的目标长度
        """
        self.target_length = target_length
    
    def dct_type2(self, x: np.ndarray) -> np.ndarray:
        """
        实现DCT-II变换
        
        对于长度为N的信号x，DCT-II定义为：
        y_t = α_t * Σ_{n=0}^{N-1} x_n * cos[π*t*(2n+1)/(2N)]
        
        其中 α_t = sqrt(1/N) if t=0, else sqrt(2/N)
        """
        N = len(x)
        y = np.zeros(N)
        
        for t in range(N):
            # 计算归一化因子
            if t == 0:
                alpha = np.sqrt(1.0 / N)
            else:
                alpha = np.sqrt(2.0 / N)
            
            # 计算DCT系数
            sum_val = 0.0
            for n in range(N):
                sum_val += x[n] * np.cos(np.pi * t * (2 * n + 1) / (2 * N))
            
            y[t] = alpha * sum_val
        
        return y
    
    def dct_fast(self, x: np.ndarray) -> np.ndarray:
        """
        使用FFT加速的DCT-II实现
        """
        from scipy.fftpack import dct
        return dct(x, type=2, norm='ortho')
    
    def resample_segment(self, segment: torch.Tensor) -> torch.Tensor:
        """
        将变长片段重采样到固定长度
        
        Args:
            segment: [seq_len, num_heads, head_dim] 或 [seq_len, hidden_dim]
        
        Returns:
            resampled: [target_length, ...]
        """
        original_shape = segment.shape
        seq_len = original_shape[0]
        
        if seq_len == self.target_length:
            return segment
        
        # 重塑用于插值: [1, features, seq_len]
        if len(original_shape) == 3:
            # [seq_len, num_heads, head_dim] -> [num_heads * head_dim, seq_len]
            segment_flat = segment.reshape(seq_len, -1).permute(1, 0)  # [features, seq_len]
            features = segment_flat.shape[0]
        else:
            segment_flat = segment.permute(1, 0)  # [features, seq_len]
            features = segment_flat.shape[0]
        
        # 添加batch维度: [1, features, seq_len]
        segment_3d = segment_flat.unsqueeze(0).float()
        
        # 线性插值到目标长度
        resampled = F.interpolate(
            segment_3d, 
            size=self.target_length, 
            mode='linear', 
            align_corners=False
        )
        
        # 恢复形状: [target_length, features]
        resampled = resampled.squeeze(0).permute(1, 0)
        
        if len(original_shape) == 3:
            # 恢复为 [target_length, num_heads, head_dim]
            num_heads = original_shape[1]
            head_dim = original_shape[2]
            resampled = resampled.reshape(self.target_length, num_heads, head_dim)
        
        return resampled
    
    def normalize_segment(self, segment: torch.Tensor) -> torch.Tensor:
        """
        Z-Score归一化
        
        Args:
            segment: [target_length, num_heads, head_dim]
        
        Returns:
            normalized: 同样形状
        """
        # 沿序列维度计算统计量
        mean = segment.mean(dim=0, keepdim=True)
        std = segment.std(dim=0, keepdim=True) + 1e-8
        
        normalized = (segment - mean) / std
        return normalized
    
    def compute_power_spectrum(self, segment: np.ndarray) -> np.ndarray:
        """
        计算功率谱
        
        Args:
            segment: [target_length, num_heads, head_dim]
        
        Returns:
            power_spectrum: [target_length, num_heads, head_dim]
        """
        # 对序列维度（第一维）进行DCT
        # 需要对每个(head, dim)位置分别计算
        target_len, num_heads, head_dim = segment.shape
        power_spectrum = np.zeros_like(segment)
        
        for h in range(num_heads):
            for d in range(head_dim):
                signal = segment[:, h, d]
                dct_coeffs = self.dct_fast(signal)
                power_spectrum[:, h, d] = dct_coeffs ** 2
        
        return power_spectrum
    
    def analyze_segment(
        self, 
        kv_cache: torch.Tensor,  # [num_layers, seq_len, num_heads, head_dim]
        start_pos: int,
        end_pos: int,
        use_key: bool = True
    ) -> np.ndarray:
        """
        分析单个语义片段
        
        Returns:
            power_spectrum: [target_length, num_layers, num_heads, head_dim]
        """
        # 切片
        if use_key:
            segment = kv_cache[:, start_pos:end_pos, :, :]  # [num_layers, seg_len, heads, dim]
        else:
            segment = kv_cache[:, start_pos:end_pos, :, :]
        
        num_layers = segment.shape[0]
        
        all_spectra = []
        
        for layer_idx in range(num_layers):
            layer_segment = segment[layer_idx]  # [seg_len, heads, dim]
            
            # 重采样
            resampled = self.resample_segment(layer_segment)  # [target_len, heads, dim]
            
            # 归一化
            normalized = self.normalize_segment(resampled)  # [target_len, heads, dim]
            
            # 计算功率谱
            power = self.compute_power_spectrum(normalized.numpy())  # [target_len, heads, dim]
            
            all_spectra.append(power)
        
        # Stack: [num_layers, target_len, heads, dim]
        stacked = np.stack(all_spectra, axis=0)
        
        # 转置为 [target_len, num_layers, heads, dim]
        return np.transpose(stacked, (1, 0, 2, 3))
    
    def aggregate_spectra(
        self, 
        spectra_list: List[np.ndarray]
    ) -> np.ndarray:
        """
        聚合多个样本的频谱
        
        Args:
            spectra_list: List of [target_length, num_layers, num_heads, head_dim]
        
        Returns:
            mean_spectrum: [target_length]  (全部维度平均后的曲线)
        """
        if not spectra_list:
            return np.zeros(self.target_length)
        
        # Stack所有样本
        stacked = np.stack(spectra_list, axis=0)  # [num_samples, target_len, layers, heads, dim]
        
        # 逐步平均
        # 先对所有样本平均
        mean_over_samples = stacked.mean(axis=0)  # [target_len, layers, heads, dim]
        
        # 再对所有其他维度平均，得到一条曲线
        mean_curve = mean_over_samples.mean(axis=(1, 2, 3))  # [target_len]
        
        return mean_curve


# ============================================================================
# 6. 主分析流程
# ============================================================================

class KVFrequencyAnalysisPipeline:
    """完整的分析流水线"""
    
    def __init__(
        self,
        model_name_or_path: str,
        json_path: str,
        target_length: int = 128,
        device: str = 'cuda',
        target_layers: Optional[List[int]] = None
    ):
        self.json_path = json_path
        self.target_length = target_length
        
        # 初始化组件
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
        
        # 存储结果
        self.results = {
            'think': {'key': [], 'value': []},
            'tool_call': {'key': [], 'value': []},
            'tool_response': {'key': [], 'value': []},
            'answer': {'key': [], 'value': []}
        }
    
    def run(self, max_samples: Optional[int] = None) -> Dict:
        """
        运行完整分析流程
        """
        # 获取所有assistant内容
        assistant_contents = self.data_loader.extract_assistant_contents()
        
        if max_samples:
            assistant_contents = assistant_contents[:max_samples]
        
        print(f"\nProcessing {len(assistant_contents)} samples...")
        
        for idx, text in enumerate(tqdm(assistant_contents, desc="Processing samples")):
            if not text.strip():
                continue
            
            # 提取KV Cache
            sample_data = self.kv_extractor.process_sample(text, idx)
            
            if sample_data is None:
                continue
            
            # 对每个语义片段进行频域分析
            for segment in sample_data.segments:
                seg_type = segment.segment_type
                
                if seg_type not in self.results:
                    continue
                
                # 分析Key Cache
                key_spectrum = self.freq_analyzer.analyze_segment(
                    sample_data.key_cache,
                    segment.start_pos,
                    segment.end_pos,
                    use_key=True
                )
                self.results[seg_type]['key'].append(key_spectrum)
                
                # 分析Value Cache
                value_spectrum = self.freq_analyzer.analyze_segment(
                    sample_data.value_cache,
                    segment.start_pos,
                    segment.end_pos,
                    use_key=False
                )
                self.results[seg_type]['value'].append(value_spectrum)
        
        # 聚合结果
        aggregated = self._aggregate_results()
        
        return aggregated
    
    def _aggregate_results(self) -> Dict:
        """聚合所有结果"""
        aggregated = {}
        
        for seg_type in self.results:
            aggregated[seg_type] = {}
            
            for cache_type in ['key', 'value']:
                spectra_list = self.results[seg_type][cache_type]
                
                if spectra_list:
                    mean_curve = self.freq_analyzer.aggregate_spectra(spectra_list)
                    aggregated[seg_type][cache_type] = mean_curve
                    print(f"{seg_type} - {cache_type}: {len(spectra_list)} samples aggregated")
                else:
                    aggregated[seg_type][cache_type] = np.zeros(self.target_length)
                    print(f"{seg_type} - {cache_type}: No samples found")
        
        return aggregated
    
    def plot_results(
        self, 
        aggregated: Dict, 
        save_path: Optional[str] = None,
        show_plot: bool = True
    ):
        """
        可视化分析结果
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = {
            'think': '#4a90e2',
            'tool_call': '#50c878',
            'tool_response': '#ff6b6b',
            'answer': '#9b59b6'
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
        
        # 左下：低频区域放大 (0-20)
        ax3 = axes[1, 0]
        low_freq_end = min(20, self.target_length)
        for seg_type, color in colors.items():
            if seg_type in aggregated and np.any(aggregated[seg_type]['key']):
                ax3.plot(freq_axis[:low_freq_end], 
                        aggregated[seg_type]['key'][:low_freq_end],
                        label=f'<{seg_type}>', color=color, linewidth=2, marker='o', markersize=4)
        ax3.set_xlabel('Frequency Index')
        ax3.set_ylabel('Power Spectral Density')
        ax3.set_title('Key Cache - Low Frequency Region (0-20)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 右下：高频区域放大
        ax4 = axes[1, 1]
        high_freq_start = max(0, self.target_length - 30)
        for seg_type, color in colors.items():
            if seg_type in aggregated and np.any(aggregated[seg_type]['key']):
                ax4.plot(freq_axis[high_freq_start:], 
                        aggregated[seg_type]['key'][high_freq_start:],
                        label=f'<{seg_type}>', color=color, linewidth=2, marker='o', markersize=4)
        ax4.set_xlabel('Frequency Index')
        ax4.set_ylabel('Power Spectral Density')
        ax4.set_title(f'Key Cache - High Frequency Region ({high_freq_start}-{self.target_length})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def compute_frequency_band_energy(self, aggregated: Dict) -> Dict:
        """
        计算各频段能量分布
        """
        # 定义频段
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
        """
        生成分析报告
        """
        report = []
        report.append("=" * 60)
        report.append("KV Cache 频域分析报告")
        report.append("=" * 60)
        report.append("")
        
        for seg_type in ['think', 'tool_call', 'tool_response', 'answer']:
            if seg_type not in aggregated:
                continue
            
            report.append(f"\n### <{seg_type}> 片段分析")
            report.append("-" * 40)
            
            # Key Cache 分析
            if seg_type in band_energy and 'key' in band_energy[seg_type]:
                report.append("\n**Key Cache 频段能量分布:**")
                for band_name, data in band_energy[seg_type]['key'].items():
                    report.append(f"  {band_name:12s}: {data['ratio']*100:6.2f}%")
            
            # Value Cache 分析
            if seg_type in band_energy and 'value' in band_energy[seg_type]:
                report.append("\n**Value Cache 频段能量分布:**")
                for band_name, data in band_energy[seg_type]['value'].items():
                    report.append(f"  {band_name:12s}: {data['ratio']*100:6.2f}%")
        
        report.append("\n" + "=" * 60)
        report.append("分析结论")
        report.append("=" * 60)
        
        # 自动生成一些观察结论
        report.append("""
1. **思维链 (<think>) 特征**: 
   - 观察低频能量占比，高占比表示连贯的语义推理
   
2. **工具调用 (<tool_call>) 特征**:
   - 观察是否存在特定频段的能量峰值，表示结构化语法模式
   
3. **工具响应 (<tool_response>) 特征**:
   - 对比与自然语言的相似度，高频噪声可能表示数据密集型内容
""")
        
        return "\n".join(report)


# ============================================================================
# 7. 使用示例与主函数
# ============================================================================

def main():
    """主函数示例"""
    
    # 配置参数
    config = {
        'model_name_or_path': 'meta-llama/Llama-3.2-3B-Instruct',  # 替换为你的模型
        'json_path': 'agent_conversations.json',  # 替换为你的数据路径
        'target_length': 128,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'target_layers': None,  # None表示自动选择中后层
        'max_samples': 100,  # 处理的最大样本数
    }
    
    print("=" * 60)
    print("KV Cache 频域分析实验")
    print("=" * 60)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # 创建流水线
    pipeline = KVFrequencyAnalysisPipeline(
        model_name_or_path=config['model_name_or_path'],
        json_path=config['json_path'],
        target_length=config['target_length'],
        device=config['device'],
        target_layers=config['target_layers']
    )
    
    # 运行分析
    aggregated = pipeline.run(max_samples=config['max_samples'])
    
    # 计算频段能量
    band_energy = pipeline.compute_frequency_band_energy(aggregated)
    
    # 生成报告
    report = pipeline.generate_report(aggregated, band_energy)
    print(report)
    
    # 保存报告
    with open('frequency_analysis_report.txt', 'w') as f:
        f.write(report)
    
    # 可视化
    pipeline.plot_results(aggregated, save_path='frequency_analysis.png')
    
    # 保存原始数据
    np.savez('frequency_analysis_data.npz', 
             **{f"{k}_{t}": v[t] for k, v in aggregated.items() for t in ['key', 'value']})
    
    print("\nAnalysis complete!")
    print("Outputs:")
    print("  - frequency_analysis_report.txt")
    print("  - frequency_analysis.png")
    print("  - frequency_analysis_data.npz")


# ============================================================================
# 8. 测试用的模拟数据生成器
# ============================================================================

def create_test_data(output_path: str = 'test_agent_data.json', num_samples: int = 10):
    """
    创建测试用的模拟数据
    """
    samples = []
    
    for i in range(num_samples):
        sample = {
            "id": i,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to tools."
                },
                {
                    "role": "user",
                    "content": f"What is the weather in city {i}?"
                },
                {
                    "role": "assistant",
                    "content": f"<think>Let me think about this request. The user wants to know the weather in city {i}. I should use the weather API to get this information.</think><tool_call>{{\"name\": \"get_weather\", \"arguments\": {{\"city\": \"city_{i}\"}}}}</tool_call>"
                },
                {
                    "role": "assistant",
                    "content": f"<tool_response>{{\"temperature\": {20 + i}, \"condition\": \"sunny\", \"humidity\": {50 + i}}}</tool_response>"
                },
                {
                    "role": "assistant",
                    "content": f"<think>I have received the weather information. The temperature is {20 + i} degrees and it's sunny.</think><answer>The weather in city {i} is sunny with a temperature of {20 + i}°C and humidity of {50 + i}%.</answer>"
                }
            ]
        }
        samples.append(sample)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"Test data saved to {output_path}")
    return output_path


# ============================================================================
# 9. 独立的分析函数（不需要GPU时使用）
# ============================================================================

def analyze_precomputed_kv_cache(
    kv_cache_path: str,
    segments_info_path: str,
    target_length: int = 128
):
    """
    分析预先提取的KV Cache数据
    用于已经有KV Cache保存的情况
    """
    # 加载预计算的数据
    kv_data = np.load(kv_cache_path)
    with open(segments_info_path, 'r') as f:
        segments_info = json.load(f)
    
    analyzer = FrequencyAnalyzer(target_length=target_length)
    
    results = {
        'think': [],
        'tool_call': [],
        'tool_response': [],
        'answer': []
    }
    
    for sample_info in segments_info:
        sample_id = sample_info['sample_id']
        key_cache = kv_data[f'key_{sample_id}']
        
        for seg in sample_info['segments']:
            seg_type = seg['type']
            start = seg['start']
            end = seg['end']
            
            spectrum = analyzer.analyze_segment(
                torch.tensor(key_cache),
                start, end,
                use_key=True
            )
            
            if seg_type in results:
                results[seg_type].append(spectrum)
    
    # 聚合
    aggregated = {}
    for seg_type, spectra_list in results.items():
        if spectra_list:
            aggregated[seg_type] = analyzer.aggregate_spectra(spectra_list)
        else:
            aggregated[seg_type] = np.zeros(target_length)
    
    return aggregated


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='KV Cache Frequency Analysis')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B-Instruct',
                        help='Model name or path')
    parser.add_argument('--data', type=str, default='agent_conversations.json',
                        help='Path to JSON data file')
    parser.add_argument('--target_length', type=int, default=128,
                        help='Target length for resampling')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--create_test_data', action='store_true',
                        help='Create test data file')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory')
    
    args = parser.parse_args()
    
    if args.create_test_data:
        create_test_data(args.data)
    else:
        # 检查数据文件是否存在
        import os
        if not os.path.exists(args.data):
            print(f"Data file {args.data} not found. Creating test data...")
            create_test_data(args.data)
        
        # 运行主分析
        pipeline = KVFrequencyAnalysisPipeline(
            model_name_or_path=args.model,
            json_path=args.data,
            target_length=args.target_length,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        aggregated = pipeline.run(max_samples=args.max_samples)
        band_energy = pipeline.compute_frequency_band_energy(aggregated)
        report = pipeline.generate_report(aggregated, band_energy)
        
        print(report)
        
        output_prefix = os.path.join(args.output_dir, 'frequency_analysis')
        pipeline.plot_results(aggregated, save_path=f'{output_prefix}.png')
        
        with open(f'{output_prefix}_report.txt', 'w') as f:
            f.write(report)