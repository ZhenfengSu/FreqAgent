"""
Agent 轨迹频谱分析框架 (Spectral Analysis Framework for Agents)
分析 Think, Tool Call, Tool Response 的频谱特性
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, fft, fftfreq
from scipy.signal import welch
from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SegmentType(Enum):
    THINK = "think"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    SYSTEM = "system"
    USER = "user"
    OTHER = "other"

@dataclass
class Segment:
    """表示一个轨迹片段"""
    type: SegmentType
    content: str
    start_idx: int
    end_idx: int
    token_count: int

@dataclass
class SpectralMetrics:
    """频谱分析指标"""
    spectral_entropy: float
    high_freq_ratio: float
    energy_concentration: float
    dominant_frequency: float
    total_energy: float

class AgentTrajectoryAnalyzer:
    """Agent 轨迹分析器"""
    
    def __init__(self, json_data: dict):
        self.data = json_data
        self.messages = json_data.get('messages', [])
        self.segments: List[Segment] = []
        self.token_embeddings: Optional[np.ndarray] = None
        
    def parse_trajectory(self) -> List[Segment]:
        """解析轨迹，分割成不同类型的片段"""
        segments = []
        current_idx = 0
        
        for msg in self.messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'system':
                segments.append(Segment(
                    type=SegmentType.SYSTEM,
                    content=content,
                    start_idx=current_idx,
                    end_idx=current_idx + len(content),
                    token_count=self._estimate_tokens(content)
                ))
                current_idx += len(content)
                
            elif role == 'assistant':
                # 解析 assistant 消息中的 think, tool_call
                parsed = self._parse_assistant_content(content, current_idx)
                segments.extend(parsed)
                current_idx += len(content)
                
            elif role == 'user':
                # 检查是否是 tool_response
                if '<tool_response>' in content:
                    segments.append(Segment(
                        type=SegmentType.TOOL_RESPONSE,
                        content=content,
                        start_idx=current_idx,
                        end_idx=current_idx + len(content),
                        token_count=self._estimate_tokens(content)
                    ))
                else:
                    segments.append(Segment(
                        type=SegmentType.USER,
                        content=content,
                        start_idx=current_idx,
                        end_idx=current_idx + len(content),
                        token_count=self._estimate_tokens(content)
                    ))
                current_idx += len(content)
        
        self.segments = segments
        return segments
    
    def _parse_assistant_content(self, content: str, start_idx: int) -> List[Segment]:
        """解析 assistant 消息，分离 think 和 tool_call"""
        segments = []
        current_pos = 0
        
        # 匹配 <think>...</think>
        think_pattern = r'<think>(.*?)</think>'
        # 匹配 <tool_call>...</tool_call>
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        # 匹配 <answer>...</answer>
        answer_pattern = r'<answer>(.*?)</answer>'
        
        # 找到所有 think 块
        for match in re.finditer(think_pattern, content, re.DOTALL):
            segments.append(Segment(
                type=SegmentType.THINK,
                content=match.group(1),
                start_idx=start_idx + match.start(),
                end_idx=start_idx + match.end(),
                token_count=self._estimate_tokens(match.group(1))
            ))
        
        # 找到所有 tool_call 块
        for match in re.finditer(tool_call_pattern, content, re.DOTALL):
            segments.append(Segment(
                type=SegmentType.TOOL_CALL,
                content=match.group(1),
                start_idx=start_idx + match.start(),
                end_idx=start_idx + match.end(),
                token_count=self._estimate_tokens(match.group(1))
            ))
        
        return segments
    
    def _estimate_tokens(self, text: str) -> int:
        """估算 token 数量（简化版，约 4 字符/token）"""
        return max(1, len(text) // 4)
    
    def generate_synthetic_embeddings(self, dim: int = 256) -> np.ndarray:
        """
        生成模拟的 embedding 序列
        在实际应用中，这应该从模型的 hidden states 或 KV cache 中提取
        """
        if not self.segments:
            self.parse_trajectory()
        
        all_embeddings = []
        
        for segment in self.segments:
            n_tokens = segment.token_count
            
            if segment.type == SegmentType.THINK:
                # Think: 高频噪声 + 低频基准
                low_freq = np.sin(np.linspace(0, 2*np.pi, n_tokens))[:, None] * np.random.randn(dim)
                high_freq = np.random.randn(n_tokens, dim) * 0.5
                embeddings = low_freq + high_freq + np.random.randn(dim) * 0.1  # DC component
                
            elif segment.type == SegmentType.TOOL_CALL:
                # Tool Call: 脉冲信号，结构化
                base = np.zeros((n_tokens, dim))
                # 添加脉冲特性
                for i in range(n_tokens):
                    base[i] = np.sin(np.arange(dim) * (i + 1) * 0.1) * 2
                embeddings = base + np.random.randn(n_tokens, dim) * 0.1
                
            elif segment.type == SegmentType.TOOL_RESPONSE:
                # Tool Response: 混合信号
                structured = np.tile(np.sin(np.linspace(0, 4*np.pi, dim)), (n_tokens, 1))
                noise = np.random.randn(n_tokens, dim) * 0.8
                embeddings = structured + noise
                
            else:
                # 其他类型
                embeddings = np.random.randn(n_tokens, dim) * 0.3
            
            all_embeddings.append(embeddings)
        
        self.token_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
        return self.token_embeddings


class SpectralAnalyzer:
    """频谱分析器"""
    
    def __init__(self, cutoff_ratio: float = 0.2):
        self.cutoff_ratio = cutoff_ratio
    
    def compute_dct_spectrum(self, signal: np.ndarray) -> np.ndarray:
        """计算 DCT 频谱"""
        if len(signal) < 2:
            return np.array([0])
        return dct(signal, norm='ortho')
    
    def compute_fft_spectrum(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算 FFT 频谱"""
        if len(signal) < 2:
            return np.array([0]), np.array([0])
        spectrum = np.abs(fft(signal))
        freqs = fftfreq(len(signal))
        return freqs[:len(freqs)//2], spectrum[:len(spectrum)//2]
    
    def compute_power_spectral_density(self, signal: np.ndarray) -> np.ndarray:
        """计算功率谱密度"""
        if len(signal) < 2:
            return np.array([1])
        spectrum = self.compute_dct_spectrum(signal)
        return np.abs(spectrum) ** 2
    
    def compute_spectral_entropy(self, psd: np.ndarray) -> float:
        """计算谱熵"""
        psd = psd + 1e-10  # 避免 log(0)
        p = psd / np.sum(psd)
        return -np.sum(p * np.log2(p))
    
    def compute_high_freq_ratio(self, psd: np.ndarray) -> float:
        """计算高频能量占比"""
        if len(psd) < 2:
            return 0
        cutoff = int(len(psd) * (1 - self.cutoff_ratio))
        high_freq_energy = np.sum(psd[cutoff:])
        total_energy = np.sum(psd)
        return high_freq_energy / (total_energy + 1e-10)
    
    def compute_energy_concentration(self, psd: np.ndarray, top_k: int = 5) -> float:
        """计算能量集中度（前 top_k 个频率分量的能量占比）"""
        if len(psd) < top_k:
            return 1.0
        sorted_psd = np.sort(psd)[::-1]
        top_energy = np.sum(sorted_psd[:top_k])
        total_energy = np.sum(psd)
        return top_energy / (total_energy + 1e-10)
    
    def analyze_segment(self, embeddings: np.ndarray) -> SpectralMetrics:
        """分析单个片段的频谱特性"""
        if embeddings.size == 0 or len(embeddings) < 2:
            return SpectralMetrics(0, 0, 0, 0, 0)
        
        # 对每个维度计算 PSD，然后平均
        n_tokens, n_dims = embeddings.shape
        all_psd = []
        
        for d in range(n_dims):
            psd = self.compute_power_spectral_density(embeddings[:, d])
            all_psd.append(psd)
        
        # 对齐并平均
        min_len = min(len(p) for p in all_psd)
        aligned_psd = np.array([p[:min_len] for p in all_psd])
        avg_psd = np.mean(aligned_psd, axis=0)
        
        return SpectralMetrics(
            spectral_entropy=self.compute_spectral_entropy(avg_psd),
            high_freq_ratio=self.compute_high_freq_ratio(avg_psd),
            energy_concentration=self.compute_energy_concentration(avg_psd),
            dominant_frequency=np.argmax(avg_psd) / len(avg_psd),
            total_energy=np.sum(avg_psd)
        )


class TrajectoryVisualizer:
    """轨迹可视化器"""
    
    def __init__(self, analyzer: AgentTrajectoryAnalyzer, spectral: SpectralAnalyzer):
        self.analyzer = analyzer
        self.spectral = spectral
        
    def plot_segment_statistics(self, save_path: str = None):
        """绘制片段统计信息"""
        segments = self.analyzer.segments
        
        # 统计各类型片段
        type_counts = {}
        type_tokens = {}
        for seg in segments:
            t = seg.type.value
            type_counts[t] = type_counts.get(t, 0) + 1
            type_tokens[t] = type_tokens.get(t, 0) + seg.token_count
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 片段数量
        types = list(type_counts.keys())
        counts = [type_counts[t] for t in types]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        axes[0].bar(types, counts, color=colors[:len(types)])
        axes[0].set_title('Segment Count by Type', fontsize=14)
        axes[0].set_xlabel('Segment Type')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Token 数量
        tokens = [type_tokens[t] for t in types]
        axes[1].bar(types, tokens, color=colors[:len(types)])
        axes[1].set_title('Token Count by Type', fontsize=14)
        axes[1].set_xlabel('Segment Type')
        axes[1].set_ylabel('Token Count')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
    def plot_spectrogram(self, save_path: str = None):
        """绘制频谱热力图"""
        if self.analyzer.token_embeddings is None:
            self.analyzer.generate_synthetic_embeddings()
        
        embeddings = self.analyzer.token_embeddings
        if embeddings.size == 0:
            print("No embeddings to visualize")
            return
            
        n_tokens, n_dims = embeddings.shape
        
        # 计算每个位置的频谱（使用滑动窗口）
        window_size = min(32, n_tokens // 4)
        if window_size < 4:
            window_size = 4
            
        hop_size = max(1, window_size // 2)
        
        spectrograms = []
        positions = []
        
        for i in range(0, n_tokens - window_size, hop_size):
            window = embeddings[i:i+window_size]
            # 对所有维度平均
            avg_signal = np.mean(window, axis=1)
            spectrum = np.abs(dct(avg_signal, norm='ortho'))
            spectrograms.append(spectrum)
            positions.append(i + window_size // 2)
        
        if not spectrograms:
            print("Not enough data for spectrogram")
            return
            
        spectrogram = np.array(spectrograms).T
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        im = ax.imshow(spectrogram, aspect='auto', cmap='viridis',
                       extent=[0, n_tokens, window_size//2, 0])
        
        # 标注片段边界
        colors_map = {
            SegmentType.THINK: 'red',
            SegmentType.TOOL_CALL: 'yellow',
            SegmentType.TOOL_RESPONSE: 'cyan'
        }
        
        current_pos = 0
        for seg in self.analyzer.segments:
            if seg.type in colors_map:
                ax.axvline(x=current_pos, color=colors_map[seg.type], 
                          linestyle='--', alpha=0.7, linewidth=1.5)
            current_pos += seg.token_count
        
        plt.colorbar(im, ax=ax, label='Energy')
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Frequency (DCT coefficient index)', fontsize=12)
        ax.set_title('Spectrogram of Agent Trajectory\n(Red: Think, Yellow: Tool Call, Cyan: Tool Response)', 
                    fontsize=14)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
    def plot_energy_decay_curves(self, save_path: str = None):
        """绘制不同类型片段的能量衰减曲线"""
        if self.analyzer.token_embeddings is None:
            self.analyzer.generate_synthetic_embeddings()
        
        embeddings = self.analyzer.token_embeddings
        segments = self.analyzer.segments
        
        # 收集各类型的 PSD
        type_psds = {
            SegmentType.THINK: [],
            SegmentType.TOOL_CALL: [],
            SegmentType.TOOL_RESPONSE: []
        }
        
        current_pos = 0
        for seg in segments:
            if seg.type in type_psds and seg.token_count >= 4:
                seg_embeddings = embeddings[current_pos:current_pos + seg.token_count]
                if len(seg_embeddings) >= 4:
                    avg_signal = np.mean(seg_embeddings, axis=1)
                    psd = self.spectral.compute_power_spectral_density(avg_signal)
                    # 归一化
                    psd = psd / (np.max(psd) + 1e-10)
                    type_psds[seg.type].append(psd)
            current_pos += seg.token_count
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = {
            SegmentType.THINK: '#FF6B6B',
            SegmentType.TOOL_CALL: '#4ECDC4',
            SegmentType.TOOL_RESPONSE: '#45B7D1'
        }
        
        for seg_type, psds in type_psds.items():
            if psds:
                # 对齐并平均
                min_len = min(len(p) for p in psds)
                aligned = np.array([p[:min_len] for p in psds])
                avg_psd = np.mean(aligned, axis=0)
                std_psd = np.std(aligned, axis=0)
                
                freqs = np.arange(len(avg_psd)) / len(avg_psd)
                ax.plot(freqs, avg_psd, label=seg_type.value, 
                       color=colors[seg_type], linewidth=2)
                ax.fill_between(freqs, avg_psd - std_psd, avg_psd + std_psd,
                               color=colors[seg_type], alpha=0.2)
        
        ax.set_xlabel('Normalized Frequency', fontsize=12)
        ax.set_ylabel('Normalized Power Spectral Density', fontsize=12)
        ax.set_title('Energy Decay Curves by Segment Type', fontsize=14)
        ax.legend(fontsize=11)
        ax.set_xlim([0, 0.5])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
    def plot_spectral_metrics_comparison(self, save_path: str = None):
        """绘制频谱指标对比"""
        if self.analyzer.token_embeddings is None:
            self.analyzer.generate_synthetic_embeddings()
        
        embeddings = self.analyzer.token_embeddings
        segments = self.analyzer.segments
        
        # 收集各类型的指标
        type_metrics = {
            SegmentType.THINK: {'entropy': [], 'hfr': [], 'concentration': []},
            SegmentType.TOOL_CALL: {'entropy': [], 'hfr': [], 'concentration': []},
            SegmentType.TOOL_RESPONSE: {'entropy': [], 'hfr': [], 'concentration': []}
        }
        
        current_pos = 0
        for seg in segments:
            if seg.type in type_metrics and seg.token_count >= 4:
                seg_embeddings = embeddings[current_pos:current_pos + seg.token_count]
                if len(seg_embeddings) >= 4:
                    metrics = self.spectral.analyze_segment(seg_embeddings)
                    type_metrics[seg.type]['entropy'].append(metrics.spectral_entropy)
                    type_metrics[seg.type]['hfr'].append(metrics.high_freq_ratio)
                    type_metrics[seg.type]['concentration'].append(metrics.energy_concentration)
            current_pos += seg.token_count
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metric_names = ['Spectral Entropy', 'High-Freq Ratio', 'Energy Concentration']
        metric_keys = ['entropy', 'hfr', 'concentration']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for idx, (name, key) in enumerate(zip(metric_names, metric_keys)):
            data = []
            labels = []
            for seg_type in [SegmentType.THINK, SegmentType.TOOL_CALL, SegmentType.TOOL_RESPONSE]:
                if type_metrics[seg_type][key]:
                    data.append(type_metrics[seg_type][key])
                    labels.append(seg_type.value)
            
            if data:
                bp = axes[idx].boxplot(data, labels=labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors[:len(data)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            axes[idx].set_title(name, fontsize=12)
            axes[idx].set_ylabel('Value')
            axes[idx].tick_params(axis='x', rotation=30)
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('Spectral Metrics Comparison by Segment Type', fontsize=14, y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def analyze_trajectory_file(json_data: dict) -> dict:
    """分析单个轨迹文件"""
    # 初始化分析器
    trajectory_analyzer = AgentTrajectoryAnalyzer(json_data)
    segments = trajectory_analyzer.parse_trajectory()
    trajectory_analyzer.generate_synthetic_embeddings()
    
    spectral_analyzer = SpectralAnalyzer(cutoff_ratio=0.2)
    
    # 计算各类型片段的频谱指标
    results = {
        'segment_counts': {},
        'token_counts': {},
        'spectral_metrics': {}
    }
    
    embeddings = trajectory_analyzer.token_embeddings
    current_pos = 0
    
    type_metrics = {t: [] for t in SegmentType}
    
    for seg in segments:
        t = seg.type.value
        results['segment_counts'][t] = results['segment_counts'].get(t, 0) + 1
        results['token_counts'][t] = results['token_counts'].get(t, 0) + seg.token_count
        
        if seg.token_count >= 4:
            seg_embeddings = embeddings[current_pos:current_pos + seg.token_count]
            if len(seg_embeddings) >= 4:
                metrics = spectral_analyzer.analyze_segment(seg_embeddings)
                type_metrics[seg.type].append(metrics)
        
        current_pos += seg.token_count
    
    # 汇总指标
    for seg_type, metrics_list in type_metrics.items():
        if metrics_list:
            avg_metrics = {
                'avg_spectral_entropy': np.mean([m.spectral_entropy for m in metrics_list]),
                'avg_high_freq_ratio': np.mean([m.high_freq_ratio for m in metrics_list]),
                'avg_energy_concentration': np.mean([m.energy_concentration for m in metrics_list]),
                'avg_total_energy': np.mean([m.total_energy for m in metrics_list]),
                'count': len(metrics_list)
            }
            results['spectral_metrics'][seg_type.value] = avg_metrics
    
    return results, trajectory_analyzer, spectral_analyzer


def main():
    """主函数"""
    json_data = {
        "question": "I'm researching species that became invasive after people who kept them as pets released them...",
        "answer": "34689",
        "rollout_id": 1,
        "messages": [
            {"role": "system", "content": "You are a Web Information Seeking Master..."},
            {"role": "user", "content": "A conversation between User and Assistant..."},
            {"role": "assistant", "content": "<think>To determine the exact zip codes where the fish species from Finding Nemo became established as nonnative, I need highly specific, authoritative information...</think>\n<tool_call>\n{\"name\": \"search\", \"arguments\": {\"query\": [\"Finding Nemo fish nonnative found zip codes\", \"USGS Finding Nemo fish nonnative USGS\"]}}\n</tool_call>"},
            {"role": "user", "content": "<tool_response>\nA Google search for 'Finding Nemo fish nonnative found zip codes' found 9 results...\n</tool_response>"},
            {"role": "assistant", "content": "<think>The USGS Nonindigenous Aquatic Species database is likely to contain authoritative, georeferenced records...</think>\n<tool_call>\n{\"name\": \"visit\", \"arguments\": {\"url\": [\"https://nas.er.usgs.gov/queries/FactSheet.aspx?speciesID=3243\"], \"goal\": \"Extract the zip codes\"}}\n</tool_call>"},
            {"role": "user", "content": "<tool_response>\nThe useful information in https://nas.er.usgs.gov/queries/FactSheet.aspx?speciesID=3243...\n</tool_response>"},
            {"role": "assistant", "content": "<think>The previous attempts to directly access authoritative USGS fact sheets were unsuccessful...</think>\n<tool_call>\n{\"name\": \"search\", \"arguments\": {\"query\": [\"Australis nonnative zip codes USGS\", \"Finding Nemo fish nonnative zip codes\"]}}\n</tool_call>"},
        ],
        "prediction": "The clown anemonefish was first reported at Fred Howard Park, Florida (zip code 34689).",
        "termination": "generate an answer as token limit reached",
        "tool_call_search_times": 11,
        "tool_call_visit_times": 11
    }
    
    print("=" * 60)
    print("Agent 轨迹频谱分析实验")
    print("=" * 60)
    
    # 分析轨迹
    results, trajectory_analyzer, spectral_analyzer = analyze_trajectory_file(json_data)
    
    # 打印统计信息
    print("\n📊 片段统计:")
    print("-" * 40)
    for seg_type, count in results['segment_counts'].items():
        tokens = results['token_counts'].get(seg_type, 0)
        print(f"  {seg_type:15s}: {count:3d} 个片段, {tokens:6d} tokens")
    
    print("\n📈 频谱指标:")
    print("-" * 40)
    for seg_type, metrics in results['spectral_metrics'].items():
        print(f"\n  [{seg_type}]")
        print(f"    谱熵 (Spectral Entropy):     {metrics['avg_spectral_entropy']:.4f}")
        print(f"    高频能量占比 (HFR):          {metrics['avg_high_freq_ratio']:.4f}")
        print(f"    能量集中度 (Concentration):  {metrics['avg_energy_concentration']:.4f}")
        print(f"    平均总能量:                  {metrics['avg_total_energy']:.4f}")
    
    # 可视化
    print("\n🎨 生成可视化图表...")
    visualizer = TrajectoryVisualizer(trajectory_analyzer, spectral_analyzer)
    
    # 绘制各种图表
    visualizer.plot_segment_statistics()
    visualizer.plot_spectrogram()
    visualizer.plot_energy_decay_curves()
    visualizer.plot_spectral_metrics_comparison()
    
    
    return results


if __name__ == "__main__":
    results = main()