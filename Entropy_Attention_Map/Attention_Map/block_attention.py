"""
Block Attention Map Visualization Tool (Optimized Version)
用于可视化Agent对话流中不同语义块之间的注意力关系
针对大模型多卡环境进行了显存优化
"""

import json
import re
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 数据结构定义
# ============================================================================

@dataclass
class SemanticBlock:
    """语义块的数据结构"""
    block_type: str  # "system_init", "think", "tool_call", "tool_response", "answer"
    content: str
    start_idx: int = 0
    end_idx: int = 0
    token_count: int = 0
    role: str = ""
    turn_idx: int = 0  # 对话轮次

@dataclass
class AttentionData:
    """注意力数据的存储结构"""
    layer_idx: int
    attention_weights: torch.Tensor  # [num_heads, seq_len, seq_len]
    
@dataclass
class BlockAttentionResult:
    """Block级别注意力结果"""
    blocks: List[SemanticBlock]
    block_attention_matrix: np.ndarray  # [num_blocks, num_blocks]
    layer_idx: int
    head_aggregation: str  # "mean", "max", "specific_head"


# ============================================================================
# 2. 语义块解析器
# ============================================================================

class SemanticBlockParser:
    """解析对话消息，提取语义块"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def parse_messages(self, messages: List[Dict]) -> List[SemanticBlock]:
        """
        解析消息列表，返回语义块列表
        
        Args:
            messages: [{"role": "system/user/assistant", "content": "..."}]
        
        Returns:
            List[SemanticBlock]: 语义块列表
        """
        blocks = []
        turn_idx = 0
        is_first_user = True
        
        for msg_idx, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                # System prompt 单独作为一个block
                blocks.append(SemanticBlock(
                    block_type="system_init",
                    content=content,
                    role=role,
                    turn_idx=turn_idx
                ))
                
            elif role == "user":
                # 检查是否是tool_response
                if "<tool_response>" in content:
                    # 提取tool_response内容
                    responses = self._extract_tag_content(content, "tool_response")
                    for resp in responses:
                        blocks.append(SemanticBlock(
                            block_type="tool_response",
                            content=resp,
                            role=role,
                            turn_idx=turn_idx
                        ))
                else:
                    # 普通用户消息
                    if is_first_user and len(blocks) > 0 and blocks[-1].block_type == "system_init":
                        # 第一条user消息合并到system_init
                        blocks[-1].content += "\n" + content
                        blocks[-1].block_type = "system_init"
                    else:
                        blocks.append(SemanticBlock(
                            block_type="user_query",
                            content=content,
                            role=role,
                            turn_idx=turn_idx
                        ))
                    is_first_user = False
                    turn_idx += 1
                    
            elif role == "assistant":
                # 解析assistant的输出，可能包含think, tool_call, answer
                sub_blocks = self._parse_assistant_content(content, turn_idx)
                blocks.extend(sub_blocks)
                
        return blocks
    
    def _parse_assistant_content(self, content: str, turn_idx: int) -> List[SemanticBlock]:
        """解析assistant的输出内容"""
        blocks = []
        remaining = content
        
        # 提取think部分
        thinks = self._extract_tag_content(content, "think")
        for think in thinks:
            blocks.append(SemanticBlock(
                block_type="think",
                content=think,
                role="assistant",
                turn_idx=turn_idx
            ))
            remaining = remaining.replace(f"<think>{think}</think>", "", 1)
        
        # 提取tool_call部分
        tool_calls = self._extract_tag_content(content, "tool_call")
        for tc in tool_calls:
            blocks.append(SemanticBlock(
                block_type="tool_call",
                content=tc,
                role="assistant",
                turn_idx=turn_idx
            ))
            remaining = remaining.replace(f"<tool_call>{tc}</tool_call>", "", 1)
        
        # 剩余部分作为answer
        remaining = remaining.strip()
        if remaining:
            blocks.append(SemanticBlock(
                block_type="answer",
                content=remaining,
                role="assistant",
                turn_idx=turn_idx
            ))
            
        return blocks
    
    def _extract_tag_content(self, text: str, tag: str) -> List[str]:
        """提取指定标签内的内容"""
        pattern = f"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches
    
    def tokenize_and_index_blocks(self, blocks: List[SemanticBlock], 
                                   chat_template: bool = True) -> Tuple[List[SemanticBlock], torch.Tensor]:
        """
        对blocks进行tokenize，并记录每个block的token索引范围
        
        Returns:
            blocks: 更新了索引的blocks
            input_ids: 完整的token序列
        """
        # 重建完整的消息用于tokenize
        full_text = ""
        for block in blocks:
            full_text += block.content + " "
        
        # 获取完整序列的tokens
        full_tokens = self.tokenizer.encode(full_text, return_tensors="pt")[0]
        
        # 逐个block计算其token范围
        current_idx = 0
        for block in blocks:
            block_tokens = self.tokenizer.encode(block.content, add_special_tokens=False)
            block.start_idx = current_idx
            block.end_idx = current_idx + len(block_tokens)
            block.token_count = len(block_tokens)
            current_idx = block.end_idx
            
        return blocks, full_tokens


# ============================================================================
# 3. 优化的注意力提取器 (显存友好版本)
# ============================================================================

class OptimizedAttentionExtractor:
    """
    优化的注意力提取器
    - 支持分层提取，避免同时存储所有层的attention
    - 立即将attention转移到CPU
    - 支持分块聚合，避免存储完整的attention矩阵
    """
    
    def __init__(self, model, blocks: List[SemanticBlock] = None):
        """
        Args:
            model: HuggingFace模型
            blocks: 语义块列表，用于实时聚合（可选）
        """
        self.model = model
        self.blocks = blocks
        self.hooks = []
        self.current_layer_idx = None
        
        # 存储block-level的聚合结果，而不是完整的attention矩阵
        self.block_attention_cache: Dict[int, np.ndarray] = {}
        
        # 临时存储当前层的attention（用于实时聚合）
        self.temp_attention: Optional[torch.Tensor] = None
        
        # 聚合器
        self.aggregator = BlockAttentionAggregator(aggregation_method="mean")
        
    def set_blocks(self, blocks: List[SemanticBlock]):
        """设置语义块（用于实时聚合）"""
        self.blocks = blocks
        
    def _create_aggregating_hook(self, layer_idx: int):
        """
        创建一个会实时聚合attention的hook
        这样可以避免存储完整的[num_heads, seq_len, seq_len]矩阵
        """
        def hook(module, input, output):
            # output通常是 (attn_output, attn_weights, past_key_value)
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]  # [batch, num_heads, seq_len, seq_len]
                if attn_weights is not None:
                    # 立即在GPU上进行head聚合，然后转移到CPU
                    # 这样可以减少数据传输量
                    with torch.no_grad():
                        # 对heads取平均: [batch, num_heads, seq_len, seq_len] -> [batch, seq_len, seq_len]
                        attn_mean = attn_weights.mean(dim=1)
                        
                        if self.blocks is not None:
                            # 实时计算block-level attention
                            block_attn = self._compute_block_attention_gpu(
                                attn_mean[0], self.blocks
                            )
                            self.block_attention_cache[layer_idx] = block_attn
                        else:
                            # 如果没有blocks信息，存储聚合后的attention
                            self.temp_attention = attn_mean[0].cpu().numpy()
                            
                    # 显式释放GPU内存
                    del attn_weights
                    torch.cuda.empty_cache()
                    
        return hook
    
    def _compute_block_attention_gpu(self, 
                                      attention_matrix: torch.Tensor,
                                      blocks: List[SemanticBlock]) -> np.ndarray:
        """
        在GPU上计算block-level attention，然后转移到CPU
        这比先转移整个矩阵到CPU再计算更高效
        
        Args:
            attention_matrix: [seq_len, seq_len] on GPU
            blocks: 语义块列表
        """
        num_blocks = len(blocks)
        block_attention = torch.zeros((num_blocks, num_blocks), device=attention_matrix.device)
        
        seq_len = attention_matrix.shape[0]
        
        for i, block_i in enumerate(blocks):
            for j, block_j in enumerate(blocks):
                start_i = min(block_i.start_idx, seq_len - 1)
                end_i = min(block_i.end_idx, seq_len)
                start_j = min(block_j.start_idx, seq_len - 1)
                end_j = min(block_j.end_idx, seq_len)
                
                if end_i > start_i and end_j > start_j:
                    sub_matrix = attention_matrix[start_i:end_i, start_j:end_j]
                    block_attention[i, j] = sub_matrix.mean()
        
        # 转移到CPU并转换为numpy
        result = block_attention.cpu().numpy()
        
        # 释放GPU tensor
        del block_attention
        
        return result
    
    def _create_simple_hook(self, layer_idx: int):
        """
        创建简单的hook，只提取并立即转移到CPU
        用于不进行实时聚合的场景
        """
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    with torch.no_grad():
                        # 对heads取平均后立即转移到CPU
                        attn_mean = attn_weights.mean(dim=1)[0].cpu()
                        self.temp_attention = attn_mean.numpy()
                    
                    del attn_weights
                    torch.cuda.empty_cache()
                    
        return hook
    
    def register_hook_for_layer(self, layer_idx: int, use_aggregation: bool = True):
        """
        只为单个层注册hook
        
        Args:
            layer_idx: 要注册的层索引
            use_aggregation: 是否使用实时聚合
        """
        self.remove_hooks()
        self.current_layer_idx = layer_idx
        self.temp_attention = None
        
        # 获取模型的attention层
        layers = self._get_model_layers()
        
        if layer_idx < len(layers):
            attn_module = layers[layer_idx].self_attn
            
            if use_aggregation and self.blocks is not None:
                hook = attn_module.register_forward_hook(
                    self._create_aggregating_hook(layer_idx)
                )
            else:
                hook = attn_module.register_forward_hook(
                    self._create_simple_hook(layer_idx)
                )
            
            self.hooks.append(hook)
            print(f"Registered hook on layer {layer_idx}")
    
    def _get_model_layers(self):
        """获取模型的层列表"""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        else:
            raise ValueError("Unsupported model architecture")
    
    def remove_hooks(self):
        """移除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def get_block_attention(self, layer_idx: int) -> Optional[np.ndarray]:
        """获取指定层的block attention"""
        return self.block_attention_cache.get(layer_idx)
    
    def get_temp_attention(self) -> Optional[np.ndarray]:
        """获取临时存储的attention矩阵"""
        return self.temp_attention
    
    def clear_cache(self):
        """清除所有缓存"""
        self.block_attention_cache.clear()
        self.temp_attention = None
        gc.collect()
        torch.cuda.empty_cache()


class ChunkedAttentionExtractor:
    """
    分块注意力提取器
    对于超长序列，分块提取attention以减少显存使用
    """
    
    def __init__(self, model, chunk_size: int = 2048):
        """
        Args:
            model: HuggingFace模型
            chunk_size: 每个chunk的最大序列长度
        """
        self.model = model
        self.chunk_size = chunk_size
        
    def extract_block_attention_chunked(self,
                                         input_ids: torch.Tensor,
                                         blocks: List[SemanticBlock],
                                         layer_idx: int) -> np.ndarray:
        """
        分块提取并聚合block-level attention
        
        注意：这种方法对于causal attention是近似的，
        因为跨chunk的attention需要特殊处理
        """
        seq_len = input_ids.shape[1]
        num_blocks = len(blocks)
        
        # 初始化block attention矩阵
        block_attention = np.zeros((num_blocks, num_blocks))
        block_counts = np.zeros((num_blocks, num_blocks))
        
        # 获取layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        else:
            layers = self.model.transformer.h
        
        # 分块处理
        for chunk_start in range(0, seq_len, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            
            # 准备chunk输入
            chunk_input = input_ids[:, :chunk_end].to(self.model.device)
            
            # 创建临时hook
            temp_attention = [None]
            
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                    with torch.no_grad():
                        # 只取当前chunk对应的部分
                        attn = output[1].mean(dim=1)[0]  # [seq_len, seq_len]
                        # 只需要chunk_start:chunk_end行的attention
                        if chunk_start > 0:
                            temp_attention[0] = attn[chunk_start:chunk_end, :].cpu().numpy()
                        else:
                            temp_attention[0] = attn.cpu().numpy()
            
            handle = layers[layer_idx].self_attn.register_forward_hook(hook)
            
            try:
                with torch.no_grad():
                    self.model(chunk_input, output_attentions=False)
            finally:
                handle.remove()
            
            # 聚合到block attention
            if temp_attention[0] is not None:
                attn_chunk = temp_attention[0]
                
                for i, block_i in enumerate(blocks):
                    # 检查block_i是否在当前chunk的query范围内
                    if block_i.start_idx >= chunk_end or block_i.end_idx <= chunk_start:
                        continue
                    
                    # 计算block_i在当前chunk中的有效范围
                    local_start_i = max(0, block_i.start_idx - chunk_start)
                    local_end_i = min(chunk_end - chunk_start, block_i.end_idx - chunk_start)
                    
                    if local_end_i <= local_start_i:
                        continue
                    
                    for j, block_j in enumerate(blocks):
                        # block_j在完整序列中的范围
                        start_j = block_j.start_idx
                        end_j = min(block_j.end_idx, chunk_end)
                        
                        if end_j <= start_j:
                            continue
                        
                        # 提取子矩阵
                        if local_start_i < attn_chunk.shape[0] and start_j < attn_chunk.shape[1]:
                            actual_end_i = min(local_end_i, attn_chunk.shape[0])
                            actual_end_j = min(end_j, attn_chunk.shape[1])
                            
                            sub_matrix = attn_chunk[local_start_i:actual_end_i, start_j:actual_end_j]
                            if sub_matrix.size > 0:
                                block_attention[i, j] += sub_matrix.sum()
                                block_counts[i, j] += sub_matrix.size
            
            # 清理
            del chunk_input
            torch.cuda.empty_cache()
        
        # 平均
        block_counts[block_counts == 0] = 1
        block_attention = block_attention / block_counts
        
        return block_attention


# ============================================================================
# 4. Block Attention 聚合计算器
# ============================================================================

class BlockAttentionAggregator:
    """将token级别的注意力聚合为block级别"""
    
    def __init__(self, aggregation_method: str = "mean"):
        """
        Args:
            aggregation_method: "mean" 平均, "sum" 求和, "max" 最大值
        """
        self.aggregation_method = aggregation_method
        
    def aggregate(self, 
                  attention_matrix: np.ndarray, 
                  blocks: List[SemanticBlock],
                  head_aggregation: str = "mean") -> np.ndarray:
        """
        将token-level attention聚合为block-level attention
        
        Args:
            attention_matrix: [num_heads, seq_len, seq_len] 或 [seq_len, seq_len]
            blocks: 语义块列表，包含每个块的token索引范围
            head_aggregation: 对多头的聚合方式 "mean", "max"
        
        Returns:
            block_attention: [num_blocks, num_blocks]
        """
        # 如果有多头，先聚合
        if len(attention_matrix.shape) == 3:
            if head_aggregation == "mean":
                attention_matrix = attention_matrix.mean(axis=0)
            elif head_aggregation == "max":
                attention_matrix = attention_matrix.max(axis=0)
            else:
                attention_matrix = attention_matrix.mean(axis=0)
        
        num_blocks = len(blocks)
        block_attention = np.zeros((num_blocks, num_blocks))
        
        seq_len = attention_matrix.shape[0]
        
        for i, block_i in enumerate(blocks):  # Query blocks
            for j, block_j in enumerate(blocks):  # Key blocks
                # 提取block_i对block_j的注意力子矩阵
                start_i = min(block_i.start_idx, seq_len - 1)
                end_i = min(block_i.end_idx, seq_len)
                start_j = min(block_j.start_idx, seq_len - 1)
                end_j = min(block_j.end_idx, seq_len)
                
                if end_i <= start_i or end_j <= start_j:
                    continue
                    
                sub_matrix = attention_matrix[start_i:end_i, start_j:end_j]
                
                # 根据聚合方法计算block-level score
                if self.aggregation_method == "mean":
                    score = sub_matrix.mean() if sub_matrix.size > 0 else 0
                elif self.aggregation_method == "sum":
                    score = sub_matrix.sum() if sub_matrix.size > 0 else 0
                elif self.aggregation_method == "max":
                    score = sub_matrix.max() if sub_matrix.size > 0 else 0
                else:
                    score = sub_matrix.mean() if sub_matrix.size > 0 else 0
                    
                block_attention[i, j] = score
                
        return block_attention
    
    def normalize(self, block_attention: np.ndarray, method: str = "row") -> np.ndarray:
        """
        归一化block attention矩阵
        
        Args:
            method: "row" 按行归一化, "global" 全局归一化, "none" 不归一化
        """
        if method == "row":
            row_sums = block_attention.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # 避免除零
            return block_attention / row_sums
        elif method == "global":
            max_val = block_attention.max()
            if max_val > 0:
                return block_attention / max_val
            return block_attention
        else:
            return block_attention


# ============================================================================
# 5. 可视化器
# ============================================================================

class BlockAttentionVisualizer:
    """Block Attention Map可视化"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        self.figsize = figsize
        self.block_type_colors = {
            "system_init": "#FF6B6B",
            "user_query": "#4ECDC4",
            "think": "#45B7D1",
            "tool_call": "#96CEB4",
            "tool_response": "#FFEAA7",
            "answer": "#DDA0DD"
        }
        
    def create_block_labels(self, blocks: List[SemanticBlock], 
                            max_content_len: int = 20) -> List[str]:
        """创建块标签"""
        labels = []
        for i, block in enumerate(blocks):
            content_preview = block.content[:max_content_len]
            if len(block.content) > max_content_len:
                content_preview += "..."
            content_preview = content_preview.replace("\n", " ")
            label = f"[{block.block_type}]\nT{block.turn_idx}\n({block.token_count}tok)"
            labels.append(label)
        return labels
    
    def plot_attention_map(self, 
                           block_attention: np.ndarray,
                           blocks: List[SemanticBlock],
                           layer_idx: int,
                           title: str = None,
                           save_path: str = None,
                           show_values: bool = True):
        """
        绘制Block Attention热力图
        
        Args:
            block_attention: [num_blocks, num_blocks] 的注意力矩阵
            blocks: 语义块列表
            layer_idx: 层索引
            title: 图标题
            save_path: 保存路径
            show_values: 是否显示数值
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 创建热力图
        sns.heatmap(
            block_attention,
            annot=show_values,
            fmt=".3f",
            cmap="YlOrRd",
            square=True,
            linewidths=0.5,
            linecolor='white',
            cbar_kws={'label': 'Attention Score'},
            ax=ax
        )
        
        # 设置标签
        labels = self.create_block_labels(blocks)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels, rotation=0, fontsize=8)
        
        ax.set_xlabel("Key Blocks (Context)", fontsize=12)
        ax.set_ylabel("Query Blocks (Generation)", fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f"Block Attention Map - Layer {layer_idx}", fontsize=14, fontweight='bold')
        
        # 添加block类型颜色条
        self._add_block_type_legend(ax, blocks)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
            
        plt.show()
        
    def _add_block_type_legend(self, ax, blocks: List[SemanticBlock]):
        """添加block类型图例"""
        from matplotlib.patches import Patch
        
        # 获取所有出现的block类型
        block_types = list(set(b.block_type for b in blocks))
        
        legend_elements = [
            Patch(facecolor=self.block_type_colors.get(bt, "#CCCCCC"), 
                  edgecolor='black',
                  label=bt)
            for bt in block_types
        ]
        
        ax.legend(handles=legend_elements, loc='upper left', 
                  bbox_to_anchor=(1.15, 1), title="Block Types")
        
    def plot_multi_layer_comparison(self,
                                    attention_maps: Dict[int, np.ndarray],
                                    blocks: List[SemanticBlock],
                                    save_path: str = None):
        """
        对比多层的Block Attention Map
        
        Args:
            attention_maps: {layer_idx: block_attention_matrix}
        """
        num_layers = len(attention_maps)
        fig, axes = plt.subplots(1, num_layers, figsize=(8*num_layers, 8))
        
        if num_layers == 1:
            axes = [axes]
            
        labels = self.create_block_labels(blocks)
        
        for idx, (layer_idx, attn_matrix) in enumerate(attention_maps.items()):
            ax = axes[idx]
            
            sns.heatmap(
                attn_matrix,
                annot=True,
                fmt=".3f",
                cmap="YlOrRd",
                square=True,
                linewidths=0.5,
                linecolor='white',
                ax=ax
            )
            
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(labels, rotation=0, fontsize=8)
            ax.set_title(f"Layer {layer_idx}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Key Blocks")
            ax.set_ylabel("Query Blocks")
            
        plt.suptitle("Block Attention Map Comparison Across Layers", 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
            
        plt.show()
        
    def plot_attention_flow(self,
                            block_attention: np.ndarray,
                            blocks: List[SemanticBlock],
                            top_k: int = 5,
                            save_path: str = None):
        """
        绘制注意力流向图（显示每个生成块主要关注哪些上下文块）
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        num_blocks = len(blocks)
        
        # 绘制块节点
        for i, block in enumerate(blocks):
            color = self.block_type_colors.get(block.block_type, "#CCCCCC")
            ax.scatter(i, 0, s=300, c=color, zorder=3, edgecolors='black', linewidth=2)
            ax.annotate(f"{block.block_type}\nT{block.turn_idx}", 
                       (i, -0.1), ha='center', fontsize=8, rotation=45)
        
        # 绘制注意力流向（用箭头）
        for i in range(num_blocks):
            # 获取该块关注最多的top_k个块
            attention_row = block_attention[i]
            top_indices = np.argsort(attention_row)[-top_k:][::-1]
            
            for rank, j in enumerate(top_indices):
                if i != j and attention_row[j] > 0.01:  # 忽略自注意力和很小的值
                    alpha = attention_row[j]
                    ax.annotate(
                        '',
                        xy=(j, 0), xytext=(i, 0),
                        arrowprops=dict(
                            arrowstyle='->',
                            color='blue',
                            alpha=min(alpha * 3, 1),
                            linewidth=1 + alpha * 3,
                            connectionstyle='arc3,rad=0.3'
                        )
                    )
                    
        ax.set_xlim(-1, num_blocks)
        ax.set_ylim(-0.5, 0.5)
        ax.axis('off')
        ax.set_title("Attention Flow Between Blocks", fontsize=14, fontweight='bold')
        
        # 添加图例
        self._add_block_type_legend(ax, blocks)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        plt.show()


# ============================================================================
# 6. 主Pipeline类 (优化版本)
# ============================================================================

class BlockAttentionAnalyzer:
    """
    Block Attention分析的主类，整合所有组件
    针对大模型进行了显存优化
    """
    
    def __init__(self, 
                 model_name_or_path: str,
                 device_map: str = "auto",
                 torch_dtype = torch.float16,
                 use_flash_attn: bool = False,
                 load_in_8bit: bool = False,
                 load_in_4bit: bool = False,
                 max_memory: Dict[int, str] = None):
        """
        Args:
            model_name_or_path: 模型路径或HuggingFace模型名
            device_map: 设备映射策略
            torch_dtype: 数据类型
            use_flash_attn: 是否使用flash attention (注意：使用flash attn时无法获取attention weights)
            load_in_8bit: 是否使用8bit量化
            load_in_4bit: 是否使用4bit量化
            max_memory: 每张卡的最大显存使用，如 {0: "20GiB", 1: "20GiB", ...}
        """
        print(f"Loading model: {model_name_or_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        
        # 配置量化
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("Using 4-bit quantization")
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            print("Using 8-bit quantization")
        
        # 设置max_memory以均衡显存使用
        if max_memory is None and torch.cuda.device_count() > 1:
            # 自动设置，为每张卡预留一些空间
            max_memory = {}
            for i in range(torch.cuda.device_count()):
                # 获取每张卡的总显存
                total_mem = torch.cuda.get_device_properties(i).total_memory
                # 使用80%的显存
                max_memory[i] = f"{int(total_mem * 0.8 / 1024**3)}GiB"
            max_memory["cpu"] = "50GiB"
            print(f"Auto max_memory: {max_memory}")
        
        # 加载模型
        # 注意：必须使用 attn_implementation="eager" 来获取attention weights
        # 但eager模式在某些情况下可能导致显存集中
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="eager",  # 必须使用eager以获取attention weights
            quantization_config=quantization_config,
            max_memory=max_memory,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        
        # 计算层索引
        self.num_layers = self._get_num_layers()
        self.layer_2_3 = int(self.num_layers * 2 / 3)
        self.last_layer = self.num_layers - 1
        
        print(f"Model loaded. Total layers: {self.num_layers}")
        print(f"Will extract attention from layers: {self.layer_2_3} (2/3) and {self.last_layer} (last)")
        
        # 打印设备分布
        self._print_device_distribution()
        
        # 初始化组件
        self.parser = SemanticBlockParser(self.tokenizer)
        self.aggregator = BlockAttentionAggregator(aggregation_method="mean")
        self.visualizer = BlockAttentionVisualizer()
        
    def _get_num_layers(self) -> int:
        """获取模型层数"""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return len(self.model.model.layers)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return len(self.model.transformer.h)
        else:
            raise ValueError("Cannot determine number of layers")
    
    def _get_model_layers(self):
        """获取模型的层列表"""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        else:
            raise ValueError("Unsupported model architecture")
    
    def _print_device_distribution(self):
        """打印模型在各设备上的分布"""
        if hasattr(self.model, 'hf_device_map'):
            device_map = self.model.hf_device_map
            device_counts = {}
            for name, device in device_map.items():
                device = str(device)
                device_counts[device] = device_counts.get(device, 0) + 1
            print(f"Model device distribution: {device_counts}")
    
    def _get_layer_device(self, layer_idx: int) -> torch.device:
        """获取指定层所在的设备"""
        layers = self._get_model_layers()
        if layer_idx < len(layers):
            # 尝试获取层的设备
            for param in layers[layer_idx].parameters():
                return param.device
        return torch.device("cuda:0")
    
    def analyze_conversation(self, 
                            messages: List[Dict],
                            normalize: str = "row",
                            layer_indices: List[int] = None,
                            use_chunked: bool = False,
                            chunk_size: int = 2048) -> Dict[str, Any]:
        """
        分析单个对话的Block Attention
        
        Args:
            messages: 对话消息列表
            normalize: 归一化方法
            layer_indices: 要分析的层索引列表，默认为 [2/3层, 最后一层]
            use_chunked: 是否使用分块处理（对于超长序列）
            chunk_size: 分块大小
            
        Returns:
            分析结果字典
        """
        if layer_indices is None:
            layer_indices = [self.layer_2_3, self.last_layer]
        
        # 1. 解析语义块
        blocks = self.parser.parse_messages(messages)
        print(f"Parsed {len(blocks)} semantic blocks:")
        for i, block in enumerate(blocks):
            print(f"  [{i}] {block.block_type} (turn {block.turn_idx}): {block.content[:50]}...")
        
        # 2. 构建完整输入
        if hasattr(self.tokenizer, 'apply_chat_template'):
            input_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        else:
            input_text = ""
            for msg in messages:
                input_text += f"<|{msg['role']}|>\n{msg['content']}\n"
        
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        
        # 重新计算每个block的token索引
        blocks = self._recompute_block_indices(blocks, input_text)
        
        seq_len = input_ids.shape[1]
        print(f"Total tokens: {seq_len}")
        
        # 3. 逐层提取attention（显存优化的关键）
        block_attention_maps = {}
        
        if use_chunked and seq_len > chunk_size:
            # 使用分块处理
            print(f"Using chunked extraction (chunk_size={chunk_size})")
            chunked_extractor = ChunkedAttentionExtractor(self.model, chunk_size)
            
            for layer_idx in layer_indices:
                print(f"Extracting attention from layer {layer_idx}...")
                block_attn = chunked_extractor.extract_block_attention_chunked(
                    input_ids, blocks, layer_idx
                )
                block_attn = self.aggregator.normalize(block_attn, method=normalize)
                block_attention_maps[layer_idx] = block_attn
                
                # 清理显存
                gc.collect()
                torch.cuda.empty_cache()
        else:
            # 使用优化的hook方式，逐层提取
            extractor = OptimizedAttentionExtractor(self.model)
            extractor.set_blocks(blocks)
            
            for layer_idx in layer_indices:
                print(f"Extracting attention from layer {layer_idx}...")
                
                # 只为当前层注册hook
                extractor.register_hook_for_layer(layer_idx, use_aggregation=True)
                
                # 前向传播
                # 将输入放到第一层所在的设备
                input_device = self._get_layer_device(0)
                input_ids_device = input_ids.to(input_device)
                
                with torch.no_grad():
                    try:
                        self.model(input_ids_device, output_attentions=False, use_cache=False)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"OOM on layer {layer_idx}, trying chunked extraction...")
                            extractor.remove_hooks()
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                            # 回退到分块处理
                            chunked_extractor = ChunkedAttentionExtractor(self.model, chunk_size)
                            block_attn = chunked_extractor.extract_block_attention_chunked(
                                input_ids, blocks, layer_idx
                            )
                            block_attn = self.aggregator.normalize(block_attn, method=normalize)
                            block_attention_maps[layer_idx] = block_attn
                            continue
                        else:
                            raise e
                
                # 获取block attention
                block_attn = extractor.get_block_attention(layer_idx)
                if block_attn is not None:
                    block_attn = self.aggregator.normalize(block_attn, method=normalize)
                    block_attention_maps[layer_idx] = block_attn
                
                # 清理
                extractor.remove_hooks()
                extractor.clear_cache()
                del input_ids_device
                gc.collect()
                torch.cuda.empty_cache()
            
        result = {
            "blocks": blocks,
            "block_attention_maps": block_attention_maps,
            "layer_indices": layer_indices,
            "total_tokens": seq_len
        }
            
        return result
    
    def analyze_conversation_sequential(self, 
                                        messages: List[Dict],
                                        normalize: str = "row",
                                        layer_indices: List[int] = None) -> Dict[str, Any]:
        """
        顺序分析方法 - 完全顺序地处理每一层
        这是最保守的方法，最大程度减少显存使用
        
        Args:
            messages: 对话消息列表
            normalize: 归一化方法
            layer_indices: 要分析的层索引列表
            
        Returns:
            分析结果字典
        """
        if layer_indices is None:
            layer_indices = [self.layer_2_3, self.last_layer]
        
        # 1. 解析语义块
        blocks = self.parser.parse_messages(messages)
        print(f"Parsed {len(blocks)} semantic blocks")
        
        # 2. 构建完整输入
        if hasattr(self.tokenizer, 'apply_chat_template'):
            input_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        else:
            input_text = ""
            for msg in messages:
                input_text += f"<|{msg['role']}|>\n{msg['content']}\n"
        
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        blocks = self._recompute_block_indices(blocks, input_text)
        
        seq_len = input_ids.shape[1]
        print(f"Total tokens: {seq_len}")
        
        # 3. 逐层处理
        block_attention_maps = {}
        layers = self._get_model_layers()
        
        for layer_idx in layer_indices:
            print(f"Processing layer {layer_idx}...")
            
            # 创建用于存储结果的容器
            result_container = {"block_attention": None}
            
            def create_hook(blocks_ref, result_ref):
                def hook(module, input, output):
                    if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                        with torch.no_grad():
                            attn_weights = output[1]  # [batch, num_heads, seq_len, seq_len]
                            # 立即在GPU上聚合
                            attn_mean = attn_weights.mean(dim=1)[0]  # [seq_len, seq_len]
                            
                            # 计算block attention
                            num_blocks = len(blocks_ref)
                            block_attn = torch.zeros((num_blocks, num_blocks), 
                                                    device=attn_mean.device)
                            
                            for i, block_i in enumerate(blocks_ref):
                                for j, block_j in enumerate(blocks_ref):
                                    si = min(block_i.start_idx, seq_len - 1)
                                    ei = min(block_i.end_idx, seq_len)
                                    sj = min(block_j.start_idx, seq_len - 1)
                                    ej = min(block_j.end_idx, seq_len)
                                    
                                    if ei > si and ej > sj:
                                        sub = attn_mean[si:ei, sj:ej]
                                        block_attn[i, j] = sub.mean()
                            
                            result_ref["block_attention"] = block_attn.cpu().numpy()
                            
                            # 立即释放
                            del attn_weights, attn_mean, block_attn
                            
                return hook
            
            # 注册hook
            handle = layers[layer_idx].self_attn.register_forward_hook(
                create_hook(blocks, result_container)
            )
            
            try:
                # 前向传播
                input_device = self._get_layer_device(0)
                input_ids_device = input_ids.to(input_device)
                
                with torch.no_grad():
                    self.model(input_ids_device, output_attentions=False, use_cache=False)
                
                # 获取结果
                if result_container["block_attention"] is not None:
                    block_attn = self.aggregator.normalize(
                        result_container["block_attention"], 
                        method=normalize
                    )
                    block_attention_maps[layer_idx] = block_attn
                    
            finally:
                handle.remove()
                gc.collect()
                torch.cuda.empty_cache()
        
        return {
            "blocks": blocks,
            "block_attention_maps": block_attention_maps,
            "layer_indices": layer_indices,
            "total_tokens": seq_len
        }
    
    def _recompute_block_indices(self, blocks: List[SemanticBlock], 
                                  full_text: str) -> List[SemanticBlock]:
        """重新计算blocks在完整文本中的token索引"""
        current_pos = 0
        
        for block in blocks:
            # 查找block内容在全文中的位置
            search_content = block.content[:50] if len(block.content) > 50 else block.content
            content_start = full_text.find(search_content, current_pos)
            
            if content_start == -1:
                content_start = current_pos
            
            # 计算token索引
            prefix_tokens = self.tokenizer.encode(
                full_text[:content_start], 
                add_special_tokens=False
            )
            content_tokens = self.tokenizer.encode(
                block.content, 
                add_special_tokens=False
            )
            
            block.start_idx = len(prefix_tokens)
            block.end_idx = block.start_idx + len(content_tokens)
            block.token_count = len(content_tokens)
            
            current_pos = content_start + len(block.content)
            
        return blocks
    
    def visualize_results(self, 
                         result: Dict[str, Any],
                         save_dir: str = None):
        """可视化分析结果"""
        blocks = result["blocks"]
        attention_maps = result["block_attention_maps"]
        
        # 分别绘制每一层
        for layer_idx, block_attn in attention_maps.items():
            save_path = None
            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                save_path = f"{save_dir}/block_attention_layer_{layer_idx}.png"
            
            self.visualizer.plot_attention_map(
                block_attn, 
                blocks, 
                layer_idx,
                save_path=save_path
            )
        
        # 多层对比
        if len(attention_maps) > 1:
            save_path = f"{save_dir}/block_attention_comparison.png" if save_dir else None
            self.visualizer.plot_multi_layer_comparison(
                attention_maps,
                blocks,
                save_path=save_path
            )
            
    def analyze_from_json(self, json_path: str, sample_idx: int = 0) -> Dict[str, Any]:
        """从JSON文件加载并分析"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            messages = data[sample_idx]["messages"]
        else:
            messages = data["messages"]
            
        return self.analyze_conversation(messages)


# ============================================================================
# 7. 轻量级分析器 (使用采样方式)
# ============================================================================

class LightweightBlockAttentionAnalyzer:
    """
    轻量级Block Attention分析器
    使用采样方式来减少显存使用 - 只对每个block的部分token进行采样
    """
    
    def __init__(self, 
                 model_name_or_path: str,
                 device_map: str = "auto",
                 torch_dtype = torch.float16,
                 sample_ratio: float = 0.1,
                 max_samples_per_block: int = 50):
        """
        Args:
            sample_ratio: 每个block采样的token比例
            max_samples_per_block: 每个block最大采样token数
        """
        print(f"Loading model: {model_name_or_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True
        )
        self.model.eval()
        
        self.sample_ratio = sample_ratio
        self.max_samples_per_block = max_samples_per_block
        
        # 获取层数
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.num_layers = len(self.model.model.layers)
            self.layers = self.model.model.layers
        else:
            self.num_layers = len(self.model.transformer.h)
            self.layers = self.model.transformer.h
            
        self.parser = SemanticBlockParser(self.tokenizer)
        self.aggregator = BlockAttentionAggregator()
        self.visualizer = BlockAttentionVisualizer()
        
    def _get_sample_indices(self, blocks: List[SemanticBlock]) -> Dict[int, List[int]]:
        """为每个block生成采样索引"""
        sample_indices = {}
        
        for i, block in enumerate(blocks):
            num_tokens = block.end_idx - block.start_idx
            num_samples = min(
                max(1, int(num_tokens * self.sample_ratio)),
                self.max_samples_per_block
            )
            
            if num_samples >= num_tokens:
                indices = list(range(block.start_idx, block.end_idx))
            else:
                indices = np.linspace(
                    block.start_idx, 
                    block.end_idx - 1, 
                    num_samples, 
                    dtype=int
                ).tolist()
            
            sample_indices[i] = indices
            
        return sample_indices
    
    def analyze_with_sampling(self,
                             messages: List[Dict],
                             layer_indices: List[int] = None,
                             normalize: str = "row") -> Dict[str, Any]:
        """
        使用采样方式分析attention
        
        Args:
            messages: 消息列表
            layer_indices: 层索引列表
            normalize: 归一化方法
        """
        if layer_indices is None:
            layer_2_3 = int(self.num_layers * 2 / 3)
            layer_indices = [layer_2_3, self.num_layers - 1]
        
        # 解析blocks
        blocks = self.parser.parse_messages(messages)
        
        # 构建输入
        if hasattr(self.tokenizer, 'apply_chat_template'):
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            input_text = "".join([f"<|{m['role']}|>\n{m['content']}\n" for m in messages])
        
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        
        # 重新计算block索引
        current_pos = 0
        for block in blocks:
            search_content = block.content[:50]
            content_start = input_text.find(search_content, current_pos)
            if content_start == -1:
                content_start = current_pos
            
            prefix_tokens = self.tokenizer.encode(
                input_text[:content_start], add_special_tokens=False
            )
            content_tokens = self.tokenizer.encode(
                block.content, add_special_tokens=False
            )
            
            block.start_idx = len(prefix_tokens)
            block.end_idx = block.start_idx + len(content_tokens)
            block.token_count = len(content_tokens)
            current_pos = content_start + len(block.content)
        
        seq_len = input_ids.shape[1]
        print(f"Total tokens: {seq_len}, Blocks: {len(blocks)}")
        
        # 获取采样索引
        sample_indices = self._get_sample_indices(blocks)
        
        # 逐层分析
        block_attention_maps = {}
        
        for layer_idx in layer_indices:
            print(f"Processing layer {layer_idx} with sampling...")
            
            num_blocks = len(blocks)
            block_attention = np.zeros((num_blocks, num_blocks))
            
            result_container = {"attn": None}
            
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                    with torch.no_grad():
                        # 只保存采样位置的attention
                        attn = output[1].mean(dim=1)[0]  # [seq_len, seq_len]
                        result_container["attn"] = attn.cpu()
                        del output[1]
            
            handle = self.layers[layer_idx].self_attn.register_forward_hook(hook)
            
            try:
                input_device = next(self.model.parameters()).device
                with torch.no_grad():
                    self.model(input_ids.to(input_device), output_attentions=False, use_cache=False)
                
                if result_container["attn"] is not None:
                    attn = result_container["attn"].numpy()
                    
                    # 使用采样计算block attention
                    for i in range(num_blocks):
                        for j in range(num_blocks):
                            i_indices = sample_indices[i]
                            j_indices = sample_indices[j]
                            
                            # 过滤有效索引
                            i_indices = [idx for idx in i_indices if idx < seq_len]
                            j_indices = [idx for idx in j_indices if idx < seq_len]
                            
                            if i_indices and j_indices:
                                sub_attn = attn[np.ix_(i_indices, j_indices)]
                                block_attention[i, j] = sub_attn.mean()
                    
                    block_attention = self.aggregator.normalize(block_attention, normalize)
                    block_attention_maps[layer_idx] = block_attention
                    
            finally:
                handle.remove()
                gc.collect()
                torch.cuda.empty_cache()
        
        return {
            "blocks": blocks,
            "block_attention_maps": block_attention_maps,
            "layer_indices": layer_indices,
            "total_tokens": seq_len
        }


# ============================================================================
# 8. 简化版本 (不需要实际模型，用于演示)
# ============================================================================

class DemoBlockAttentionAnalyzer:
    """
    演示版本的Block Attention分析器
    不需要加载实际模型，使用模拟的attention数据
    """
    
    def __init__(self):
        self.parser = None
        self.aggregator = BlockAttentionAggregator(aggregation_method="mean")
        self.visualizer = BlockAttentionVisualizer()
        
    def parse_messages_simple(self, messages: List[Dict]) -> List[SemanticBlock]:
        """简化版本的消息解析"""
        blocks = []
        turn_idx = 0
        is_first_user = True
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                blocks.append(SemanticBlock(
                    block_type="system_init",
                    content=content,
                    role=role,
                    turn_idx=turn_idx,
                    token_count=len(content.split()) * 2
                ))
                
            elif role == "user":
                if "<tool_response>" in content:
                    match = re.search(r"<tool_response>(.*?)</tool_response>", content, re.DOTALL)
                    if match:
                        blocks.append(SemanticBlock(
                            block_type="tool_response",
                            content=match.group(1),
                            role=role,
                            turn_idx=turn_idx,
                            token_count=len(match.group(1).split()) * 2
                        ))
                else:
                    if is_first_user and blocks and blocks[-1].block_type == "system_init":
                        blocks[-1].content += "\n" + content
                        blocks[-1].token_count += len(content.split()) * 2
                    else:
                        blocks.append(SemanticBlock(
                            block_type="user_query",
                            content=content,
                            role=role,
                            turn_idx=turn_idx,
                            token_count=len(content.split()) * 2
                        ))
                    is_first_user = False
                    turn_idx += 1
                    
            elif role == "assistant":
                remaining = content
                
                for match in re.finditer(r"<think>(.*?)</think>", content, re.DOTALL):
                    blocks.append(SemanticBlock(
                        block_type="think",
                        content=match.group(1),
                        role=role,
                        turn_idx=turn_idx,
                        token_count=len(match.group(1).split()) * 2
                    ))
                    remaining = remaining.replace(match.group(0), "")
                
                for match in re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
                    blocks.append(SemanticBlock(
                        block_type="tool_call",
                        content=match.group(1),
                        role=role,
                        turn_idx=turn_idx,
                        token_count=len(match.group(1).split()) * 2
                    ))
                    remaining = remaining.replace(match.group(0), "")
                
                remaining = remaining.strip()
                if remaining:
                    blocks.append(SemanticBlock(
                        block_type="answer",
                        content=remaining,
                        role=role,
                        turn_idx=turn_idx,
                        token_count=len(remaining.split()) * 2
                    ))
        
        current_idx = 0
        for block in blocks:
            block.start_idx = current_idx
            block.end_idx = current_idx + block.token_count
            current_idx = block.end_idx
            
        return blocks
    
    def generate_mock_attention(self, 
                                 blocks: List[SemanticBlock],
                                 pattern: str = "realistic") -> np.ndarray:
        """生成模拟的block-level attention矩阵"""
        n = len(blocks)
        
        if pattern == "random":
            attn = np.random.rand(n, n)
            
        elif pattern == "diagonal":
            attn = np.eye(n) * 0.5 + np.random.rand(n, n) * 0.1
            
        elif pattern == "realistic":
            attn = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if j > i:
                        attn[i, j] = 0
                    elif i == j:
                        attn[i, j] = np.random.uniform(0.3, 0.5)
                    else:
                        query_type = blocks[i].block_type
                        key_type = blocks[j].block_type
                        
                        if key_type == "system_init":
                            attn[i, j] = np.random.uniform(0.2, 0.4)
                        elif query_type == "tool_call" and key_type == "tool_response":
                            attn[i, j] = np.random.uniform(0.15, 0.3)
                        elif query_type == "answer" and key_type == "think":
                            attn[i, j] = np.random.uniform(0.2, 0.35)
                        elif query_type == "think":
                            distance = i - j
                            attn[i, j] = np.random.uniform(0.1, 0.2) / (1 + distance * 0.1)
                        else:
                            distance = i - j
                            attn[i, j] = np.random.uniform(0.05, 0.15) / (1 + distance * 0.2)
        else:
            attn = np.random.rand(n, n)
            
        row_sums = attn.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        attn = attn / row_sums
        
        return attn
    
    def demo_analyze(self, messages: List[Dict]) -> Dict[str, Any]:
        """演示分析"""
        blocks = self.parse_messages_simple(messages)
        
        print(f"Parsed {len(blocks)} semantic blocks:")
        for i, block in enumerate(blocks):
            print(f"  [{i}] {block.block_type} (turn {block.turn_idx}, {block.token_count} tokens)")
        
        block_attention_layer_21 = self.generate_mock_attention(blocks, "realistic")
        block_attention_layer_31 = self.generate_mock_attention(blocks, "realistic")
        
        return {
            "blocks": blocks,
            "block_attention_maps": {
                21: block_attention_layer_21,
                31: block_attention_layer_31
            },
            "layer_indices": [21, 31],
            "total_tokens": sum(b.token_count for b in blocks)
        }
    
    def visualize_demo(self, result: Dict[str, Any]):
        """可视化演示结果"""
        blocks = result["blocks"]
        attention_maps = result["block_attention_maps"]
        
        for layer_idx, block_attn in attention_maps.items():
            self.visualizer.plot_attention_map(
                block_attn,
                blocks,
                layer_idx,
                title=f"Block Attention Map - Layer {layer_idx} (Demo)"
            )
        
        if len(attention_maps) > 1:
            self.visualizer.plot_multi_layer_comparison(attention_maps, blocks)


# ============================================================================
# 9. 使用示例
# ============================================================================

def create_example_messages():
    """创建示例对话消息"""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to tools. Use the search tool to find information."
        },
        {
            "role": "user",
            "content": "What is the weather like in Beijing today?"
        },
        {
            "role": "assistant",
            "content": """<think>The user is asking about the weather in Beijing. I need to use the weather tool to get current weather information.</think>
<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing"}}</tool_call>"""
        },
        {
            "role": "user",
            "content": """<tool_response>{"temperature": 25, "condition": "sunny", "humidity": 45}</tool_response>"""
        },
        {
            "role": "assistant",
            "content": """<think>I got the weather data. The temperature is 25°C and it's sunny with 45% humidity. I should provide a clear and helpful response.</think>
Based on the current weather data, Beijing is having a beautiful day! The temperature is 25°C (77°F) with sunny skies and a comfortable humidity level of 45%. It's a great day to be outdoors!"""
        },
        {
            "role": "user", 
            "content": "What about tomorrow?"
        },
        {
            "role": "assistant",
            "content": """<think>The user wants tomorrow's forecast. I need to call the weather tool again with a forecast parameter.</think>
<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing", "date": "tomorrow"}}</tool_call>"""
        },
        {
            "role": "user",
            "content": """<tool_response>{"temperature": 28, "condition": "partly cloudy", "humidity": 50}</tool_response>"""
        },
        {
            "role": "assistant",
            "content": """<think>Tomorrow's forecast shows slightly warmer weather at 28°C with partly cloudy conditions.</think>
Tomorrow in Beijing will be slightly warmer at 28°C (82°F) with partly cloudy skies. The humidity will be around 50%. Still a nice day, but you might want to carry a light jacket in case of any brief showers!"""
        }
    ]
    return messages


def main_demo():
    """演示主函数（不需要实际模型）"""
    print("=" * 60)
    print("Block Attention Map Visualization - Demo Mode")
    print("=" * 60)
    
    messages = create_example_messages()
    demo_analyzer = DemoBlockAttentionAnalyzer()
    result = demo_analyzer.demo_analyze(messages)
    demo_analyzer.visualize_demo(result)
    
    return result


def main_real(model_path: str, 
              json_path: str = None,
              use_4bit: bool = False,
              use_8bit: bool = False,
              use_sampling: bool = False,
              use_chunked: bool = False):
    """
    使用真实模型的主函数
    
    Args:
        model_path: 模型路径
        json_path: 可选的JSON数据路径
        use_4bit: 是否使用4bit量化
        use_8bit: 是否使用8bit量化
        use_sampling: 是否使用采样方式（更省显存）
        use_chunked: 是否使用分块处理
    """
    print("=" * 60)
    print("Block Attention Map Visualization - Real Model Mode")
    print("=" * 60)
    
    # 打印GPU信息
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB")
    
    # 准备消息
    if json_path:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            messages = data[0]["messages"]
        else:
            messages = data["messages"]
    else:
        messages = create_example_messages()
    
    if use_sampling:
        # 使用采样分析器
        analyzer = LightweightBlockAttentionAnalyzer(
            model_name_or_path=model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            sample_ratio=0.1,
            max_samples_per_block=50
        )
        result = analyzer.analyze_with_sampling(messages)
    else:
        # 使用标准分析器
        analyzer = BlockAttentionAnalyzer(
            model_name_or_path=model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=use_4bit,
            load_in_8bit=use_8bit
        )
        
        if use_chunked:
            result = analyzer.analyze_conversation(
                messages, 
                use_chunked=True, 
                chunk_size=2048
            )
        else:
            # 使用顺序分析方法（最省显存）
            result = analyzer.analyze_conversation_sequential(messages)
    
    # 可视化
    analyzer.visualizer = BlockAttentionVisualizer()
    
    blocks = result["blocks"]
    attention_maps = result["block_attention_maps"]
    
    for layer_idx, block_attn in attention_maps.items():
        analyzer.visualizer.plot_attention_map(
            block_attn, blocks, layer_idx,
            save_path=f"./attention_maps/block_attention_layer_{layer_idx}.png"
        )
    
    if len(attention_maps) > 1:
        analyzer.visualizer.plot_multi_layer_comparison(
            attention_maps, blocks,
            save_path="./attention_maps/block_attention_comparison.png"
        )
    
    return result


# ============================================================================
# 运行入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Block Attention Map Visualization (Optimized)")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "real"],
                       help="运行模式: demo (演示模式) 或 real (需要实际模型)")
    parser.add_argument("--model", type=str, default=None,
                       help="模型路径 (仅在real模式下需要)")
    parser.add_argument("--json", type=str, default=None,
                       help="输入JSON文件路径")
    parser.add_argument("--use-4bit", action="store_true",
                       help="使用4bit量化以节省显存")
    parser.add_argument("--use-8bit", action="store_true",
                       help="使用8bit量化以节省显存")
    parser.add_argument("--use-sampling", action="store_true",
                       help="使用采样方式分析（最省显存）")
    parser.add_argument("--use-chunked", action="store_true",
                       help="使用分块处理长序列")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        result = main_demo()
    else:
        if not args.model:
            print("Error: --model is required in real mode")
            exit(1)
        result = main_real(
            args.model, 
            args.json,
            use_4bit=args.use_4bit,
            use_8bit=args.use_8bit,
            use_sampling=args.use_sampling,
            use_chunked=args.use_chunked
        )