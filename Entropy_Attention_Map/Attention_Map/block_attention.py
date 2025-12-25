"""
Block Attention Map Visualization Tool
用于可视化Agent对话流中不同语义块之间的注意力关系
"""

import json
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
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
# 3. 注意力提取器 (使用Hook)
# ============================================================================

class AttentionExtractor:
    """使用register_hook方式提取指定层的注意力权重"""
    
    def __init__(self, model, layer_indices: List[int]):
        """
        Args:
            model: HuggingFace模型
            layer_indices: 需要提取的层索引列表，如 [21, 31] 表示2/3层和最后一层
        """
        self.model = model
        self.layer_indices = layer_indices
        self.attention_cache: Dict[int, torch.Tensor] = {}
        self.hooks = []
        
    def _create_hook(self, layer_idx: int):
        """创建attention hook"""
        def hook(module, input, output):
            # output通常是 (attn_output, attn_weights, past_key_value)
            # 但不同模型结构可能不同
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]  # [batch, num_heads, seq_len, seq_len]
                if attn_weights is not None:
                    self.attention_cache[layer_idx] = attn_weights.detach().cpu()
        return hook
    
    def register_hooks(self):
        """注册hooks到指定层"""
        # 清除旧的hooks
        self.remove_hooks()
        self.attention_cache.clear()
        
        # 获取模型的attention层
        # 适配不同的模型架构
        if hasattr(self.model, 'model'):
            # LLaMA, Mistral等
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer'):
            # GPT-2等
            layers = self.model.transformer.h
        else:
            raise ValueError("Unsupported model architecture")
        
        for layer_idx in self.layer_indices:
            if layer_idx < len(layers):
                attn_module = layers[layer_idx].self_attn
                hook = attn_module.register_forward_hook(self._create_hook(layer_idx))
                self.hooks.append(hook)
                print(f"Registered hook on layer {layer_idx}")
                
    def remove_hooks(self):
        """移除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def get_attention_weights(self) -> Dict[int, torch.Tensor]:
        """获取缓存的注意力权重"""
        return self.attention_cache


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
        
        for i, block_i in enumerate(blocks):  # Query blocks
            for j, block_j in enumerate(blocks):  # Key blocks
                # 提取block_i对block_j的注意力子矩阵
                start_i, end_i = block_i.start_idx, block_i.end_idx
                start_j, end_j = block_j.start_idx, block_j.end_idx
                
                # 边界检查
                if end_i > attention_matrix.shape[0] or end_j > attention_matrix.shape[1]:
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
# 6. 主Pipeline类
# ============================================================================

class BlockAttentionAnalyzer:
    """Block Attention分析的主类，整合所有组件"""
    
    def __init__(self, 
                 model_name_or_path: str,
                 device_map: str = "auto",
                 torch_dtype = torch.float16):
        """
        Args:
            model_name_or_path: 模型路径或HuggingFace模型名
            device_map: 设备映射策略
            torch_dtype: 数据类型
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
            attn_implementation="eager",  # 必须使用eager以获取attention weights
            output_attentions=True
        )
        self.model.eval()
        
        # 计算层索引
        self.num_layers = self._get_num_layers()
        self.layer_2_3 = int(self.num_layers * 2 / 3)
        self.last_layer = self.num_layers - 1
        
        print(f"Model loaded. Total layers: {self.num_layers}")
        print(f"Will extract attention from layers: {self.layer_2_3} (2/3) and {self.last_layer} (last)")
        
        # 初始化组件
        self.parser = SemanticBlockParser(self.tokenizer)
        self.aggregator = BlockAttentionAggregator(aggregation_method="mean")
        self.visualizer = BlockAttentionVisualizer()
        
        # 初始化attention extractor
        self.extractor = AttentionExtractor(
            self.model, 
            [self.layer_2_3, self.last_layer]
        )
        
    def _get_num_layers(self) -> int:
        """获取模型层数"""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return len(self.model.model.layers)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return len(self.model.transformer.h)
        else:
            raise ValueError("Cannot determine number of layers")
    
    def analyze_conversation(self, 
                            messages: List[Dict],
                            normalize: str = "row",
                            return_raw: bool = False) -> Dict[str, Any]:
        """
        分析单个对话的Block Attention
        
        Args:
            messages: 对话消息列表
            normalize: 归一化方法
            return_raw: 是否返回原始attention矩阵
            
        Returns:
            分析结果字典
        """
        # 1. 解析语义块
        blocks = self.parser.parse_messages(messages)
        print(f"Parsed {len(blocks)} semantic blocks:")
        for i, block in enumerate(blocks):
            print(f"  [{i}] {block.block_type} (turn {block.turn_idx}): {block.content[:50]}...")
        
        # 2. 构建完整输入
        # 使用chat template如果可用
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
        
        print(f"Total tokens: {input_ids.shape[1]}")
        
        # 3. 注册hooks并前向传播
        self.extractor.register_hooks()
        
        with torch.no_grad():
            input_ids = input_ids.to(self.model.device)
            outputs = self.model(input_ids, output_attentions=True)
        
        # 4. 获取attention weights
        attention_weights = self.extractor.get_attention_weights()
        
        # 如果hook没有捕获到，尝试从outputs获取
        if not attention_weights and hasattr(outputs, 'attentions') and outputs.attentions:
            print("Getting attention from model outputs...")
            for layer_idx in [self.layer_2_3, self.last_layer]:
                if layer_idx < len(outputs.attentions):
                    attention_weights[layer_idx] = outputs.attentions[layer_idx].cpu()
        
        self.extractor.remove_hooks()
        
        # 5. 聚合为Block-level attention
        block_attention_maps = {}
        
        for layer_idx, attn in attention_weights.items():
            # attn shape: [batch, num_heads, seq_len, seq_len]
            attn_np = attn[0].numpy()  # 取第一个batch
            
            # 聚合
            block_attn = self.aggregator.aggregate(attn_np, blocks, head_aggregation="mean")
            block_attn = self.aggregator.normalize(block_attn, method=normalize)
            
            block_attention_maps[layer_idx] = block_attn
            
        result = {
            "blocks": blocks,
            "block_attention_maps": block_attention_maps,
            "layer_indices": list(attention_weights.keys()),
            "total_tokens": input_ids.shape[1]
        }
        
        if return_raw:
            result["raw_attention"] = {k: v.numpy() for k, v in attention_weights.items()}
            
        return result
    
    def _recompute_block_indices(self, blocks: List[SemanticBlock], 
                                  full_text: str) -> List[SemanticBlock]:
        """重新计算blocks在完整文本中的token索引"""
        current_pos = 0
        
        for block in blocks:
            # 查找block内容在全文中的位置
            content_start = full_text.find(block.content[:50], current_pos)  # 用前50字符查找
            
            if content_start == -1:
                # 找不到精确匹配，使用估计位置
                content_start = current_pos
            
            # 计算token索引
            prefix_tokens = self.tokenizer.encode(full_text[:content_start], add_special_tokens=False)
            content_tokens = self.tokenizer.encode(block.content, add_special_tokens=False)
            
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
# 7. 简化版本 (不需要实际模型，用于演示)
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
                    token_count=len(content.split()) * 2  # 估计token数
                ))
                
            elif role == "user":
                if "<tool_response>" in content:
                    # 提取tool_response
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
                # 解析think, tool_call, answer
                remaining = content
                
                # Think
                for match in re.finditer(r"<think>(.*?)</think>", content, re.DOTALL):
                    blocks.append(SemanticBlock(
                        block_type="think",
                        content=match.group(1),
                        role=role,
                        turn_idx=turn_idx,
                        token_count=len(match.group(1).split()) * 2
                    ))
                    remaining = remaining.replace(match.group(0), "")
                
                # Tool call
                for match in re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
                    blocks.append(SemanticBlock(
                        block_type="tool_call",
                        content=match.group(1),
                        role=role,
                        turn_idx=turn_idx,
                        token_count=len(match.group(1).split()) * 2
                    ))
                    remaining = remaining.replace(match.group(0), "")
                
                # Answer
                remaining = remaining.strip()
                if remaining:
                    blocks.append(SemanticBlock(
                        block_type="answer",
                        content=remaining,
                        role=role,
                        turn_idx=turn_idx,
                        token_count=len(remaining.split()) * 2
                    ))
        
        # 设置索引
        current_idx = 0
        for block in blocks:
            block.start_idx = current_idx
            block.end_idx = current_idx + block.token_count
            current_idx = block.end_idx
            
        return blocks
    
    def generate_mock_attention(self, 
                                 blocks: List[SemanticBlock],
                                 pattern: str = "realistic") -> np.ndarray:
        """
        生成模拟的block-level attention矩阵
        
        Args:
            pattern: 
                - "realistic": 模拟真实的attention模式
                - "random": 随机attention
                - "diagonal": 对角线模式
        """
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
                        # Causal: 不能看未来
                        attn[i, j] = 0
                    elif i == j:
                        # 自注意力
                        attn[i, j] = np.random.uniform(0.3, 0.5)
                    else:
                        # 基于block类型设置attention强度
                        query_type = blocks[i].block_type
                        key_type = blocks[j].block_type
                        
                        # System prompt 通常被强烈关注
                        if key_type == "system_init":
                            attn[i, j] = np.random.uniform(0.2, 0.4)
                            
                        # Tool call 关注 tool response
                        elif query_type == "tool_call" and key_type == "tool_response":
                            attn[i, j] = np.random.uniform(0.15, 0.3)
                            
                        # Answer 关注 think
                        elif query_type == "answer" and key_type == "think":
                            attn[i, j] = np.random.uniform(0.2, 0.35)
                            
                        # Think 关注所有之前的内容
                        elif query_type == "think":
                            distance = i - j
                            attn[i, j] = np.random.uniform(0.1, 0.2) / (1 + distance * 0.1)
                            
                        else:
                            distance = i - j
                            attn[i, j] = np.random.uniform(0.05, 0.15) / (1 + distance * 0.2)
        else:
            attn = np.random.rand(n, n)
            
        # 归一化每行
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
        
        # 生成模拟的attention
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
# 8. 使用示例
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
    
    # 创建示例消息
    messages = create_example_messages()
    
    # 使用演示分析器
    demo_analyzer = DemoBlockAttentionAnalyzer()
    
    # 分析
    result = demo_analyzer.demo_analyze(messages)
    
    # 可视化
    demo_analyzer.visualize_demo(result)
    
    return result


def main_real(model_path: str, json_path: str = None):
    """
    使用真实模型的主函数
    
    Args:
        model_path: 模型路径
        json_path: 可选的JSON数据路径
    """
    print("=" * 60)
    print("Block Attention Map Visualization - Real Model Mode")
    print("=" * 60)
    
    # 初始化分析器
    analyzer = BlockAttentionAnalyzer(
        model_name_or_path=model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # 准备消息
    if json_path:
        result = analyzer.analyze_from_json(json_path)
    else:
        messages = create_example_messages()
        result = analyzer.analyze_conversation(messages)
    
    # 可视化
    analyzer.visualize_results(result, save_dir="./attention_maps")
    
    return result


# ============================================================================
# 运行入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Block Attention Map Visualization")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "real"],
                       help="运行模式: demo (演示模式，不需要模型) 或 real (需要实际模型)")
    parser.add_argument("--model", type=str, default=None,
                       help="模型路径 (仅在real模式下需要)")
    parser.add_argument("--json", type=str, default=None,
                       help="输入JSON文件路径")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        result = main_demo()
    else:
        if not args.model:
            print("Error: --model is required in real mode")
            exit(1)
        result = main_real(args.model, args.json)