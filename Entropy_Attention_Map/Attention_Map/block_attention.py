"""
Block Attention Map Visualization Tool
======================================
使用Hook机制提取指定层的注意力权重，聚合为语义块级别的Attention Map

修复版本 - 2024
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from collections import defaultdict
import warnings
import gc

warnings.filterwarnings('ignore')


# ==================== 数据结构定义 ====================

@dataclass
class SemanticBlock:
    """语义块数据结构"""
    block_type: str  # system_init, think, tool_call, tool_response, answer
    content: str
    start_idx: int
    end_idx: int
    role: str
    turn_idx: int  # 对话轮次
    
    def __repr__(self):
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Block({self.block_type}, tokens={self.start_idx}:{self.end_idx}, turn={self.turn_idx})"
    
    @property
    def length(self) -> int:
        return self.end_idx - self.start_idx


# ==================== 显存高效的Attention提取器 ====================

class SDPAAttentionExtractor:
    """
    针对使用SDPA/FlashAttention的模型
    在模型输出后，使用保存的hidden states重新计算attention
    
    显存优化策略:
    1. 分块计算Q*K^T
    2. 使用float16
    3. 边计算边移到CPU
    4. 及时释放中间结果
    """
    
    def __init__(self, model, target_layers: List[int], chunk_size: int = 1024):
        """
        Args:
            model: HuggingFace模型
            target_layers: 需要提取的层索引列表
            chunk_size: 分块大小，越小越省显存，但更慢
        """
        self.model = model
        self.target_layers = target_layers
        self.chunk_size = chunk_size
        self.hidden_states_cache: Dict[int, torch.Tensor] = {}
        self.hooks = []
        
        # 获取模型配置
        self.config = model.config
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = getattr(self.config, 'num_key_value_heads', self.num_heads)
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        
    def _get_layers(self):
        """获取模型的layers模块"""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        else:
            raise ValueError(f"Unsupported model structure: {type(self.model)}")
        
    def register_hooks(self):
        """注册hooks来捕获每层输入的hidden states"""
        self.clear()
        
        layers = self._get_layers()
        
        for layer_idx in self.target_layers:
            if layer_idx >= len(layers):
                print(f"  Warning: layer {layer_idx} out of range, skipping")
                continue
                
            layer = layers[layer_idx]
            
            def make_hook(idx):
                def hook_fn(module, args, kwargs):
                    # 捕获输入hidden states
                    if len(args) > 0:
                        # 只保存需要的，立即detach并clone
                        hidden = args[0].detach().clone()
                        self.hidden_states_cache[idx] = hidden
                    return None
                return hook_fn
            
            # 注册到整个layer
            hook = layer.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)
            self.hooks.append(hook)
        
        print(f"  Registered {len(self.hooks)} layer hooks for layers {self.target_layers}")

    def _compute_attention_memory_efficient(
        self,
        attn_module,
        hidden_states: torch.Tensor,
        apply_input_norm: bool = True,
        input_layernorm=None
    ) -> torch.Tensor:
        """
        显存高效版本 - 分块计算attention
        
        Args:
            attn_module: attention模块
            hidden_states: 输入hidden states
            apply_input_norm: 是否应用input layernorm
            input_layernorm: layernorm模块
            
        Returns:
            attention: [seq_len, seq_len] 已平均的attention矩阵
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        chunk_size = self.chunk_size
        
        # 动态调整chunk_size以适应显存
        if seq_len > 16000:
            chunk_size = min(chunk_size, 512)
        elif seq_len > 8000:
            chunk_size = min(chunk_size, 1024)
        
        print(f"    Computing attention: seq_len={seq_len}, chunk_size={chunk_size}")
        
        with torch.no_grad():
            # 应用input layernorm（如果需要）
            if apply_input_norm and input_layernorm is not None:
                hidden_states = input_layernorm(hidden_states)
            
            # 使用autocast进行混合精度计算
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # 1. 计算Q, K
                q = attn_module.q_proj(hidden_states)  # [B, S, H*D]
                k = attn_module.k_proj(hidden_states)
                
                # 释放hidden_states
                del hidden_states
                
                # 2. Reshape
                q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
                k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
                
                # 3. 处理GQA (Grouped Query Attention)
                if self.num_kv_heads != self.num_heads:
                    n_rep = self.num_heads // self.num_kv_heads
                    k = k.unsqueeze(3).expand(-1, -1, -1, n_rep, -1)
                    k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
                
                # 4. 转置为attention计算格式
                q = q.transpose(1, 2).contiguous()  # [B, H, S, D]
                k = k.transpose(1, 2).contiguous()  # [B, H, S, D]
                
                scale = self.head_dim ** -0.5
                
                # 5. 计算 K^T
                k_t = k.transpose(-2, -1).contiguous()  # [B, H, D, S]
                del k
                torch.cuda.empty_cache()
            
            # 6. 分块计算，结果直接聚合到CPU
            result = torch.zeros(seq_len, seq_len, dtype=torch.float32)
            n_chunks = (seq_len + chunk_size - 1) // chunk_size
            
            for c in range(n_chunks):
                c_start = c * chunk_size
                c_end = min(c_start + chunk_size, seq_len)
                
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    # 取query chunk: [B, H, chunk_len, D]
                    q_c = q[:, :, c_start:c_end, :]
                    
                    # 计算attention scores: [B, H, chunk_len, S]
                    scores = torch.matmul(q_c, k_t) * scale
                    
                    del q_c
                
                # Causal mask: 位置i只能看到[0, i]
                row_indices = torch.arange(c_start, c_end, device=device).unsqueeze(1)
                col_indices = torch.arange(seq_len, device=device).unsqueeze(0)
                causal_mask = col_indices > row_indices  # [chunk, S]
                
                scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                
                # Softmax (用float32保证数值稳定)
                attn_probs = F.softmax(scores.float(), dim=-1)
                
                # 对batch和heads取平均，移到CPU
                attn_avg = attn_probs.mean(dim=(0, 1)).cpu()  # [chunk_len, S]
                result[c_start:c_end, :] = attn_avg
                
                # 清理
                del scores, attn_probs, attn_avg, causal_mask
                
                if (c + 1) % 10 == 0 or c == n_chunks - 1:
                    torch.cuda.empty_cache()
                    print(f"      Chunk {c+1}/{n_chunks} done")
            
            del q, k_t
            torch.cuda.empty_cache()
            gc.collect()
        
        return result
  
    def compute_attention_weights(self) -> Dict[int, torch.Tensor]:
        """
        计算所有目标层的attention weights
        
        Returns:
            Dict[layer_idx, attention_matrix]: 每层的attention矩阵 [seq, seq]
        """
        attention_weights = {}
        layers = self._get_layers()
        
        for layer_idx in self.target_layers:
            if layer_idx not in self.hidden_states_cache:
                print(f"  Warning: No hidden states cached for layer {layer_idx}")
                continue
            
            print(f"  Processing layer {layer_idx}...")
            
            hidden_states = self.hidden_states_cache[layer_idx]
            layer = layers[layer_idx]
            
            # 获取input_layernorm（如果存在）
            input_layernorm = None
            if hasattr(layer, 'input_layernorm'):
                input_layernorm = layer.input_layernorm
            elif hasattr(layer, 'ln_1'):
                input_layernorm = layer.ln_1
            
            # 获取attention模块
            if hasattr(layer, 'self_attn'):
                attn_module = layer.self_attn
            elif hasattr(layer, 'attention'):
                attn_module = layer.attention
            elif hasattr(layer, 'attn'):
                attn_module = layer.attn
            else:
                print(f"    Cannot find attention module in layer {layer_idx}")
                continue
            
            try:
                attn = self._compute_attention_memory_efficient(
                    attn_module,
                    hidden_states,
                    apply_input_norm=True,
                    input_layernorm=input_layernorm
                )
                
                attention_weights[layer_idx] = attn
                print(f"    Layer {layer_idx}: shape {attn.shape}")
                
            except Exception as e:
                print(f"    Error at layer {layer_idx}: {e}")
                import traceback
                traceback.print_exc()
            
            # 处理完一层后清理缓存
            del self.hidden_states_cache[layer_idx]
            torch.cuda.empty_cache()
            gc.collect()
        
        return attention_weights
    
    def clear(self):
        """清除hooks和缓存"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.hidden_states_cache.clear()
        torch.cuda.empty_cache()
        gc.collect()


# ==================== 语义块解析器 ====================

class SemanticBlockParser:
    """解析消息列表，提取语义块并精确计算token位置"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def parse_messages_with_positions(
        self, 
        messages: List[Dict], 
        input_ids: torch.Tensor
    ) -> List[SemanticBlock]:
        """
        解析消息并计算每个块的精确token位置
        
        Args:
            messages: 消息列表
            input_ids: 实际的input_ids tensor
            
        Returns:
            语义块列表
        """
        if input_ids.dim() == 2:
            input_ids = input_ids[0]
        
        total_tokens = len(input_ids)
        full_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        
        blocks = []
        
        # 构建消息到token的映射
        processed_messages = self._preprocess_messages(messages)
        
        # 使用累积编码来确定每个消息的token范围
        current_pos = 0
        
        for turn_idx, msg in enumerate(processed_messages):
            role = msg['role']
            content = msg['content']
            
            # 估算这个消息的token范围
            msg_tokens = self._estimate_message_tokens(content, role, turn_idx == 0)
            end_pos = min(current_pos + msg_tokens, total_tokens)
            
            if role == 'system_init':
                sub_blocks = self._parse_content_into_blocks(
                    content, current_pos, end_pos, turn_idx, 'system'
                )
                blocks.extend(sub_blocks)
                
            elif role == 'assistant':
                sub_blocks = self._parse_content_into_blocks(
                    content, current_pos, end_pos, turn_idx, 'assistant'
                )
                blocks.extend(sub_blocks)
                
            elif role == 'user':
                block_type = 'tool_response' if self._is_tool_response(content) else 'user_message'
                blocks.append(SemanticBlock(
                    block_type=block_type,
                    content=self._truncate(content),
                    start_idx=current_pos,
                    end_idx=end_pos,
                    role='user',
                    turn_idx=turn_idx
                ))
            
            current_pos = end_pos
        
        # 确保最后一个block覆盖到末尾
        if blocks and blocks[-1].end_idx < total_tokens:
            blocks[-1].end_idx = total_tokens
        
        # 验证并修复block边界
        blocks = self._validate_and_fix_blocks(blocks, total_tokens)
        
        return blocks
    
    def _preprocess_messages(self, messages: List[Dict]) -> List[Dict]:
        """预处理消息，合并system和第一个user"""
        processed = []
        system_content = ""
        first_user_processed = False
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                system_content = content
            elif role == 'user' and not first_user_processed:
                combined = system_content + "\n\n" + content if system_content else content
                processed.append({'role': 'system_init', 'content': combined})
                first_user_processed = True
            else:
                processed.append(msg)
                
        return processed
    
    def _estimate_message_tokens(self, content: str, role: str, is_first: bool) -> int:
        """估算消息的token数量"""
        tokens = self.tokenizer.encode(content, add_special_tokens=False)
        # 添加一些额外的tokens用于角色标记等
        overhead = 10 if is_first else 5
        return len(tokens) + overhead
    
    def _parse_content_into_blocks(
        self,
        content: str,
        start_pos: int,
        end_pos: int,
        turn_idx: int,
        role: str
    ) -> List[SemanticBlock]:
        """将内容解析为多个语义块"""
        blocks = []
        total_tokens = end_pos - start_pos
        
        if total_tokens <= 0:
            return blocks
        
        # 定义匹配模式
        patterns = [
            (r'<think>(.*?)</think>', 'think'),
            (r'<tool_call>(.*?)</tool_call>', 'tool_call'),
            (r'<tool_calls>(.*?)</tool_calls>', 'tool_call'),
        ]
        
        # 找所有匹配
        matches = []
        for pattern, block_type in patterns:
            for m in re.finditer(pattern, content, re.DOTALL):
                matches.append({
                    'char_start': m.start(),
                    'char_end': m.end(),
                    'text': m.group(0),
                    'type': block_type
                })
        
        matches.sort(key=lambda x: x['char_start'])
        
        if not matches:
            # 没有特殊标签
            block_type = 'system_init' if role == 'system' else 'answer'
            blocks.append(SemanticBlock(
                block_type=block_type,
                content=self._truncate(content),
                start_idx=start_pos,
                end_idx=end_pos,
                role=role,
                turn_idx=turn_idx
            ))
        else:
            # 按字符位置比例分配token
            content_len = max(len(content), 1)
            current_char = 0
            current_token = start_pos
            
            for match in matches:
                # 匹配前的文本
                if match['char_start'] > current_char:
                    text_before = content[current_char:match['char_start']].strip()
                    if text_before:
                        char_ratio = (match['char_start'] - current_char) / content_len
                        token_count = max(1, int(char_ratio * total_tokens))
                        token_end = min(current_token + token_count, end_pos)
                        
                        blocks.append(SemanticBlock(
                            block_type='answer' if role == 'assistant' else 'system_init',
                            content=self._truncate(text_before),
                            start_idx=current_token,
                            end_idx=token_end,
                            role=role,
                            turn_idx=turn_idx
                        ))
                        current_token = token_end
                
                # 匹配的块
                char_ratio = (match['char_end'] - match['char_start']) / content_len
                token_count = max(1, int(char_ratio * total_tokens))
                token_end = min(current_token + token_count, end_pos)
                
                blocks.append(SemanticBlock(
                    block_type=match['type'],
                    content=self._truncate(match['text']),
                    start_idx=current_token,
                    end_idx=token_end,
                    role=role,
                    turn_idx=turn_idx
                ))
                current_token = token_end
                current_char = match['char_end']
            
            # 剩余文本
            if current_char < len(content) and current_token < end_pos:
                remaining = content[current_char:].strip()
                if remaining:
                    blocks.append(SemanticBlock(
                        block_type='answer' if role == 'assistant' else 'system_init',
                        content=self._truncate(remaining),
                        start_idx=current_token,
                        end_idx=end_pos,
                        role=role,
                        turn_idx=turn_idx
                    ))
        
        return blocks
    
    def _is_tool_response(self, content: str) -> bool:
        """检查内容是否是tool response"""
        return '<tool_response>' in content or '<tool_result>' in content
    
    def _truncate(self, text: str, max_len: int = 80) -> str:
        """截断文本"""
        text = text.replace('\n', ' ').strip()
        return text[:max_len] + "..." if len(text) > max_len else text
    
    def _validate_and_fix_blocks(self, blocks: List[SemanticBlock], total_tokens: int) -> List[SemanticBlock]:
        """验证并修复block边界"""
        if not blocks:
            return blocks
        
        fixed_blocks = []
        
        for i, block in enumerate(blocks):
            # 确保索引在有效范围内
            block.start_idx = max(0, min(block.start_idx, total_tokens - 1))
            block.end_idx = max(block.start_idx + 1, min(block.end_idx, total_tokens))
            
            # 确保不与前一个block重叠
            if fixed_blocks and block.start_idx < fixed_blocks[-1].end_idx:
                block.start_idx = fixed_blocks[-1].end_idx
                if block.start_idx >= block.end_idx:
                    block.end_idx = block.start_idx + 1
            
            if block.end_idx > block.start_idx:
                fixed_blocks.append(block)
        
        return fixed_blocks


# ==================== Block Attention计算 ====================

class BlockAttentionCalculator:
    """计算Block级别的Attention矩阵"""
    
    def __init__(self, blocks: List[SemanticBlock], total_tokens: int):
        self.blocks = blocks
        self.total_tokens = total_tokens
        
    def compute_block_attention(self, token_attention: torch.Tensor) -> np.ndarray:
        """
        将token级别的attention矩阵聚合为block级别
        
        Args:
            token_attention: [seq_len, seq_len] 的token级attention矩阵
            
        Returns:
            block_attention: [num_blocks, num_blocks] 的block级attention矩阵
        """
        num_blocks = len(self.blocks)
        block_attention = np.zeros((num_blocks, num_blocks))
        
        if torch.is_tensor(token_attention):
            token_attention_np = token_attention.cpu().numpy()
        else:
            token_attention_np = token_attention
        
        seq_len = token_attention_np.shape[0]
        
        for i, query_block in enumerate(self.blocks):
            for j, key_block in enumerate(self.blocks):
                # 因果性：query只能attend到位置 <= 自己的key
                if query_block.start_idx >= key_block.start_idx:
                    q_start = min(query_block.start_idx, seq_len - 1)
                    q_end = min(query_block.end_idx, seq_len)
                    k_start = min(key_block.start_idx, seq_len - 1)
                    k_end = min(key_block.end_idx, seq_len)
                    
                    if q_end > q_start and k_end > k_start:
                        sub_matrix = token_attention_np[q_start:q_end, k_start:k_end]
                        # 使用mean聚合
                        block_attention[i, j] = np.mean(sub_matrix)
        
        return block_attention
    
    def compute_block_attention_weighted(
        self, 
        token_attention: torch.Tensor,
        weight_by_length: bool = True
    ) -> np.ndarray:
        """
        加权版本的block attention计算
        
        Args:
            token_attention: token级attention
            weight_by_length: 是否按block长度加权
        """
        num_blocks = len(self.blocks)
        block_attention = np.zeros((num_blocks, num_blocks))
        
        if torch.is_tensor(token_attention):
            token_attention_np = token_attention.cpu().numpy()
        else:
            token_attention_np = token_attention
        
        seq_len = token_attention_np.shape[0]
        
        for i, query_block in enumerate(self.blocks):
            # 归一化因子：这个query block的总attention
            q_start = min(query_block.start_idx, seq_len - 1)
            q_end = min(query_block.end_idx, seq_len)
            
            total_attention = 0.0
            block_attentions = []
            
            for j, key_block in enumerate(self.blocks):
                if query_block.start_idx >= key_block.start_idx:
                    k_start = min(key_block.start_idx, seq_len - 1)
                    k_end = min(key_block.end_idx, seq_len)
                    
                    if q_end > q_start and k_end > k_start:
                        sub_matrix = token_attention_np[q_start:q_end, k_start:k_end]
                        attn_sum = np.sum(sub_matrix)
                        block_attentions.append((j, attn_sum))
                        total_attention += attn_sum
            
            # 归一化
            if total_attention > 0:
                for j, attn_sum in block_attentions:
                    block_attention[i, j] = attn_sum / total_attention
        
        return block_attention
    
    def get_block_labels(self) -> List[str]:
        """生成block标签"""
        labels = []
        type_abbrev = {
            'system_init': 'SYS',
            'think': 'THK',
            'tool_call': 'CALL',
            'tool_response': 'RESP',
            'answer': 'ANS',
            'user_message': 'USR'
        }
        for block in self.blocks:
            abbrev = type_abbrev.get(block.block_type, block.block_type[:4].upper())
            labels.append(f"{abbrev}_{block.turn_idx}")
        return labels


# ==================== 可视化器 ====================

class BlockAttentionVisualizer:
    """Block Attention Map可视化"""
    
    def __init__(self, figsize: Tuple[int, int] = (14, 6)):
        self.figsize = figsize
        self.type_colors = {
            'system_init': '#E8E8E8',
            'think': '#FFE4B5',
            'tool_call': '#98FB98',
            'tool_response': '#ADD8E6',
            'answer': '#DDA0DD',
            'user_message': '#F0E68C'
        }
        
    def plot_attention_map(
        self, 
        block_attention: np.ndarray,
        blocks: List[SemanticBlock],
        layer_idx: int,
        title: str = "Block Attention Map",
        ax=None,
        show_values: bool = True,
        cmap: str = 'Blues'
    ):
        """绘制单个attention map"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制热力图
        im = ax.imshow(block_attention, cmap=cmap, aspect='auto', vmin=0)
        
        # 生成标签
        labels = []
        type_abbrev = {
            'system_init': 'SYS',
            'think': 'THK', 
            'tool_call': 'CALL',
            'tool_response': 'RESP',
            'answer': 'ANS',
            'user_message': 'USR'
        }
        for block in blocks:
            abbrev = type_abbrev.get(block.block_type, block.block_type[:4])
            labels.append(f"{abbrev}_{block.turn_idx}")
        
        # 设置刻度
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        
        # 添加数值标注
        if show_values:
            max_val = block_attention.max() if block_attention.max() > 0 else 1
            for i in range(len(blocks)):
                for j in range(len(blocks)):
                    value = block_attention[i, j]
                    if value > 0.001:
                        text_color = 'white' if value > 0.5 * max_val else 'black'
                        ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                               fontsize=6, color=text_color)
        
        ax.set_xlabel('Key Blocks (Context)', fontsize=10)
        ax.set_ylabel('Query Blocks (Generation)', fontsize=10)
        ax.set_title(f'{title} - Layer {layer_idx}', fontsize=12)
        
        # 添加colorbar
        plt.colorbar(im, ax=ax, shrink=0.8, label='Attention Weight')
        
        return ax
    
    def plot_comparison(
        self,
        attention_maps: Dict[int, np.ndarray],
        blocks: List[SemanticBlock],
        save_path: str = None,
        title_prefix: str = ""
    ):
        """绘制多层attention对比图"""
        n_layers = len(attention_maps)
        if n_layers == 0:
            print("No attention maps to plot!")
            return
        
        # 动态调整图片大小
        fig_width = min(8 * n_layers, 24)
        fig_height = 7
        
        fig, axes = plt.subplots(1, n_layers, figsize=(fig_width, fig_height))
        
        if n_layers == 1:
            axes = [axes]
        
        for idx, (layer_idx, block_attn) in enumerate(sorted(attention_maps.items())):
            self.plot_attention_map(
                block_attn, 
                blocks, 
                layer_idx,
                title=f"{title_prefix}Layer {layer_idx}" if title_prefix else f"Layer {layer_idx}",
                ax=axes[idx]
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
    def plot_block_summary(
        self,
        blocks: List[SemanticBlock],
        save_path: str = None
    ):
        """绘制block概览图"""
        fig, ax = plt.subplots(figsize=(14, 4))
        
        y_pos = 0
        for i, block in enumerate(blocks):
            color = self.type_colors.get(block.block_type, '#CCCCCC')
            width = block.end_idx - block.start_idx
            
            rect = plt.Rectangle(
                (block.start_idx, y_pos - 0.4), width, 0.8,
                facecolor=color, edgecolor='black', linewidth=0.5
            )
            ax.add_patch(rect)
            
            # 添加标签
            if width > 50:
                label = f"{block.block_type[:4]}_{block.turn_idx}"
                ax.text(
                    block.start_idx + width/2, y_pos, label,
                    ha='center', va='center', fontsize=7
                )
        
        ax.set_xlim(0, blocks[-1].end_idx if blocks else 100)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('Token Position')
        ax.set_title('Semantic Block Layout')
        ax.set_yticks([])
        
        # 添加图例
        legend_elements = [
            plt.Rectangle((0,0), 1, 1, facecolor=color, label=block_type)
            for block_type, color in self.type_colors.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right', ncol=3, fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


# ==================== 主Pipeline ====================

class BlockAttentionAnalyzer:
    """Block Attention分析主类"""
    
    def __init__(
        self,
        model_name_or_path: str,
        target_layers: List[int] = None,
        device_map: str = "auto",
        torch_dtype=torch.float16,
        chunk_size: int = 1024
    ):
        print(f"Loading model: {model_name_or_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        
        # 加载模型
        load_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }
        
        # 尝试使用SDPA
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                attn_implementation="sdpa",
                **load_kwargs
            )
            print("  Using SDPA attention implementation")
        except Exception as e:
            print(f"  SDPA not available ({e}), using default")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **load_kwargs
            )
            
        self.model.eval()
        
        # 确定目标层
        num_layers = self.model.config.num_hidden_layers
        if target_layers is None:
            # 默认选择中间层和最后一层
            target_layers = [num_layers // 2, num_layers - 1]
        
        # 确保层索引有效
        self.target_layers = [l for l in target_layers if 0 <= l < num_layers]
        print(f"Target layers: {self.target_layers} (total layers: {num_layers})")
        
        # 初始化组件
        self.extractor = SDPAAttentionExtractor(self.model, self.target_layers, chunk_size)
        self.parser = SemanticBlockParser(self.tokenizer)
        self.visualizer = BlockAttentionVisualizer()
        
    def analyze(
        self,
        messages: List[Dict],
        visualize: bool = True,
        save_path: str = None,
        show_block_layout: bool = False
    ) -> Dict[str, Any]:
        """
        分析单个对话的Block Attention
        
        Args:
            messages: 对话消息列表
            visualize: 是否可视化
            save_path: 图片保存路径
            show_block_layout: 是否显示block布局图
            
        Returns:
            分析结果字典
        """
        
        # 1. 准备输入
        print("\nPreparing input...")
        input_ids = self._prepare_input(messages)
        input_ids = input_ids.to(self.model.device)
        seq_len = input_ids.shape[1]
        print(f"Total tokens: {seq_len}")
        
        # 2. 解析语义块
        print("\nParsing semantic blocks...")
        blocks = self.parser.parse_messages_with_positions(messages, input_ids)
        print(f"Found {len(blocks)} blocks:")
        for block in blocks:
            print(f"  {block}")
        
        # 3. 注册hooks
        print("\nRegistering hooks...")
        self.extractor.register_hooks()
        
        # 4. 前向传播
        print("\nRunning forward pass...")
        with torch.no_grad():
            _ = self.model(input_ids)
        
        # 5. 计算attention weights
        print("\nComputing attention weights...")
        attention_weights = self.extractor.compute_attention_weights()
        
        # 6. 清理
        self.extractor.clear()
        
        # 7. 聚合为block attention
        print("\nAggregating to block attention...")
        block_attention_maps = {}
        calculator = BlockAttentionCalculator(blocks, seq_len)
        
        for layer_idx, attn in attention_weights.items():
            block_attn = calculator.compute_block_attention(attn)
            block_attention_maps[layer_idx] = block_attn
            print(f"  Layer {layer_idx}: block attention shape = {block_attn.shape}")
        
        # 8. 可视化
        if visualize and block_attention_maps:
            print("\nGenerating visualizations...")
            
            if show_block_layout:
                self.visualizer.plot_block_summary(blocks)
            
            self.visualizer.plot_comparison(
                block_attention_maps,
                blocks,
                save_path=save_path
            )
        
        return {
            'blocks': blocks,
            'block_attention_maps': block_attention_maps,
            'token_attention_maps': attention_weights,
            'num_tokens': seq_len,
            'input_ids': input_ids.cpu()
        }
    
    def _prepare_input(self, messages: List[Dict]) -> torch.Tensor:
        """准备模型输入"""
        # 优先使用chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=False
                )
                return input_ids
            except Exception as e:
                print(f"  Chat template failed: {e}, falling back to manual encoding")
        
        # Fallback: 手动拼接
        full_text = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            full_text += f"<|{role}|>\n{content}\n"
        
        input_ids = self.tokenizer(full_text, return_tensors="pt")['input_ids']
        return input_ids
    
    def analyze_from_json(
        self,
        json_path: str,
        sample_idx: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """从JSON文件加载并分析"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            if sample_idx >= len(data):
                raise IndexError(f"sample_idx {sample_idx} out of range (total: {len(data)})")
            sample = data[sample_idx]
        else:
            sample = data
            
        messages = sample.get('messages', sample)
        return self.analyze(messages, **kwargs)


# ==================== 命令行接口 ====================

def parse_arguments():
    """解析命令行参数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Block Attention Map Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python block_attention.py --model /path/to/model
  python block_attention.py --json data.json --sample_idx 0
  python block_attention.py --layers 10,20,30 --chunk_size 512
        """
    )
    
    parser.add_argument(
        '--model', type=str,
        default='/mnt/lc_share/modelscope/models/Qwen/WebAgent/WebSailor-3B',
        help='Model path or name'
    )
    parser.add_argument(
        '--json', type=str, default=None,
        help='Path to JSON file with conversation data'
    )
    parser.add_argument(
        '--sample_idx', type=int, default=0,
        help='Sample index in JSON file'
    )
    parser.add_argument(
        '--output', type=str, default='block_attention_map.png',
        help='Output path for visualization'
    )
    parser.add_argument(
        '--layers', type=str, default=None,
        help='Comma-separated layer indices (e.g., "10,20,30")'
    )
    parser.add_argument(
        '--chunk_size', type=int, default=1024,
        help='Chunk size for memory-efficient attention computation'
    )
    parser.add_argument(
        '--show_layout', action='store_true',
        help='Show block layout visualization'
    )
    
    return parser.parse_args()


def create_sample_messages():
    """创建示例消息"""
    return [
        {
            "role": "system",
            "content": "You are a helpful AI assistant with access to tools."
        },
        {
            "role": "user", 
            "content": "Calculate 25 * 47 and tell me if the result is prime."
        },
        {
            "role": "assistant",
            "content": """<think>I need to calculate 25 * 47 first, then check if it's prime.</think>
<tool_call>{"name": "calculator", "arguments": {"expression": "25 * 47"}}</tool_call>"""
        },
        {
            "role": "user",
            "content": "<tool_response>1175</tool_response>"
        },
        {
            "role": "assistant",
            "content": """<think>Got 1175. Now I need to check if it's prime.</think>
<tool_call>{"name": "is_prime", "arguments": {"number": 1175}}</tool_call>"""
        },
        {
            "role": "user",
            "content": "<tool_response>False. 1175 = 5^2 * 47</tool_response>"
        },
        {
            "role": "assistant",
            "content": "The result of 25 × 47 is **1175**. This number is not prime - it can be factored as 5² × 47."
        }
    ]


def main():
    """主函数"""
    import sys
    import os
    
    args = parse_arguments()
    
    # 解析layers参数
    target_layers = None
    if args.layers:
        try:
            target_layers = [int(l.strip()) for l in args.layers.split(',')]
        except ValueError:
            print(f"Error: Invalid layers format '{args.layers}'")
            sys.exit(1)
    
    # 加载消息
    if args.json:
        if not os.path.exists(args.json):
            print(f"Error: JSON file not found: {args.json}")
            sys.exit(1)
        
        print(f"Loading from JSON: {args.json}, sample index: {args.sample_idx}")
        
        with open(args.json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            if args.sample_idx >= len(data):
                print(f"Error: sample_idx {args.sample_idx} out of range")
                sys.exit(1)
            sample = data[args.sample_idx]
        else:
            sample = data
        
        messages = sample.get('messages', sample)
    else:
        print("Using default sample messages")
        messages = create_sample_messages()
    
    # 显示消息概览
    print(f"\nLoaded {len(messages)} messages:")
    for idx, msg in enumerate(messages[:5]):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')[:80]
        print(f"  [{idx}] {role}: {content}...")
    if len(messages) > 5:
        print(f"  ... and {len(messages) - 5} more")
    
    print("\n" + "="*60)
    print("Block Attention Map Analyzer")
    print("="*60)
    
    try:
        analyzer = BlockAttentionAnalyzer(
            model_name_or_path=args.model,
            target_layers=target_layers,
            device_map="auto",
            torch_dtype=torch.float16,
            chunk_size=args.chunk_size
        )
        
        results = analyzer.analyze(
            messages=messages,
            visualize=True,
            save_path=args.output,
            show_block_layout=args.show_layout
        )
        
        print("\n" + "="*60)
        print("Analysis Complete!")
        print(f"  Total blocks: {len(results['blocks'])}")
        print(f"  Total tokens: {results['num_tokens']}")
        print(f"  Layers analyzed: {list(results['block_attention_maps'].keys())}")
        print(f"  Output saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()