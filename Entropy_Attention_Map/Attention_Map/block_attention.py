"""
Block Attention Map Visualization Tool
======================================
使用Hook机制提取指定层的注意力权重，聚合为语义块级别的Attention Map
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from collections import defaultdict
import warnings
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


class AttentionHookManager:
    """
    注意力Hook管理器
    使用hook提取QKV states，然后重计算attention weights
    避免直接输出attention导致的显存问题
    """
    
    def __init__(self, model, target_layers: List[int]):
        """
        Args:
            model: HuggingFace模型
            target_layers: 需要提取的层索引列表
        """
        self.model = model
        self.target_layers = target_layers
        self.hooks = []
        self.attention_weights: Dict[int, torch.Tensor] = {}
        self.model_type = self._detect_model_type()
        
    def _detect_model_type(self) -> str:
        """检测模型类型"""
        model_name = self.model.__class__.__name__.lower()
        config_name = self.model.config.__class__.__name__.lower()
        
        if 'qwen' in model_name or 'qwen' in config_name:
            return 'qwen'
        elif 'llama' in model_name or 'llama' in config_name:
            return 'llama'
        elif 'mistral' in model_name:
            return 'mistral'
        else:
            return 'llama'  # 默认
    
    def _get_layer_module(self, layer_idx: int):
        """获取指定层的模块"""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h[layer_idx]
        else:
            raise ValueError(f"Unsupported model structure: {type(self.model)}")
    
    def _get_attention_module(self, layer_idx: int):
        """获取指定层的attention模块"""
        layer = self._get_layer_module(layer_idx)
        if hasattr(layer, 'self_attn'):
            return layer.self_attn
        elif hasattr(layer, 'attention'):
            return layer.attention
        elif hasattr(layer, 'attn'):
            return layer.attn
        else:
            raise ValueError(f"Cannot find attention module in layer {layer_idx}")
    
    def register_hooks(self):
        """注册forward hooks到目标层的attention模块"""
        self.clear_hooks()
        self.attention_weights.clear()
        
        for layer_idx in self.target_layers:
            attn_module = self._get_attention_module(layer_idx)
            
            # 创建pre-hook来捕获输入hidden states
            def make_pre_hook(idx):
                def hook_fn(module, args):
                    try:
                        # 获取hidden_states (通常是第一个位置参数)
                        if len(args) > 0:
                            hidden_states = args[0]
                            
                            # 重新计算attention
                            attn_weights = self._compute_attention_from_hidden(
                                module, hidden_states, idx
                            )
                            if attn_weights is not None:
                                self.attention_weights[idx] = attn_weights.detach().cpu()
                    except Exception as e:
                        print(f"  Hook error at layer {idx}: {e}")
                    
                    return None  # 不修改输入
                return hook_fn
            
            hook = attn_module.register_forward_pre_hook(make_pre_hook(layer_idx))
            self.hooks.append(hook)
            
        print(f"  Registered {len(self.hooks)} hooks")
            
    def _compute_attention_from_hidden(
        self, 
        attn_module, 
        hidden_states: torch.Tensor,
        layer_idx: int
    ) -> Optional[torch.Tensor]:
        """
        从hidden states重新计算attention weights
        """
        try:
            batch_size, seq_len, hidden_size = hidden_states.shape
            config = self.model.config
            
            num_heads = config.num_attention_heads
            num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
            head_dim = hidden_size // num_heads
            
            with torch.no_grad():
                # 获取Q, K投影权重并计算
                if hasattr(attn_module, 'q_proj') and hasattr(attn_module, 'k_proj'):
                    q = attn_module.q_proj(hidden_states)
                    k = attn_module.k_proj(hidden_states)
                elif hasattr(attn_module, 'qkv_proj'):
                    # 某些模型使用合并的QKV投影
                    qkv = attn_module.qkv_proj(hidden_states)
                    qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
                    q = qkv[:, :, 0].reshape(batch_size, seq_len, -1)
                    k = qkv[:, :, 1].reshape(batch_size, seq_len, -1)
                else:
                    return None
                
                # Reshape for multi-head attention
                q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
                
                # 处理GQA: 复制K heads来匹配Q heads
                if num_kv_heads != num_heads:
                    n_rep = num_heads // num_kv_heads
                    k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
                    k = k.reshape(batch_size, num_heads, seq_len, head_dim)
                
                # 计算attention scores
                scale = head_dim ** -0.5
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
                
                # 应用causal mask
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=attn_weights.device, dtype=torch.bool),
                    diagonal=1
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
                
                # Softmax
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
                
                return attn_weights
                
        except Exception as e:
            print(f"    Attention computation error: {e}")
            return None
    
    def clear_hooks(self):
        """清除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def get_aggregated_attention(self, layer_idx: int) -> Optional[torch.Tensor]:
        """
        获取聚合后的attention weights (对所有heads取平均)
        
        Returns:
            attention: [seq_len, seq_len]
        """
        if layer_idx not in self.attention_weights:
            print(f"  Warning: No attention weights found for layer {layer_idx}")
            return None
        
        attn_weights = self.attention_weights[layer_idx]
        # [batch, heads, seq, seq] -> [seq, seq]
        aggregated = attn_weights.mean(dim=(0, 1))
        return aggregated


# ==================== 另一种方案：使用output_hidden_states ====================

class SimpleAttentionExtractor:
    """
    简化版本：使用前向传播后手动计算attention
    更可靠但需要额外一次前向传播
    """
    
    def __init__(self, model, target_layers: List[int]):
        self.model = model
        self.target_layers = target_layers
        self.attention_weights: Dict[int, torch.Tensor] = {}
        
    def extract_attention(self, input_ids: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        提取指定层的attention weights
        通过手动计算QK得到
        """
        config = self.model.config
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
        hidden_size = config.hidden_size
        head_dim = hidden_size // num_heads
        
        self.attention_weights.clear()
        batch_size, seq_len = input_ids.shape
        
        # 获取layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
            embed_tokens = self.model.model.embed_tokens
            norm = self.model.model.norm if hasattr(self.model.model, 'norm') else None
        else:
            raise ValueError("Unsupported model structure")
        
        with torch.no_grad():
            # 获取embeddings
            hidden_states = embed_tokens(input_ids)
            
            # 逐层前向
            for layer_idx, layer in enumerate(layers):
                # 保存当前hidden states用于attention计算
                if layer_idx in self.target_layers:
                    attn_weights = self._compute_layer_attention(
                        layer.self_attn, 
                        hidden_states,
                        num_heads,
                        num_kv_heads,
                        head_dim
                    )
                    if attn_weights is not None:
                        self.attention_weights[layer_idx] = attn_weights.cpu()
                        print(f"  Layer {layer_idx}: captured attention shape {attn_weights.shape}")
                
                # 继续前向传播
                layer_output = layer(hidden_states)
                if isinstance(layer_output, tuple):
                    hidden_states = layer_output[0]
                else:
                    hidden_states = layer_output
        
        return self.attention_weights
    
    def _compute_layer_attention(
        self,
        attn_module,
        hidden_states: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int
    ) -> Optional[torch.Tensor]:
        """计算单层的attention weights"""
        try:
            batch_size, seq_len, hidden_size = hidden_states.shape
            
            # 先做layer norm (如果attention模块前有的话)
            # 大多数模型在attention前会做norm
            
            # 计算Q, K
            if hasattr(attn_module, 'q_proj') and hasattr(attn_module, 'k_proj'):
                q = attn_module.q_proj(hidden_states)
                k = attn_module.k_proj(hidden_states)
            else:
                return None
            
            # Reshape
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            
            # GQA处理
            if num_kv_heads != num_heads:
                n_rep = num_heads // num_kv_heads
                k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
                k = k.reshape(batch_size, num_heads, seq_len, head_dim)
            
            # Attention scores
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            # Causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=attn_weights.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            
            # Softmax
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
            
            return attn_weights
            
        except Exception as e:
            print(f"    Error computing attention: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_aggregated_attention(self, layer_idx: int) -> Optional[torch.Tensor]:
        """获取聚合后的attention (对heads取平均)"""
        if layer_idx not in self.attention_weights:
            return None
        attn = self.attention_weights[layer_idx]
        return attn.mean(dim=(0, 1))


# ==================== 使用SDPA后处理方案 ====================

class SDPAAttentionExtractor:
    """
    针对使用SDPA/FlashAttention的模型
    在模型输出后，使用保存的hidden states重新计算attention
    """
    
    def __init__(self, model, target_layers: List[int]):
        self.model = model
        self.target_layers = target_layers
        self.hidden_states_cache: Dict[int, torch.Tensor] = {}
        self.hooks = []
        
    def register_hooks(self):
        """注册hooks来捕获每层输入的hidden states"""
        self.clear()
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        else:
            raise ValueError("Unsupported model structure")
        
        for layer_idx in self.target_layers:
            layer = layers[layer_idx]
            
            def make_hook(idx):
                def hook_fn(module, args, kwargs):
                    # 捕获输入hidden states
                    if len(args) > 0:
                        self.hidden_states_cache[idx] = args[0].detach().clone()
                    return None
                return hook_fn
            
            # 注册到整个layer而不是attention模块
            hook = layer.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)
            self.hooks.append(hook)
        
        print(f"  Registered {len(self.hooks)} layer hooks")
    
    def compute_attention_weights(self) -> Dict[int, torch.Tensor]:
        """根据缓存的hidden states计算attention weights"""
        config = self.model.config
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
        hidden_size = config.hidden_size
        head_dim = hidden_size // num_heads
        
        attention_weights = {}
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        else:
            return attention_weights
        
        for layer_idx in self.target_layers:
            if layer_idx not in self.hidden_states_cache:
                print(f"  No hidden states cached for layer {layer_idx}")
                continue
            
            hidden_states = self.hidden_states_cache[layer_idx]
            layer = layers[layer_idx]
            attn_module = layer.self_attn
            
            # 应用input layernorm (如果存在)
            if hasattr(layer, 'input_layernorm'):
                hidden_states = layer.input_layernorm(hidden_states)
            
            try:
                batch_size, seq_len, _ = hidden_states.shape
                
                with torch.no_grad():
                    q = attn_module.q_proj(hidden_states)
                    k = attn_module.k_proj(hidden_states)
                    
                    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                    k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
                    
                    if num_kv_heads != num_heads:
                        n_rep = num_heads // num_kv_heads
                        k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
                        k = k.reshape(batch_size, num_heads, seq_len, head_dim)
                    
                    scale = head_dim ** -0.5
                    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
                    
                    causal_mask = torch.triu(
                        torch.ones(seq_len, seq_len, device=attn.device, dtype=torch.bool),
                        diagonal=1
                    )
                    attn = attn.masked_fill(causal_mask, float('-inf'))
                    attn = F.softmax(attn, dim=-1, dtype=torch.float32)
                    
                    attention_weights[layer_idx] = attn.cpu()
                    print(f"  Layer {layer_idx}: computed attention shape {attn.shape}")
                    
            except Exception as e:
                print(f"  Error at layer {layer_idx}: {e}")
                import traceback
                traceback.print_exc()
        
        return attention_weights
    
    def clear(self):
        """清除hooks和缓存"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.hidden_states_cache.clear()
    
    def get_aggregated_attention(self, attention_weights: Dict[int, torch.Tensor], layer_idx: int) -> Optional[torch.Tensor]:
        """获取聚合后的attention"""
        if layer_idx not in attention_weights:
            return None
        return attention_weights[layer_idx].mean(dim=(0, 1))


# ==================== 语义块解析器 ====================

class SemanticBlockParser:
    """解析消息列表，提取语义块"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def parse_messages(self, messages: List[Dict]) -> Tuple[List[SemanticBlock], List[int]]:
        """
        解析消息列表，返回语义块列表和完整的token ids
        
        注意：第一个user中的think和tool_call属于提示词部分，合并到system
        """
        blocks = []
        all_tokens = []
        current_offset = 0
        
        # 处理消息，合并system和第一个user
        processed_messages = self._preprocess_messages(messages)
        
        for turn_idx, msg in enumerate(processed_messages):
            role = msg['role']
            content = msg['content']
            
            # 使用chat template更准确地获取token
            if role == 'system_init':
                # 编码system_init内容
                tokens = self._encode_with_role(content, 'system', is_first=(turn_idx == 0))
                
                block = SemanticBlock(
                    block_type='system_init',
                    content=content[:100] + "..." if len(content) > 100 else content,
                    start_idx=current_offset,
                    end_idx=current_offset + len(tokens),
                    role='system',
                    turn_idx=turn_idx
                )
                blocks.append(block)
                all_tokens.extend(tokens)
                current_offset += len(tokens)
                
            elif role == 'assistant':
                sub_blocks = self._parse_assistant_content(content, current_offset, turn_idx)
                for block in sub_blocks:
                    blocks.append(block)
                tokens = self._encode_with_role(content, 'assistant', is_first=False)
                all_tokens.extend(tokens)
                
                # 更新block的end_idx
                if sub_blocks:
                    total_len = len(tokens)
                    # 重新计算每个子块的位置
                    self._recalculate_block_positions(sub_blocks, current_offset, content)
                
                current_offset += len(tokens)
                
            elif role == 'user':
                sub_blocks = self._parse_user_content(content, current_offset, turn_idx)
                for block in sub_blocks:
                    blocks.append(block)
                tokens = self._encode_with_role(content, 'user', is_first=False)
                all_tokens.extend(tokens)
                current_offset += len(tokens)
        
        return blocks, all_tokens
    
    def _encode_with_role(self, content: str, role: str, is_first: bool = False) -> List[int]:
        """根据角色编码内容"""
        # 简化处理：直接编码内容
        tokens = self.tokenizer.encode(content, add_special_tokens=is_first)
        return tokens
    
    def _recalculate_block_positions(self, sub_blocks: List[SemanticBlock], base_offset: int, full_content: str):
        """重新计算子块的token位置"""
        current_pos = 0
        current_offset = base_offset
        
        for block in sub_blocks:
            # 找到block内容在full_content中的位置
            block_text = block.content
            if block_text in full_content[current_pos:]:
                text_before = full_content[current_pos:full_content.find(block_text, current_pos)]
                tokens_before = self.tokenizer.encode(text_before, add_special_tokens=False)
                current_offset += len(tokens_before)
                
                block_tokens = self.tokenizer.encode(block_text, add_special_tokens=False)
                block.start_idx = current_offset
                block.end_idx = current_offset + len(block_tokens)
                
                current_offset = block.end_idx
                current_pos = full_content.find(block_text, current_pos) + len(block_text)
    
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
                combined_content = system_content + "\n\n" + content if system_content else content
                processed.append({'role': 'system_init', 'content': combined_content})
                first_user_processed = True
            else:
                processed.append(msg)
                
        return processed
    
    def _parse_assistant_content(self, content: str, start_offset: int, turn_idx: int) -> List[SemanticBlock]:
        """解析assistant消息内容"""
        blocks = []
        
        # 正则表达式匹配各种标签
        patterns = [
            (r'<think>(.*?)</think>', 'think'),
            (r'<tool_call>(.*?)</tool_call>', 'tool_call'),
            (r'<tool_calls>(.*?)</tool_calls>', 'tool_call'),
        ]
        
        # 找到所有标签
        tag_positions = []
        for pattern, block_type in patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                tag_positions.append({
                    'start': match.start(),
                    'end': match.end(),
                    'content': match.group(0),
                    'type': block_type
                })
        
        tag_positions.sort(key=lambda x: x['start'])
        
        current_pos = 0
        current_offset = start_offset
        
        for tag_info in tag_positions:
            # 标签前的文本作为answer
            if tag_info['start'] > current_pos:
                text_before = content[current_pos:tag_info['start']].strip()
                if text_before:
                    tokens = self.tokenizer.encode(text_before, add_special_tokens=False)
                    blocks.append(SemanticBlock(
                        block_type='answer',
                        content=text_before[:50] + "..." if len(text_before) > 50 else text_before,
                        start_idx=current_offset,
                        end_idx=current_offset + len(tokens),
                        role='assistant',
                        turn_idx=turn_idx
                    ))
                    current_offset += len(tokens)
            
            # 标签内容
            tag_content = tag_info['content']
            tokens = self.tokenizer.encode(tag_content, add_special_tokens=False)
            blocks.append(SemanticBlock(
                block_type=tag_info['type'],
                content=tag_content[:50] + "..." if len(tag_content) > 50 else tag_content,
                start_idx=current_offset,
                end_idx=current_offset + len(tokens),
                role='assistant',
                turn_idx=turn_idx
            ))
            current_offset += len(tokens)
            current_pos = tag_info['end']
        
        # 剩余文本
        if current_pos < len(content):
            remaining = content[current_pos:].strip()
            if remaining:
                tokens = self.tokenizer.encode(remaining, add_special_tokens=False)
                blocks.append(SemanticBlock(
                    block_type='answer',
                    content=remaining[:50] + "..." if len(remaining) > 50 else remaining,
                    start_idx=current_offset,
                    end_idx=current_offset + len(tokens),
                    role='assistant',
                    turn_idx=turn_idx
                ))
        
        return blocks if blocks else [SemanticBlock(
            block_type='answer',
            content=content[:50] + "..." if len(content) > 50 else content,
            start_idx=start_offset,
            end_idx=start_offset + len(self.tokenizer.encode(content, add_special_tokens=False)),
            role='assistant',
            turn_idx=turn_idx
        )]
    
    def _parse_user_content(self, content: str, start_offset: int, turn_idx: int) -> List[SemanticBlock]:
        """解析user消息内容"""
        tokens = self.tokenizer.encode(content, add_special_tokens=False)
        
        # 检查是否是tool_response
        if '<tool_response>' in content or '<tool_result>' in content:
            block_type = 'tool_response'
        else:
            block_type = 'user_message'
        
        return [SemanticBlock(
            block_type=block_type,
            content=content[:50] + "..." if len(content) > 50 else content,
            start_idx=start_offset,
            end_idx=start_offset + len(tokens),
            role='user',
            turn_idx=turn_idx
        )]


# ==================== Block Attention计算 ====================

class BlockAttentionCalculator:
    """计算Block级别的Attention矩阵"""
    
    def __init__(self, blocks: List[SemanticBlock], total_tokens: int):
        self.blocks = blocks
        self.total_tokens = total_tokens
        self._adjust_block_indices()
        
    def _adjust_block_indices(self):
        """确保block索引不超过总token数"""
        for block in self.blocks:
            block.start_idx = min(block.start_idx, self.total_tokens - 1)
            block.end_idx = min(block.end_idx, self.total_tokens)
            if block.end_idx <= block.start_idx:
                block.end_idx = block.start_idx + 1
        
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
        
        for i, gen_block in enumerate(self.blocks):
            for j, ctx_block in enumerate(self.blocks):
                # 因果性：只计算生成块对之前上下文块的注意力
                if gen_block.start_idx >= ctx_block.start_idx:
                    gen_start = min(gen_block.start_idx, seq_len - 1)
                    gen_end = min(gen_block.end_idx, seq_len)
                    ctx_start = min(ctx_block.start_idx, seq_len - 1)
                    ctx_end = min(ctx_block.end_idx, seq_len)
                    
                    if gen_end > gen_start and ctx_end > ctx_start:
                        sub_matrix = token_attention_np[gen_start:gen_end, ctx_start:ctx_end]
                        block_attention[i, j] = np.mean(sub_matrix)
        
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
        
    def plot_attention_map(
        self, 
        block_attention: np.ndarray,
        blocks: List[SemanticBlock],
        layer_idx: int,
        title: str = "Block Attention Map",
        ax=None
    ):
        """绘制单个attention map"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        im = ax.imshow(block_attention, cmap='Blues', aspect='auto')
        
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
        
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        
        # 添加数值标注
        max_val = block_attention.max() if block_attention.max() > 0 else 1
        for i in range(len(blocks)):
            for j in range(len(blocks)):
                value = block_attention[i, j]
                if value > 0.001:
                    text_color = 'white' if value > 0.5 * max_val else 'black'
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                           fontsize=6, color=text_color)
        
        ax.set_xlabel('Key Blocks (Context)', fontsize=10)
        ax.set_ylabel('Query Blocks (Generation)', fontsize=10)
        ax.set_title(f'{title}\nLayer {layer_idx}', fontsize=12)
        
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        return ax
    
    def plot_comparison(
        self,
        attention_maps: Dict[int, np.ndarray],
        blocks: List[SemanticBlock],
        save_path: str = None
    ):
        """绘制多层attention对比图"""
        n_layers = len(attention_maps)
        if n_layers == 0:
            print("No attention maps to plot!")
            return
            
        fig, axes = plt.subplots(1, n_layers, figsize=(7*n_layers, 6))
        
        if n_layers == 1:
            axes = [axes]
        
        for idx, (layer_idx, block_attn) in enumerate(attention_maps.items()):
            self.plot_attention_map(
                block_attn, 
                blocks, 
                layer_idx,
                title=f"Layer {layer_idx}",
                ax=axes[idx]
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()


# ==================== 主Pipeline ====================

class BlockAttentionAnalyzer:
    """Block Attention分析主类"""
    
    def __init__(
        self,
        model_name_or_path: str,
        target_layers: List[int] = None,
        device_map: str = "auto",
        torch_dtype=torch.float16
    ):
        print(f"Loading model: {model_name_or_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        
        # 尝试使用SDPA，如果不可用则用默认
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                attn_implementation="sdpa"
            )
        except Exception as e:
            print(f"SDPA not available, using default: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
            
        self.model.eval()
        
        # 确定目标层
        num_layers = self.model.config.num_hidden_layers
        if target_layers is None:
            target_layers = [int(num_layers * 2 / 3), num_layers - 1]
        
        # 确保层索引有效
        self.target_layers = [min(l, num_layers - 1) for l in target_layers]
        print(f"Target layers: {self.target_layers} (total layers: {num_layers})")
        
        # 初始化组件
        self.extractor = SDPAAttentionExtractor(self.model, self.target_layers)
        self.parser = SemanticBlockParser(self.tokenizer)
        self.visualizer = BlockAttentionVisualizer()
        
    def analyze(
        self,
        messages: List[Dict],
        visualize: bool = True,
        save_path: str = None
    ) -> Dict[str, Any]:
        """分析单个对话的Block Attention"""
        
        # 1. 解析语义块
        print("Parsing semantic blocks...")
        blocks, token_list = self.parser.parse_messages(messages)
        print(f"Found {len(blocks)} blocks:")
        for block in blocks:
            print(f"  {block}")
        
        # 2. 准备输入
        print("\nPreparing input...")
        
        # 使用chat template如果可用
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=False
                )
            except:
                # fallback
                full_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                input_ids = self.tokenizer(full_text, return_tensors="pt")['input_ids']
        else:
            full_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            input_ids = self.tokenizer(full_text, return_tensors="pt")['input_ids']
        
        input_ids = input_ids.to(self.model.device)
        seq_len = input_ids.shape[1]
        print(f"Total tokens: {seq_len}")
        
        # 3. 注册hooks
        print("\nRegistering hooks...")
        self.extractor.register_hooks()
        
        # 4. 前向传播
        print("Running forward pass...")
        with torch.no_grad():
            _ = self.model(input_ids)
        
        # 5. 计算attention weights
        print("\nComputing attention weights...")
        attention_weights = self.extractor.compute_attention_weights()
        
        # 6. 清理
        self.extractor.clear()
        
        # 7. 聚合为block attention
        print("\nAggregating to block attention...")
        
        # 重新计算block位置以匹配实际token数
        blocks = self._reparse_blocks_for_chat_template(messages, seq_len)
        
        block_attention_maps = {}
        calculator = BlockAttentionCalculator(blocks, seq_len)
        
        for layer_idx, attn in attention_weights.items():
            aggregated = attn.mean(dim=(0, 1))  # [seq, seq]
            block_attn = calculator.compute_block_attention(aggregated)
            block_attention_maps[layer_idx] = block_attn
            print(f"  Layer {layer_idx}: block attention shape = {block_attn.shape}")
        
        # 8. 可视化
        if visualize and block_attention_maps:
            print("\nGenerating visualizations...")
            self.visualizer.plot_comparison(
                block_attention_maps,
                blocks,
                save_path=save_path
            )
        
        return {
            'blocks': blocks,
            'block_attention_maps': block_attention_maps,
            'num_tokens': seq_len
        }
    
    def _reparse_blocks_for_chat_template(self, messages: List[Dict], total_tokens: int) -> List[SemanticBlock]:
        """
        根据chat template重新解析blocks
        这是一个简化版本，将整个序列均匀分配给各个消息
        """
        blocks = []
        
        # 合并system和第一个user
        combined_messages = []
        system_content = ""
        first_user_done = False
        
        for msg in messages:
            if msg['role'] == 'system':
                system_content = msg['content']
            elif msg['role'] == 'user' and not first_user_done:
                combined_content = system_content + "\n" + msg['content'] if system_content else msg['content']
                combined_messages.append({'role': 'system_init', 'content': combined_content})
                first_user_done = True
            else:
                combined_messages.append(msg)
        
        # 估算每个消息的token数量
        msg_tokens = []
        for msg in combined_messages:
            tokens = self.tokenizer.encode(msg['content'], add_special_tokens=False)
            msg_tokens.append(len(tokens))
        
        # 加上special tokens的估计
        total_estimated = sum(msg_tokens)
        scale = total_tokens / max(total_estimated, 1)
        
        current_pos = 0
        for turn_idx, (msg, n_tokens) in enumerate(zip(combined_messages, msg_tokens)):
            scaled_tokens = int(n_tokens * scale)
            if scaled_tokens == 0:
                scaled_tokens = 1
            
            role = msg['role']
            content = msg['content']
            
            if role == 'system_init':
                # 解析system_init中的子块
                sub_blocks = self._parse_content_blocks(content, current_pos, scaled_tokens, turn_idx, 'system')
                blocks.extend(sub_blocks)
            elif role == 'assistant':
                sub_blocks = self._parse_content_blocks(content, current_pos, scaled_tokens, turn_idx, 'assistant')
                blocks.extend(sub_blocks)
            elif role == 'user':
                block_type = 'tool_response' if '<tool_response>' in content else 'user_message'
                blocks.append(SemanticBlock(
                    block_type=block_type,
                    content=content[:50] + "...",
                    start_idx=current_pos,
                    end_idx=min(current_pos + scaled_tokens, total_tokens),
                    role='user',
                    turn_idx=turn_idx
                ))
            
            current_pos += scaled_tokens
        
        # 确保最后一个block的end_idx不超过total_tokens
        if blocks:
            blocks[-1].end_idx = min(blocks[-1].end_idx, total_tokens)
        
        return blocks
    
    def _parse_content_blocks(
        self, 
        content: str, 
        start_pos: int, 
        total_tokens: int, 
        turn_idx: int, 
        role: str
    ) -> List[SemanticBlock]:
        """解析内容中的子块"""
        blocks = []
        
        patterns = [
            (r'<think>.*?</think>', 'think'),
            (r'<tool_call>.*?</tool_call>', 'tool_call'),
            (r'<tool_calls>.*?</tool_calls>', 'tool_call'),
        ]
        
        # 找所有匹配
        matches = []
        for pattern, block_type in patterns:
            for m in re.finditer(pattern, content, re.DOTALL):
                matches.append((m.start(), m.end(), m.group(0), block_type))
        
        matches.sort(key=lambda x: x[0])
        
        if not matches:
            # 没有特殊标签，作为一个块
            block_type = 'system_init' if role == 'system' else 'answer'
            blocks.append(SemanticBlock(
                block_type=block_type,
                content=content[:50] + "...",
                start_idx=start_pos,
                end_idx=start_pos + total_tokens,
                role=role,
                turn_idx=turn_idx
            ))
        else:
            # 根据字符位置比例分配token位置
            content_len = len(content)
            
            current_char = 0
            current_token = start_pos
            
            for char_start, char_end, matched_text, block_type in matches:
                # 匹配前的文本
                if char_start > current_char:
                    before_ratio = (char_start - current_char) / content_len
                    before_tokens = int(before_ratio * total_tokens)
                    if before_tokens > 0:
                        text = content[current_char:char_start].strip()
                        if text:
                            blocks.append(SemanticBlock(
                                block_type='answer' if role == 'assistant' else 'system_init',
                                content=text[:50] + "...",
                                start_idx=current_token,
                                end_idx=current_token + before_tokens,
                                role=role,
                                turn_idx=turn_idx
                            ))
                        current_token += before_tokens
                
                # 匹配的块
                match_ratio = (char_end - char_start) / content_len
                match_tokens = max(1, int(match_ratio * total_tokens))
                
                blocks.append(SemanticBlock(
                    block_type=block_type,
                    content=matched_text[:50] + "...",
                    start_idx=current_token,
                    end_idx=current_token + match_tokens,
                    role=role,
                    turn_idx=turn_idx
                ))
                current_token += match_tokens
                current_char = char_end
            
            # 剩余文本
            if current_char < content_len:
                remaining = content[current_char:].strip()
                if remaining and current_token < start_pos + total_tokens:
                    blocks.append(SemanticBlock(
                        block_type='answer' if role == 'assistant' else 'system_init',
                        content=remaining[:50] + "...",
                        start_idx=current_token,
                        end_idx=start_pos + total_tokens,
                        role=role,
                        turn_idx=turn_idx
                    ))
        
        return blocks
    
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
            sample = data[sample_idx]
        else:
            sample = data
            
        messages = sample.get('messages', sample)
        return self.analyze(messages, **kwargs)


# ==================== 测试用例 ====================

def create_sample_messages():
    """创建示例消息"""
    return [
        {
            "role": "system",
            "content": "You are a helpful AI assistant with tools."
        },
        {
            "role": "user", 
            "content": """<think>Example thinking</think>
<tool_call>example()</tool_call>
Calculate 25 * 47."""
        },
        {
            "role": "assistant",
            "content": """<think>I need to calculate 25 * 47.</think>
<tool_call>calculator("25 * 47")</tool_call>"""
        },
        {
            "role": "user",
            "content": "<tool_response>1175</tool_response>"
        },
        {
            "role": "assistant",
            "content": """<think>The result is 1175.</think>
<tool_call>search("1175")</tool_call>"""
        },
        {
            "role": "user",
            "content": "<tool_response>1175 is a composite number.</tool_response>"
        },
        {
            "role": "assistant",
            "content": "The result of 25 × 47 is 1175, which is a composite number."
        }
    ]


def main():
    """主函数"""
    import sys
    import os
    
    # 解析命令行参数
    args = parse_arguments()
    
    model_path = args.model
    json_path = args.json
    sample_idx = args.sample_idx
    save_path = args.output
    target_layers = args.layers
    
    # 确定使用的消息
    messages = None
    if json_path:
        # 从JSON文件加载消息
        if not os.path.exists(json_path):
            print(f"Error: JSON file not found: {json_path}")
            sys.exit(1)
        
        print(f"Loading messages from JSON file: {json_path}")
        print(f"Using sample index: {sample_idx}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            if sample_idx >= len(data):
                print(f"Error: sample_idx {sample_idx} out of range (total samples: {len(data)})")
                sys.exit(1)
            sample = data[sample_idx]
        else:
            sample = data
        
        messages = sample.get('messages', sample)
        print(f"Loaded {len(messages)} messages from JSON")
        
        # 显示消息概览
        print("\nMessages overview:")
        for idx, msg in enumerate(messages[:5]):  # 只显示前5条
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            content_preview = content[:100] + "..." if len(content) > 100 else content
            print(f"  [{idx}] {role}: {content_preview}")
        if len(messages) > 5:
            print(f"  ... and {len(messages) - 5} more messages")
        print()
    else:
        # 使用示例消息
        messages = create_sample_messages()
        print("Using default sample messages\n")
    
    print("="*60)
    print("Block Attention Map Analyzer")
    print("="*60)
    
    try:
        # 如果指定了layers参数，转换为整数列表
        if target_layers:
            try:
                target_layers = [int(l) for l in target_layers.split(',')]
            except ValueError:
                print(f"Error: Invalid layers format '{target_layers}'. Expected comma-separated integers.")
                sys.exit(1)
        
        analyzer = BlockAttentionAnalyzer(
            model_name_or_path=model_path,
            target_layers=target_layers,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        results = analyzer.analyze(
            messages=messages,
            visualize=True,
            save_path=save_path
        )
        
        print("\n" + "="*60)
        print("Analysis Complete!")
        print(f"Total blocks: {len(results['blocks'])}")
        print(f"Total tokens: {results['num_tokens']}")
        print(f"Layers analyzed: {list(results['block_attention_maps'].keys())}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def parse_arguments():
    """解析命令行参数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Block Attention Map Analyzer - Analyze attention patterns in transformer models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用默认示例消息
  python block_attention.py
  
  # 指定模型路径
  python block_attention.py --model /path/to/model
  
  # 从JSON文件加载第一条消息
  python block_attention.py --json trajectories.json
  
  # 从JSON文件加载指定索引的消息
  python block_attention.py --json trajectories.json --sample_idx 5
  
  # 指定分析层并自定义输出路径
  python block_attention.py --json trajectories.json --layers 20,25 --output my_result.png
  
  # 完整示例
  python block_attention.py \\
      --model /mnt/lc_share/modelscope/models/Qwen/WebAgent/WebSailor-3B \\
      --json trajectories.json \\
      --sample_idx 0 \\
      --layers 20,25 \\
      --output attention_map.png
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='/mnt/lc_share/modelscope/models/Qwen/WebAgent/WebSailor-3B',
        help='Path or name of the pre-trained model (default: WebSailor-3B)'
    )
    
    parser.add_argument(
        '--json',
        type=str,
        default=None,
        help='Path to JSON file containing conversation data'
    )
    
    parser.add_argument(
        '--sample_idx',
        type=int,
        default=0,
        help='Index of sample to analyze from JSON file (default: 0)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='block_attention_map.png',
        help='Output path for the attention map visualization (default: block_attention_map.png)'
    )
    
    parser.add_argument(
        '--layers',
        type=str,
        default=None,
        help='Comma-separated list of layer indices to analyze (default: auto-select)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    main()
