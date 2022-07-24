import torch
import torch.nn as nn
from torch import Tensor
import math
import torch.nn.functional as F
from activation import get_activation


class SinusoidalPositionEncoding(nn.Module):
    """定义Sin-Cos位置Embedding
    """
    def __init__(self, max_position, embedding_size):
        super(SinusoidalPositionEncoding, self).__init__()
        self.position_embeddings = nn.Embedding.from_pretrained(self.get_sinusoid_encoding_table(max_position, embedding_size), freeze=True)

    def get_sinusoid_encoding_table(self, n_position, d_hid):
        '''
        Returns: [seq_len, d_hid]
        '''
        position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid))
        embeddings_table = torch.zeros(n_position, d_hid)
        embeddings_table[:, 0::2] = torch.sin(position * div_term)
        embeddings_table[:, 1::2] = torch.cos(position * div_term)
        return embeddings_table

    def forward(self, position_ids):
        return self.position_embeddings(position_ids)


class RelativePositionsEncoding(nn.Module):
    def __init__(self, **kwargs):
        super(RelativePositionsEncoding, self).__init__()
    
    
class RoPEPositionEncoding(nn.Module):
    def __init__(self, **kwargs):
        super(RoPEPositionEncoding, self).__init__()
    
    
class RelativePositionsEncodingT5(nn.Module):
    def __init__(self, **kwargs):
        super(RelativePositionsEncodingT5, self).__init__()
    
    
class LayerNormalization(nn.Module):
    """
    自适应层归一化
    """
    def __init__(self, condition_hidden_size, hidden_size, eps=1e-12, **kwargs):
        """
        layernorm 层，兼容 conditianal layernorm，使得可以做条件文本生成、条件分类等任务详情：https://spaces.ac.cn/archives/7124
        """
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.condition_hidden_size = condition_hidden_size
        if condition_hidden_size:
            self.dense1 = nn.Linear(condition_hidden_size, hidden_size, bias=False)
            self.dense1.weight.data.uniform_(0, 0)
            self.dense2 = nn.Linear(condition_hidden_size, hidden_size, bias=False)
            self.dense2.weight.data.uniform_(0, 0)  # 全零初始化
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, inputs: Tensor, condition_inputs):
        assert inputs.shape[-1] == self.hidden_size
        u = inputs.mean(-1, keepdim=True)
        s = (inputs - u).pow(2).mean(-1, keepdim=True)
        o = (inputs - u)/torch.sqrt(s + self.eps)

        if self.condition_hidden_size:
            for _ in range(len(inputs.shape) - len(condition_inputs.shape)):
                condition_inputs = condition_inputs.unsqueeze(dim=1)

            c_w = self.dense1(condition_inputs)
            c_b = self.dense2(condition_inputs)

            return (self.weight + c_w) * o + (self.bais + c_b)
        else:
            return self.weight * o + self.bais



class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, attention_scale=True,
                 return_attention_scores=False, bias=True, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__()
        assert hidden_size % num_attention_heads == 0

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size / num_attention_heads
        self.attention_scale = attention_scale
        self.return_attention_scores = return_attention_scores

        """  K, Q, V 线性变换"""
        self.bias = bias
        self.q = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        self.a_bias, self.p_bias = kwargs.get('a_bias'), kwargs.get('p_bias')

        """  适配不同模型的位置编码  这部分在BertEmbeddings未实现"""
        if self.p_bias == 'typical_relative':  # nezha
            self.relative_positions_encoding = RelativePositionsEncoding(qlen=kwargs.get('max_position'),
                                                                         klen=kwargs.get('max_position'),
                                                                         embedding_size=self.attention_head_size,
                                                                         max_relative_position=kwargs.get(
                                                                             'max_relative_position'))
        elif self.p_bias == 'rotary':  # roformer
            self.relative_positions_encoding = RoPEPositionEncoding(max_position=kwargs.get('max_position'),
                                                                    embedding_size=self.attention_head_size)
        elif self.p_bias == 't5_relative':  # t5
            self.relative_positions = RelativePositionsEncodingT5(qlen=kwargs.get('max_position'),
                                                                  klen=kwargs.get('max_position'),
                                                                  relative_attention_num_buckets=kwargs.get(
                                                                      'relative_attention_num_buckets'),
                                                                  is_decoder=kwargs.get('is_decoder'))
            self.relative_positions_encoding = nn.Embedding(kwargs.get('relative_attention_num_buckets'),
                                                            self.num_attention_heads)

    def transpose_for_scores(self, x):
        # x为 bat_size, seq_len, hidden_size
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # bat_size, seq_len, heads_num, head_size
        return x.permute(0, 2, 1, 3)  # bat_size, heads_num, seq_len, head_size

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):

        """  hidden_states线性变换成 k, q, v 矩阵 """
        mixed_query_layer = self.q(hidden_states)
        if encoder_hidden_states is not None:
            mixed_key_layer = self.k(encoder_hidden_states)
            mixed_value_layer = self.v(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.k(hidden_states)
            mixed_value_layer = self.v(hidden_states)

        """ 计算分数 """
        # bat_size, heads_num, seq_len, head_size
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if self.p_bias == 'rotary':
            # 计算位置编码 bat_size, heads_num, seq_len, head_size
            query_layer = self.relative_positions_encoding(query_layer)
            key_layer = self.relative_positions_encoding(key_layer)

        # 交换k的最后两个维度，然后q和k执行点积, 获得attention score
        # bat_size, heads_num, seq_len, seq_len
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (self.p_bias == 'typical_relative') and hasattr(self, 'relative_positions_encoding'):
            relations_keys = self.relative_positions_encoding(attention_scores.shape[-1], attention_scores.shape[
                -1])  # [to_seq_len, to_seq_len, d_hid]
            # 旧实现，方便读者理解维度转换
            # query_layer_t = query_layer.permute(2, 0, 1, 3)
            # query_layer_r = query_layer_t.contiguous().view(from_seq_length, batch_size * num_attention_heads, self.attention_head_size)
            # key_position_scores = torch.matmul(query_layer_r, relations_keys.permute(0, 2, 1))
            # key_position_scores_r = key_position_scores.view(from_seq_length, batch_size, num_attention_heads, from_seq_length)
            # key_position_scores_r_t = key_position_scores_r.permute(1, 2, 0, 3)
            # 新实现
            key_position_scores_r_t = torch.einsum('bnih,ijh->bnij', query_layer, relations_keys)
            attention_scores = attention_scores + key_position_scores_r_t
        elif (self.p_bias == 't5_relative') and hasattr(self, 'relative_positions_encoding'):
            relations_keys = self.relative_positions(attention_scores.shape[-1], attention_scores.shape[-1])
            key_position_scores_r_t = self.relative_positions_encoding(relations_keys).permute([2, 0, 1]).unsqueeze(0)
            attention_scores = attention_scores + key_position_scores_r_t

        """  Attention缩放  """
        if self.attention_scale:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        """  执行Attention Mask  """
        # 对于mask为0部分的attention mask，值为-1e10，经过softmax后，attention_probs几乎为0
        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        # bat_size, heads_num, seq_len, seq_len
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # batch_size, num_heads, seq_len, head_size
        context_layer = torch.matmul(attention_probs, value_layer)

        if (self.p_bias == 'typical_relative') and hasattr(self, 'relative_positions_encoding'):
            relations_values = self.relative_positions_encoding(attention_scores.shape[-1], attention_scores.shape[-1])
            # 旧实现，方便读者理解维度转换
            # attention_probs_t = attention_probs.permute(2, 0, 1, 3)
            # attentions_probs_r = attention_probs_t.contiguous().view(from_seq_length, batch_size * num_attention_heads, to_seq_length)
            # value_position_scores = torch.matmul(attentions_probs_r, relations_values)
            # value_position_scores_r = value_position_scores.view(from_seq_length, batch_size, num_attention_heads, self.attention_head_size)
            # value_position_scores_r_t = value_position_scores_r.permute(1, 2, 0, 3)
            # 新实现
            value_position_scores_r_t = torch.einsum('bnij,ijh->bnih', attention_probs, relations_values)
            context_layer = context_layer + value_position_scores_r_t
            # context_layer shape: [batch_size, query_len, num_attention_heads, attention_head_size]
            # transpose、permute等维度变换操作后，tensor在内存中不再是连续存储的，而view操作要求tensor的内存连续存储，
            # 所以在调用view之前，需要contiguous来返回一个contiguous copy；

        """ 矩阵形状还原 """
        # batch_size, num_heads, seq_len, head_size
        # 还原为 batch_size, query_len, hidden_size
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 是否返回attention scores
        if self.return_attention_scores:
            # 这里返回的attention_scores没有经过softmax, 可在外部进行归一化操作
            return self.o(context_layer), attention_scores
        else:
            return self.o(context_layer)


class BertEmbeddings(nn.Module):
    """
    embedding层
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, max_position, segment_vocab_size, shared_segment_embeddings,
                 drop_rate, conditional_size=False, **kwargs):
        """
        :param vocab_size:
        :param embedding_size:
        :param hidden_size: 兼容albert，而设置的hidden_size，为了将原有的embedding_size映射到hidden_size
        :param max_position:
        :param segment_vocab_size:
        :param shared_segment_embeddings:
        :param drop_rate:
        :param conditional_size: 使用隐向量作为条件时，需要设置输入隐向量的hidden_dim
        :param kwargs:
        """
        super(BertEmbeddings, self).__init__()
        """  Token Embeddings 词汇编码  """
        self.shared_segment_embeddings = shared_segment_embeddings
        self.vocab_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        """  Position Embeddings 位置编码  """
        if kwargs.get("p_bias") == 'sinusoid':
            self.position_embeddings = SinusoidalPositionEncoding(max_position, embedding_size)

        elif kwargs.get('p_bias') in {'rotary', 'typical_relative', 't5_relative', 'other_relative'}:
            # 如果使用相对位置编码，则不声明PositionEmbeddings，这部分将在MultiHeadAttentionLayer实现
            pass
        elif max_position > 0:
            self.position_embeddings = nn.Embedding(max_position, embedding_size)

        """  Segment Embeddings segment编码  """
        if segment_vocab_size > 0 and (not shared_segment_embeddings):
            self.segment_embeddings = nn.Embedding(segment_vocab_size, embedding_size)

        # emb_scale transform_xl, xlnet特有
        self.emb_scale = kwargs.get('emb_scale', 1)

        # LayerNorm
        self.layerNorm = LayerNormalization(hidden_size=embedding_size, condition_hidden_size=conditional_size, eps=1e-12, **kwargs)
        self.dropout = nn.Dropout(drop_rate)

        if embedding_size != hidden_size:
            self.embedding_hidden_mapping_in = nn.Linear(embedding_size, hidden_size)

    def forward(self, token_ids: Tensor, segment_ids=None, conditional_emb=None, additional_embs=None):
        # 兼容自定义word_embedding
        if not token_ids.requires_grad and token_ids.dtype in (torch.long, torch.int):
            vocab_embedding = self.vocab_embeddings(token_ids)
        else:
            vocab_embedding = token_ids

        if hasattr(self, 'segment_embeddings'):
            segment_ids = torch.zeros_like(token_ids) if segment_ids is None else segment_ids
            segment_embeddings = self.segment_embeddings(segment_ids)
            embeddings = vocab_embedding + segment_embeddings
        elif self.shared_segment_embeddings:  # segment和word_embedding共享权重
            segment_ids = torch.zeros_like(token_ids) if segment_ids is None else segment_ids
            segment_embeddings = self.word_embeddings(segment_ids)
            embeddings = vocab_embedding + segment_embeddings
        else:
            embeddings = vocab_embedding

        # 额外的embedding，如词性等
        if additional_embs is not None:
            for emb in additional_embs:
                embeddings += emb

        if hasattr(self, 'position_embeddings'):
            seq_length = token_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
            position_ids = position_ids.unsqueeze(0).repeat(token_ids.shape[0], 1)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # transform_xl, xlnet特有
        if self.emb_scale != 1:
            embeddings = embeddings * self.emb_scale

        if hasattr(self, 'layerNorm'):
            embeddings = self.layerNorm((embeddings, conditional_emb))
        embeddings = self.dropout(embeddings)

        # 用于albert
        if hasattr(self, 'embedding_hidden_mapping_in'):
            embeddings = self.embedding_hidden_mapping_in(embeddings)
        return embeddings


class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_rate=0.5, hidden_act='gelu', is_dropout=False, bias=True, **kwargs):
        # 原生的tf版本的bert在激活函数后，没有添加dropout层，但是在google AI的bert-pytorch开源项目中，多了一层dropout；
        # 并且在pytorch官方的TransformerEncoderLayer的实现中，也有一层dropout层，就像这样：self.linear2(self.dropout(self.activation(self.linear1(src))))；
        # 这样不统一做法的原因不得而知，不过有没有这一层，差别可能不会很大；

        # 为了适配是否dropout，用is_dropout，dropout_rate两个参数控制；如果是实现原始的transformer，直接使用默认参数即可；如果是实现bert，则is_dropout为False，此时的dropout_rate参数并不会使用.
        super(PositionWiseFeedForward, self).__init__()

        self.is_dropout = is_dropout
        self.intermediate_act_fn = get_activation(hidden_act)
        self.intermediateDense = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.outputDense = nn.Linear(intermediate_size, hidden_size, bias=bias)
        if self.is_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: (batch size, seq len, hidden_size)
        if self.is_dropout:
            x = self.dropout(self.intermediate_act_fn(self.intermediateDense(x)))
        else:
            x = self.intermediate_act_fn(self.intermediateDense(x))

        # x shape: (batch size, seq len, intermediate_size)
        x = self.outputDense(x)

        # x shape: (batch size, seq len, hidden_size)
        return x


class BertLayer(nn.Module):
    """
        Transformer层:
        顺序为: Attention --> Add --> LayerNorm --> Feed Forward --> Add --> LayerNorm
        注意: 1、以上都不计dropout层，并不代表没有dropout，每一层的dropout使用略有不同，注意区分
              2、原始的Transformer的encoder中的Feed Forward层一共有两层linear，
              config.intermediate_size的大小不仅是第一层linear的输出尺寸，也是第二层linear的输入尺寸
    """

    def __init__(self, hidden_size, num_attention_heads, dropout_rate, attention_probs_dropout_prob, intermediate_size,
                 hidden_act,
                 is_dropout=False, conditional_size=False, **kwargs):
        super(BertLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttentionLayer(hidden_size, num_attention_heads,
                                                          attention_probs_dropout_prob, **kwargs)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layerNorm1 = LayerNormalization(hidden_size=hidden_size, eps=1e-12, condition_hidden_size=conditional_size, **kwargs)
        self.feedForward = PositionWiseFeedForward(hidden_size, intermediate_size, dropout_rate, hidden_act,
                                                   is_dropout=is_dropout, **kwargs)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layerNorm2 = LayerNormalization(hidden_size=hidden_size, eps=1e-12, condition_hidden_size=conditional_size, **kwargs)
        self.is_decoder = kwargs.get('is_decoder')
        if self.is_decoder:
            self.crossAttention = MultiHeadAttentionLayer(hidden_size, num_attention_heads,
                                                          attention_probs_dropout_prob, **kwargs)
            self.dropout3 = nn.Dropout(dropout_rate)
            self.layerNorm3 = LayerNormalization(hidden_size=hidden_size, eps=1e-12, condition_hidden_size=conditional_size, **kwargs)

    def forward(self, hidden_states, attention_mask, conditional_emb=None, encoder_hidden_states=None,
                encoder_attention_mask=None):
        """  Attention  """
        # self.decoder为true时候，这里的attention_mask是三角的
        self_attn_output = self.multiHeadAttention(hidden_states, attention_mask)

        """  Add  """
        hidden_states = hidden_states + self.dropout1(self_attn_output)

        """  LayerNorm  """
        hidden_states = self.layerNorm1((hidden_states, conditional_emb))

        # cross attention
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attn_output = self.crossAttention(hidden_states, None, encoder_hidden_states, encoder_attention_mask)
            hidden_states = hidden_states + self.dropout3(cross_attn_output)
            hidden_states = self.layerNorm3((hidden_states, conditional_emb))

        """  Feed Forward  """
        self_attn_output2 = self.feedForward(hidden_states)

        """  Add """
        hidden_states = hidden_states + self.dropout2(self_attn_output2)

        """  LayerNorm  """
        hidden_states = self.layerNorm2((hidden_states, conditional_emb))
        return hidden_states


class Identity(nn.Module):
    """
    获取第一个元素
    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, *args):
        return args[0]