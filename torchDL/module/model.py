from torch import nn
import torch
from torch.nn import LayerNorm
from activation import get_activation
from layer import BertLayer, BertEmbeddings, Identity
import copy
from snippets import get_kw



class BaseModel(nn.Module):

    def __init__(
            self,
            vocab_size,  # 词表大小
            hidden_size,  # 编码维度
            num_hidden_layers,  # Transformer总层数
            num_attention_heads,  # Attention的头数
            intermediate_size,  # FeedForward的隐层维度
            hidden_act,  # FeedForward隐层的激活函数
            dropout_rate=None,  # Dropout比例
            attention_probs_dropout_prob=None,  # Attention矩阵的Dropout比例
            embedding_size=None,  # 指定embedding_size, 不指定则使用config文件的参数
            attention_head_size=None,  # Attention中V的head_size
            attention_key_size=None,  # Attention中Q,K的head_size
            initializer_range=0.02,  # 权重初始化方差
            sequence_length=None,  # 是否固定序列长度
            keep_tokens=None,  # 要保留的词ID列表
            compound_tokens=None,  # 扩展Embedding
            residual_attention_scores=False,  # Attention矩阵加残差
            ignore_invalid_weights=False,  # 允许跳过不存在的权重
            keep_hidden_layers=None,  # 保留的hidden_layer层的id
            hierarchical_position=None,  # 是否层次分解位置编码
            **kwargs
    ):
        super(BaseModel, self).__init__()
        if keep_tokens is not None:
            vocab_size = len(keep_tokens)
        if compound_tokens is not None:
            vocab_size += len(compound_tokens)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size or self.hidden_size // self.num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate or 0
        self.attention_probs_dropout_prob = attention_probs_dropout_prob or 0
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.initializer_range = initializer_range
        self.sequence_length = sequence_length
        self.keep_tokens = keep_tokens
        self.compound_tokens = compound_tokens
        self.attention_bias = None
        self.position_bias = None
        self.attention_scores = None
        self.residual_attention_scores = residual_attention_scores
        self.ignore_invalid_weights = ignore_invalid_weights
        self.keep_hidden_layers = set(range(num_hidden_layers)) if keep_hidden_layers is None else set(
            keep_hidden_layers)
        self.hierarchical_position = hierarchical_position

    def load_weights(self, load_path, strict=True, prefix=None):
        state_dict = torch.load(load_path, map_location='cpu')
        if prefix is None:
            self.load_state_dict(state_dict, strict=strict)
        else:
            eval_str = 'self.variable_mapping()' if prefix == '' else f'self.{prefix}.variable_mapping()'
            mapping = {v: k for k, v in eval(eval_str).items()}
            mapping = mapping if prefix == '' else {k: f'{prefix}.{v}' for k, v in mapping.items()}
            state_dict_raw = {}
            for k, v in state_dict.items():
                k = mapping.get(k, k)
                state_dict_raw[k] = v
            self.load_state_dict(state_dict_raw, strict=strict)

    def save_weights(self, save_path, prefix=None):
        if prefix is None:
            torch.save(self.state_dict(), save_path)
        else:
            # 按照variable_mapping()中原始的key保存，方便其他官方代码加载模型
            eval_str = 'self.variable_mapping()' if prefix == '' else f'self.{prefix}.variable_mapping()'
            mapping = eval(eval_str)
            mapping = mapping if prefix == '' else {f'{prefix}.{k}': v for k, v in mapping.items()}
            state_dict_raw = {}
            for k, v in self.state_dict().items():
                k = mapping.get(k, k)
                state_dict_raw[k] = v
            torch.save(state_dict_raw, save_path)

    def init_model_weights(self, module):
        """ 初始化权重
        """
        if isinstance(module, (nn.Linear, nn.Embedding)) and (module.weight.requires_grad):
            # bert参数初始化, tf版本在linear和Embedding层使用的是截断正太分布, pytorch没有实现该函数,
            # 此种初始化对于加载预训练模型后进行finetune没有任何影响，
            # cf https://github.com/pytorch/pytorch/pull/5617
            # 固定的相对位置编码如Sinusoidal无需初始化
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            if hasattr(module, 'bias') and module.bias.requires_grad:  # T5等模型使用的是rmsnorm
                module.bias.data.zero_()
            if hasattr(module, 'weight') and module.weight.requires_grad:
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and (module.bias is not None) and (module.bias.requires_grad):
            module.bias.data.zero_()

    def load_pos_embeddings(self, embeddings):
        """根据hierarchical_position对pos_embedding进行修改
        """
        if self.hierarchical_position is not None:
            alpha = 0.4 if self.hierarchical_position is True else self.hierarchical_position
            embeddings = embeddings - alpha * embeddings[:1]
            embeddings = embeddings / (1 - alpha)
            position_index = torch.arange(self.max_position)[:, None]
            embeddings_x = torch.take_along_dim(embeddings,
                                                torch.div(position_index, embeddings.size(0), rounding_mode='trunc'),
                                                dim=0)
            embeddings_y = torch.take_along_dim(embeddings, position_index % embeddings.size(0), dim=0)
            embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y

        return embeddings

    def load_weights_from_pytorch_checkpoint(self, checkpoint, mapping=None):
        """根据mapping从checkpoint加载权重
        """
        # 加载模型文件
        file_state_dict = torch.load(checkpoint, map_location='cpu')
        mapping = mapping or self.variable_mapping()

        state_dict_new = {}
        parameters_set = set([i[0] for i in self.named_parameters()])  # 可更新的变量
        for new_key, old_key in mapping.items():
            if new_key not in self.state_dict():
                continue
            elif old_key in file_state_dict:  # mapping中包含，且模型结构中有
                state_dict_new[new_key] = self.load_variable(file_state_dict, old_key)
            elif (old_key not in file_state_dict) and (not self.ignore_invalid_weights):
                # mapping中包含，但模型文件中没有
                print(f'[WARNIMG] {old_key} not found in pretrain models')
            if new_key in parameters_set:
                parameters_set.remove(new_key)

        # 未能加载预训练权重的Parameter
        if not self.ignore_invalid_weights:
            for key in parameters_set:
                print(f'[WARNIMG] Parameter {key} not loaded from pretrain models')
        del file_state_dict

        # 将ckpt的权重load到模型结构中
        self.load_state_dict(state_dict_new, strict=False)

    def variable_mapping(self):
        raise NotImplementedError


class BERT(BaseModel):

    def __init__(
            self,
            max_position,  # 序列最大长度
            segment_vocab_size=2,  # segment总数目
            with_pool=False,  # 是否包含Pool部分
            with_nsp=False,  # 是否包含NSP部分
            with_mlm=False,  # 是否包含MLM部分
            custom_position_ids=False,  # 是否自行传入位置id
            custom_attention_mask=False,  # 是否自行传入attention_mask
            shared_segment_embeddings=False,  # 若True，则segment跟token共用embedding
            layer_norm_cond=None,  # conditional layer_norm
            layer_add_embs=None,  # addtional_embeddng, 比如加入词性，音调，word粒度的自定义embedding
            is_dropout=False,
            token_pad_ids=0,  # 默认0是padding ids, 但是注意google的mt5padding不是0
            **kwargs  # 其余参数
    ):
        super(BERT, self).__init__(**kwargs)
        self.max_position = max_position
        self.segment_vocab_size = segment_vocab_size
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.custom_position_ids = custom_position_ids
        self.custom_attention_mask = custom_attention_mask
        self.shared_segment_embeddings = shared_segment_embeddings
        self.is_dropout = is_dropout
        self.token_pad_ids = token_pad_ids
        if self.with_nsp and not self.with_pool:
            self.with_pool = True
        self.layer_norm_conds = layer_norm_cond
        self.layer_add_embs = layer_add_embs
        self.conditional_size = layer_norm_cond.weight.size(1) if layer_norm_cond is not None else None
        self.embeddings = BertEmbeddings(self.vocab_size, self.embedding_size, self.hidden_size, self.max_position,
                                         self.segment_vocab_size, self.shared_segment_embeddings,
                                         self.dropout_rate, self.conditional_size, **get_kw(BertEmbeddings, kwargs))
        kwargs['max_position'] = self.max_position  # 相对位置编码需要使用
        layer = BertLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate,
                          self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act,
                          is_dropout=self.is_dropout, conditional_size=self.conditional_size,
                          **get_kw(BertLayer, kwargs))
        self.encoderLayer = nn.ModuleList(
            [copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else Identity() for layer_id in
             range(self.num_hidden_layers)])
        if self.with_pool:
            # Pooler部分（提取CLS向量）
            self.pooler = nn.Linear(self.hidden_size, self.hidden_size)
            self.pooler_activation = nn.Tanh() if self.with_pool is True else get_activation(self.with_pool)
            if self.with_nsp:
                # Next Sentence Prediction部分
                # nsp的输入为pooled_output, 所以with_pool为True是使用nsp的前提条件
                self.nsp = nn.Linear(self.hidden_size, 2)
        else:
            self.pooler = None
            self.pooler_activation = None
        if self.with_mlm:
            self.mlmDense = nn.Linear(self.hidden_size, self.hidden_size)
            self.transform_act_fn = get_activation(self.hidden_act)
            self.mlmLayerNorm = LayerNorm(self.hidden_size, eps=1e-12, conditional_size=self.conditional_size)
            self.mlmDecoder = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            if kwargs.get('tie_emb_prj_weight') is True:
                self.mlmDecoder.weight = self.embeddings.word_embeddings.weight
            self.mlmBias = nn.Parameter(torch.zeros(self.vocab_size))
            self.mlmDecoder.bias = self.mlmBias

    def load_variable(self, state_dict, name, prefix='bert'):
        """加载单个变量的函数
        """
        variable = state_dict[name]
        if name in {
            f'{prefix}.embeddings.word_embeddings.weight',
            'cls.predictions.bias',
            'cls.predictions.decoder.weight',
            'cls.predictions.decoder.bias'
        }:
            return self.load_embeddings(variable)
        elif name == f'{prefix}.embeddings.position_embeddings.weight':
            return self.load_pos_embeddings(variable)
        elif name == 'cls.seq_relationship.weight':
            return variable.T
        else:
            return variable

    def variable_mapping(self, prefix='bert'):
        mapping = {
            'embeddings.word_embeddings.weight': f'{prefix}.embeddings.word_embeddings.weight',
            'embeddings.position_embeddings.weight': f'{prefix}.embeddings.position_embeddings.weight',
            'embeddings.segment_embeddings.weight': f'{prefix}.embeddings.token_type_embeddings.weight',
            'embeddings.layerNorm.weight': f'{prefix}.embeddings.LayerNorm.weight',
            'embeddings.layerNorm.bias': f'{prefix}.embeddings.LayerNorm.bias',
            'pooler.weight': f'{prefix}.pooler.dense.weight',
            'pooler.bias': f'{prefix}.pooler.dense.bias',
            'nsp.weight': 'cls.seq_relationship.weight',
            'nsp.bias': 'cls.seq_relationship.bias',
            'mlmDense.weight': 'cls.predictions.transform.dense.weight',
            'mlmDense.bias': 'cls.predictions.transform.dense.bias',
            'mlmLayerNorm.weight': 'cls.predictions.transform.LayerNorm.weight',
            'mlmLayerNorm.bias': 'cls.predictions.transform.LayerNorm.bias',
            'mlmBias': 'cls.predictions.bias',
            'mlmDecoder.weight': 'cls.predictions.decoder.weight',
            'mlmDecoder.bias': 'cls.predictions.decoder.bias'

        }
        for i in range(self.num_hidden_layers):
            prefix_i = f'{prefix}.encoder.layer.%d.' % i
            mapping.update({f'encoderLayer.{i}.multiHeadAttention.q.weight': prefix_i + 'attention.self.query.weight',
                            f'encoderLayer.{i}.multiHeadAttention.q.bias': prefix_i + 'attention.self.query.bias',
                            f'encoderLayer.{i}.multiHeadAttention.k.weight': prefix_i + 'attention.self.key.weight',
                            f'encoderLayer.{i}.multiHeadAttention.k.bias': prefix_i + 'attention.self.key.bias',
                            f'encoderLayer.{i}.multiHeadAttention.v.weight': prefix_i + 'attention.self.value.weight',
                            f'encoderLayer.{i}.multiHeadAttention.v.bias': prefix_i + 'attention.self.value.bias',
                            f'encoderLayer.{i}.multiHeadAttention.o.weight': prefix_i + 'attention.output.dense.weight',
                            f'encoderLayer.{i}.multiHeadAttention.o.bias': prefix_i + 'attention.output.dense.bias',
                            f'encoderLayer.{i}.layerNorm1.weight': prefix_i + 'attention.output.LayerNorm.weight',
                            f'encoderLayer.{i}.layerNorm1.bias': prefix_i + 'attention.output.LayerNorm.bias',
                            f'encoderLayer.{i}.feedForward.intermediateDense.weight': prefix_i + 'intermediate.dense.weight',
                            f'encoderLayer.{i}.feedForward.intermediateDense.bias': prefix_i + 'intermediate.dense.bias',
                            f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_i + 'output.dense.weight',
                            f'encoderLayer.{i}.feedForward.outputDense.bias': prefix_i + 'output.dense.bias',
                            f'encoderLayer.{i}.layerNorm2.weight': prefix_i + 'output.LayerNorm.weight',
                            f'encoderLayer.{i}.layerNorm2.bias': prefix_i + 'output.LayerNorm.bias'
                            })

        return mapping


import json
conf = json.load(open("../../pretrain_model/bert_base/config.json"))
model = BERT(max_position=512, **conf)
model.apply(model.init_model_weights)
model.load_weights_from_pytorch_checkpoint('../../pretrain_model/bert_base/pytorch_model.bin')
print(model)