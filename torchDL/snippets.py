import numpy as np
import jieba
from rouge import Rouge
import re

# 计算rouge用
rouge = Rouge()


def compute_main_metric(source, target, unit='word'):
    """计算主要metric
    # 指标名
    metric_keys = ['main', 'rouge-1', 'rouge-2', 'rouge-l']
    """

    def compute_rouge(source, target, unit='word'):
        """计算rouge-1、rouge-2、rouge-l
        """
        if unit == 'word':
            source = jieba.cut(source, HMM=False)
            target = jieba.cut(target, HMM=False)
        source, target = ' '.join(source), ' '.join(target)
        try:
            scores = rouge.get_scores(hyps=source, refs=target)
            return {
                'rouge-1': scores[0]['rouge-1']['f'],
                'rouge-2': scores[0]['rouge-2']['f'],
                'rouge-l': scores[0]['rouge-l']['f'],
            }
        except ValueError:
            return {
                'rouge-1': 0.0,
                'rouge-2': 0.0,
                'rouge-l': 0.0,
            }

    def compute_metrics(source, target, unit='word'):
        """计算所有metrics
        """
        metrics = compute_rouge(source, target, unit)
        metrics['main'] = (
                metrics['rouge-1'] * 0.2 + metrics['rouge-2'] * 0.4 +
                metrics['rouge-l'] * 0.4
        )
        return metrics

    return compute_metrics(source, target, unit)['main']


def extract_matching(texts, summaries, start_i=0, start_j=0):
    """在texts中找若干句子，使得它们连起来与summaries尽可能相似
    算法：texts和summaries都分句，然后找出summaries最长的句子，在texts
          中找与之最相似的句子作为匹配，剩下部分递归执行。
    """
    if len(texts) == 0 or len(summaries) == 0:
        return []
    i = np.argmax([len(s) for s in summaries])
    j = np.argmax([compute_main_metric(t, summaries[i], 'char') for t in texts])
    lm = extract_matching(texts[:j + 1], summaries[:i], start_i, start_j)
    rm = extract_matching(
        texts[j:], summaries[i + 1:], start_i + i + 1, start_j + j
    )
    return lm + [(start_i + i, start_j + j)] + rm


def text_split(text):
    """
    将长句按照标点分割为多个子句。
    :param text: 预计，6月26日20时至27日20时，四川盆地东部和南部、内蒙古东南部等地有大到暴雨。
    :return: ['预计，', '6月26日20时至27日20时，', '四川盆地东部和南部、内蒙古东南部等地有大到暴雨。']
    """
    ret_lst = []
    split_pattern = re.compile(r'[\n。；：？、"！，,:;?!“”]')
    idx_lst = [i.span()[0] for i in split_pattern.finditer(text)][::-1]
    for idx in idx_lst:
        ret_lst.append(text[idx + 1:])
        text = text[:idx + 1]
    ret_lst.append(text)
    ret_lst = [line for line in ret_lst if line]
    return ret_lst[::-1]


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """将序列padding到同一长度
    """
    if isinstance(inputs[0], (np.ndarray, list)):
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)
