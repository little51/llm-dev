from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from tqdm import trange
from tokenizations import tokenization_bert_word_level\
    as tokenization_bert

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = None
tokenizer = None


def load_model():
    global model
    global tokenizer
    model = GPT2LMHeadModel.from_pretrained(
        "./model/final_model", torchscript=True).eval().to(device)
    tokenizer = tokenization_bert.BertTokenizer(
        vocab_file="./vocab/vocab_user.txt")


def fast_sample_sequence(model, raw_text, length,
                         temperature=1.0, top_k=30, top_p=0.0):
    # 将输入的原始文本转换为 token id 列表
    context = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(raw_text))
    # 将 token id 列表转换为 LongTensor，并调整形状为 `[1, seq_length]`
    inputs = torch.LongTensor(context).view(1, -1).to(device)
    if len(context) > 1:
        # 如果 `context` 的长度大于 1，则使用 `model` 对前
        # `seq_length-1`个 token 进行预测，并得到过去信息 `past`
        _, past = model(inputs[:, :-1], None)[:2]
        prev = inputs[:, -1].view(1, -1)
    else:
        # 否则，将 `past` 设为 `None`，并将 `prev` 设为整个输入序列
        past = None
        prev = inputs
    # 初始化 `generate` 列表为输入文本的 token id 列表
    generate = [] + context
    # 使用 `torch.no_grad()` 上下文管理器，关闭梯度计算
    # 避免在推理阶段浪费计算资源
    with torch.no_grad():
        for i in trange(length):
            # 使用 `model` 预测下一个 token，并更新过去信息 `past`
            output = model(prev, past=past)
            output, past = output[:2]
            output = output[-1].squeeze(0) / temperature
            # 将输出进行温度缩放，然后通过 `top_k_top_p_filtering`
            # 函数过滤 logits
            filtered_logits = top_k_top_p_filtering(
                output, top_k=top_k, top_p=top_p)
            # 根据过滤后的 logits，从 softmax 分布中采样一个 token
            next_token = torch.multinomial(torch.softmax(
                filtered_logits, dim=-1), num_samples=1)
            # 将采样得到的 token 添加到 `generate` 列表中，
            # 继续下一个 token 的生成
            generate.append(next_token.item())
            prev = next_token.view(1, 1)
    # 返回生成的文本序列 `generate`，其中包含了原始文本的
    # 内容和模型生成的文本内容
    return generate


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0,
                          filter_value=-float('Inf')):
    # 确保对数概率logits的维度为 1，即一维张量
    assert logits.dim() == 1
    # 将 `top_k` 限制在 `logits` 最后一个维度的大小以内
    # 以确保不超出范围
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # 找到概率小于前 `top_k` 个最大概率的 token 对应的索引
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        # 将这些 token 对应的 logits 值设为一个非常小
        # 的负无穷值 `filter_value`
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # 对 `logits` 进行降序排序，得到排序后的概率值和对应的索引
        sorted_logits, sorted_indices = torch.sort(logits,
                                                   descending=True)
        # 计算排序后概率值的累积分布
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        # 找到累积概率超过阈值 `top_p` 的 token 对应的索引
        sorted_indices_to_remove = cumulative_probs > top_p
        # 对 `sorted_indices_to_remove` 进行处理，使得保留
        # 第一个超过阈值的 token
        sorted_indices_to_remove[...,
                                 1:] = \
            sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # 根据索引找到需要移除的 token
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        # 将这些 token 对应的 logits 值设为 `filter_value`
        logits[indices_to_remove] = filter_value
    return logits


def is_word(word):
    '''
    判断生成的token是否为英文
    '''
    for item in list(word):
        if item not in 'qwertyuiopasdfghjklzxcvbnm':
            return False
    return True


def join_text(texts):
    '''
    把生成的token数组连接起来形成句子
    '''
    for i, item in enumerate(texts[:-1]):  # 确保英文前后有空格
        if is_word(item) and is_word(texts[i + 1]):
            texts[i] = item + ' '
        for i, item in enumerate(texts):
            if item == '[MASK]':
                texts[i] = ''
            elif item == '[CLS]':
                texts[i] = '\n\n'
            elif item == '[SEP]':
                texts[i] = '\n'
            elif item == '[UNK]':
                texts[i] = ''
    return ''.join(texts).replace('##', '').strip()


def generate_text(prompt, max_len, batch_size):
    '''
    以prompt开头，推理生成后续文本
    '''
    if model is None:
        load_model()
    generates = []
    for i in range(batch_size):
        generate = fast_sample_sequence(model, prompt, max_len)
        texts = tokenizer.convert_ids_to_tokens(generate)
        generates.append(join_text(texts))
    return generates


if __name__ == "__main__":
    while True:
        prompt = input("请输入文本，回车退出：")
        if prompt == "":
            break
        print(generate_text(prompt, 50, 5))
