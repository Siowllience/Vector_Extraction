import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from torch.nn import functional as F
import os

# ========== 基础配置 ==========
model_path = "/data/home/jinyuehan/LLMBrain/LLMmodel/LLaDA/llada-7b"
save_dir="/data/home/jinyuehan/LLMBrain/12.18/attention_LLM/LLaDA/7B"
if not os.path.exists(save_dir): 
    os.mkdir(save_dir)
has_instruction = False
token_begin = "▁"
has_bos = True
prefix = 'instr_' if has_instruction else ''

# ========== 保留原token_groups/merge_attentions函数 ==========
def token_groups(words, tokens):
    groups = []
    words_iter = iter(words)
    word = next(words_iter)
    text_buf = ''
    id_buf = []
    for i, token in enumerate(tokens):
        if text_buf != word:
            text_buf = text_buf + token
            id_buf.append(i)
        else:
            groups.append(id_buf.copy())
            text_buf = token
            id_buf = [i]
            word = next(words_iter)
    groups.append(id_buf.copy())
    return groups

def merge_attentions(attn_mat, tok_groups):
    arrays = []
    for group in tok_groups:
        array = 0
        for i in group:
            array += attn_mat[:, i]
        arrays.append(array)
    mat1 = np.stack(arrays).T
    arrays = []
    for group in tok_groups:
        array = np.mean([mat1[i, :] for i in group], axis=0)
        arrays.append(array)
    mat2 = np.stack(arrays)
    return mat2

# ========== 读取句子（原逻辑不变） ==========
with open(f'/data/home/jinyuehan/LLMBrain/12.18/reading_brain_sentences.txt', 'r') as f:
    articles = f.read().strip().split('\n\n')
article_sentences = []
n_sents = []
n_words = []
for article in articles:
    sentences = article.strip().split('\n')
    article_sentences.append(sentences)
    n_sents.append(len(sentences))
    for sentence in sentences:
        n_words.append(len(sentence.strip().split()))
max_n_sents = max(n_sents)
max_sent_len = max(n_words)

# ========== Tokenizer初始化 ==========
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code=True,
    use_fast=False,
    padding_side="right",
    add_bos_token=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ========== 模型加载 ==========
model = AutoModel.from_pretrained(
    model_path, 
    device_map="auto", 
    load_in_8bit=False,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
model.eval()
n_layer = model.config.num_hidden_layers
n_head = model.config.num_attention_heads
print(f"✅ 加载LLaDA-7B成功：层数={n_layer}，注意力头数={n_head}")

# ========== 迭代提取注意力 ==========
tensor_per_article = [[] for j in range(n_layer)]
for article in article_sentences:
    attention_per_sentence = [[] for j in range(n_layer)]
    for sentence in article:
        s_words = sentence.split()
        s_tokens = [x.replace(token_begin, '') for x in tokenizer.tokenize(sentence)]
        s_groups = token_groups(s_words, s_tokens)

        # 扩散模型forward（核心修改：传入timestep）
        if not has_instruction:
            encoded_input = tokenizer(sentence, return_tensors='pt').to('cuda')
        else:
            encoded_input = tokenizer(' '.join([instruction, sentence]), return_tensors='pt').to('cuda')
        timestep = torch.tensor([100], device="cuda")  # 固定扩散步数
        outputs = model(**encoded_input, timestep=timestep, output_attentions=True)
        attentions = outputs.attentions

        for lyr in range(n_layer):
            attn_tensor = attentions[lyr].detach().cpu()
            # NaN检查
            if np.isnan(np.sum(attn_tensor.numpy())):
                print(f"⚠️ Layer {lyr} has NaN, replace with 0")
                attn_tensor = torch.zeros_like(attn_tensor)
            attn_tensor = attn_tensor.squeeze()
            # 移除BOS token（与原逻辑一致）
            if not has_instruction:
                attn_array = attn_tensor.numpy()[:, 1:, 1:] if has_bos else attn_tensor.numpy()
            else:
                attn_array = attn_tensor.numpy()[:, len_instruct + 1:, len_instruct + 1:] if has_bos else attn_tensor.numpy()[:, len_instruct:, len_instruct:]
            # 子词合并（原逻辑不变）
            list_head_attn = []
            for head in range(n_head):
                head_attn_array = attn_array[head]
                merged_head_attn_array = merge_attentions(head_attn_array, s_groups)
                list_head_attn.append(merged_head_attn_array)
            merged_attn_array = np.stack(list_head_attn)
            attn_tensor = torch.tensor(merged_attn_array)
            pad_len = max_sent_len - attn_tensor.size()[-1]
            attn_tensor = F.pad(attn_tensor, (0, pad_len, 0, pad_len))
            attention_per_sentence[lyr].append(attn_tensor)

    # 拼接文章内句子（原逻辑不变）
    for lyr in range(n_layer):
        tensors_to_stack = attention_per_sentence[lyr]
        stacked_tensors = torch.stack(tensors_to_stack)
        pad_n = max_n_sents - len(tensors_to_stack)
        stacked_tensors = F.pad(stacked_tensors, (0, 0, 0, 0, 0, 0, 0, pad_n))
        tensor_per_article[lyr].append(stacked_tensors)

# 保存注意力矩阵（原逻辑不变）
for lyr in range(n_layer):
    tensors_to_stack = tensor_per_article[lyr]
    stacked_tensors = torch.stack(tensors_to_stack)
    stacked_np = stacked_tensors.numpy()
    np.save(f'{save_dir}/{prefix}rb_p1_layer{lyr}.npy', stacked_np)
print(f"✅ LLaDA-7B注意力提取完成，保存至：{save_dir}")