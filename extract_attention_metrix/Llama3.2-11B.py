import torch
from transformers import (
    AutoTokenizer, AutoConfig,
    MllamaForCausalLM,
    BitsAndBytesConfig  # 新增：导入量化配置类
)
import numpy as np
from torch.nn import functional as F
import sys
import os

# ========== 1. 基础配置+修复目录创建 ==========
model_path = "/data/home/jinyuehan/LLMBrain/LLMmodel/Llama/llama3.2/11B/LLM-Research/Llama-3.2-11B-Vision"
model_name = model_path.split('/')[-1]
save_dir="/data/home/jinyuehan/LLMBrain/12.18/attention_LLM/Llama/llama3.2/11B"
os.makedirs(save_dir, exist_ok=True)  # 递归创建目录
has_instruction = False
token_begin = "Ġ"  # Llama3.2的token前缀是Ġ
has_bos = True

# ========== 2. 指令配置（保持不变） ==========
instruction = 'Please translate sentence into German:'
prefix = 'instr_' if has_instruction else ''

# ========== 3. 保留原函数 ==========
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

# ========== 4. 读取句子（保持不变） ==========
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

# ========== 5. Tokenizer初始化（适配mllama） ==========
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code=True,
    use_fast=False,  # 多模态模型必须用slow Tokenizer
    padding_side="right",
    add_bos_token=True,
    add_eos_token=False,
    legacy=False
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ========== 6. 模型加载（核心修复：简化量化配置） ==========
# 创建8bit量化配置（移除无效参数，仅保留核心）
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True  # 仅保留有效参数，消除警告
)

# 加载原生多模态配置
config = AutoConfig.from_pretrained(
    model_path,
    trust_remote_code=True
)

# 用MllamaForCausalLM加载模型（移除无效的trust_remote_code）
model = MllamaForCausalLM.from_pretrained(
    model_path, 
    config=config,
    device_map="auto", 
    quantization_config=bnb_config,  # 新版量化配置
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="eager"
)
model.eval()

# 先获取config中的层数（仅作参考），实际以attention返回为准
config_n_layer = model.config.num_hidden_layers
n_head = model.config.num_attention_heads
print(f"✅ 加载Llama3.2-11B-Vision成功：config层数={config_n_layer}，注意力头数={n_head}")

# ========== 7. 注意力提取（核心修复：适配实际attention层数 + BFloat16转float32） ==========
tensor_per_article = []  # 动态初始化，适配实际层数

for article in article_sentences:
    attention_per_sentence = []  # 每个article的attention列表
    for idx, sentence in enumerate(article):
        s_words = sentence.split()
        s_tokens = [x.replace(token_begin, '') for x in tokenizer.tokenize(sentence)]
        s_groups = token_groups(s_words, s_tokens)

        # 编码文本（纯文本输入，忽略视觉分支）
        if not has_instruction:
            encoded_input = tokenizer(sentence, return_tensors='pt').to('cuda')
        else:
            encoded_input = tokenizer(' '.join([instruction, sentence]), return_tensors='pt').to('cuda')
        
        # 前向传播获取attention（多模态模型仅返回文本分支的attention）
        outputs = model(**encoded_input, output_attentions=True)
        attentions = outputs.attentions
        actual_n_layer = len(attentions)  # 实际返回的attention层数（32）
        
        # 第一次处理sentence时，初始化层数相关列表
        if idx == 0:
            # 初始化当前article的attention_per_sentence
            attention_per_sentence = [[] for _ in range(actual_n_layer)]
            # 全局tensor_per_article仅初始化一次
            if len(tensor_per_article) == 0:
                tensor_per_article = [[] for _ in range(actual_n_layer)]
                print(f"📌 实际有效attention层数: {actual_n_layer}（多模态模型文本分支）")
        
        # 遍历实际的attention层数（32层）
        for lyr in range(actual_n_layer):
            print(f'Layer {lyr}:')
            attn_tensor = attentions[lyr].detach().cpu()
            
            # ========== 关键修复1：BFloat16转float32后再检查NaN ==========
            attn_tensor_float = attn_tensor.float()  # 转换为NumPy支持的float32
            assert not np.isnan(np.sum(attn_tensor_float.numpy())), f"layer {lyr} has NaN attentions"
            
            assert attn_tensor.size()[1] == n_head
            attn_tensor_squeezed = attn_tensor.squeeze()
            print(attn_tensor_squeezed.size())

            # ========== 关键修复2：BFloat16转float32后再转numpy ==========
            attn_tensor_squeezed_float = attn_tensor_squeezed.float()  # 转float32
            # 移除BOS token的attention（如果有）
            if not has_instruction:
                attn_array = attn_tensor_squeezed_float.numpy()[:, 1:, 1:] if has_bos else attn_tensor_squeezed_float.numpy()
            else:
                attn_array = attn_tensor_squeezed_float.numpy()[:, len_instruct + 1:, len_instruct + 1:] if has_bos else attn_tensor_squeezed_float.numpy()

            print(attn_array.shape)
            list_head_attn = []
            for head in range(n_head):
                head_attn_array = attn_array[head]
                merged_head_attn_array = merge_attentions(head_attn_array, s_groups)
                list_head_attn.append(merged_head_attn_array)
            merged_attn_array = np.stack(list_head_attn)
            print(merged_attn_array.shape, len(s_words))
            attn_tensor_new = torch.tensor(merged_attn_array)
            pad_len = max_sent_len - attn_tensor_new.size()[-1]

            # 填充到统一长度
            attn_tensor_padded = F.pad(attn_tensor_new, (0, pad_len, 0, pad_len))
            print(attn_tensor_padded.size())
            attention_per_sentence[lyr].append(attn_tensor_padded)

    # 堆叠当前article的所有sentence的attention
    for lyr in range(actual_n_layer):
        tensors_to_stack = attention_per_sentence[lyr]
        stacked_tensors = torch.stack(tensors_to_stack)
        pad_n = max_n_sents - len(tensors_to_stack)
        # 填充到max_n_sents长度
        stacked_tensors = F.pad(stacked_tensors, (0, 0, 0, 0, 0, 0, 0, pad_n))
        print(f'Layer {lyr} article shape: {stacked_tensors.size()}')
        tensor_per_article[lyr].append(stacked_tensors)

# ========== 8. 保存结果（适配实际层数） ==========
for lyr in range(actual_n_layer):
    tensors_to_stack = tensor_per_article[lyr]
    stacked_tensors = torch.stack(tensors_to_stack)
    stacked_np = stacked_tensors.numpy()
    print(f'All final shape (Layer {lyr}): {stacked_np.shape}')
    np.save(f'{save_dir}/{prefix}rb_p1_layer{lyr}.npy', stacked_np)

print(f"✅ 所有层保存完成！共保存 {actual_n_layer} 层（多模态模型文本分支有效层数）")