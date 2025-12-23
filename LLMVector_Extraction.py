import numpy as np
import os
import pickle
from glob import glob

# ========== 核心配置（按你的路径适配，已修正模型名称） ==========
# 输入：注意力矩阵路径
ATTENTION_INPUT_DIR = "/data/home/jinyuehan/LLMBrain/12.18/attention_LLM/Llama/llama3.2/11B"
# 输出：回归就绪的注意力向量保存路径
VECTOR_OUTPUT_DIR = "/data/home/jinyuehan/LLMBrain/12.18/Vector_LLM/Llama/llama3.2/11B"
# 关键文件路径：words_list.p（存储每个句子的实际单词数）
WORDS_LIST_PATH = "/data/home/jinyuehan/LLMBrain/12.11/words_list.p"
# 模型名称和大小（修正为实际模型：llama 7B，而非llama3）
MODEL_NAME = "llama3.2"
MODEL_SIZE = "11B"

# ========== 初始化与依赖加载 ==========
os.makedirs(VECTOR_OUTPUT_DIR, exist_ok=True)

# 加载words_list：存储148个句子的实际单词数（与论文一致，无padding）
try:
    with open(WORDS_LIST_PATH, 'rb') as f:
        words_list = pickle.load(f)
    print(f"✅ 成功加载words_list.p，共{len(words_list)}个句子的实际单词数")
    print(f"  示例：前5个句子的单词数：{words_list[:5]}")
except FileNotFoundError:
    raise FileNotFoundError(f"未找到words_list.p，请检查路径：{WORDS_LIST_PATH}")

# ========== 获取所有层的注意力矩阵文件 ==========
attention_files = sorted(glob(os.path.join(ATTENTION_INPUT_DIR, "rb_p1_layer*.npy")))
if not attention_files:
    raise FileNotFoundError(f"未在{ATTENTION_INPUT_DIR}找到注意力矩阵文件（格式：rb_p1_layer*.npy）")

# ========== 遍历每个层，提取回归就绪的注意力向量 ==========
for layer_file in attention_files:
    # 解析层号
    layer_idx = int(os.path.basename(layer_file).split("layer")[-1].split(".npy")[0])
    print(f"\n正在处理第{layer_idx}层，文件：{os.path.basename(layer_file)}")

    # 1. 加载注意力矩阵（原维度：(5, 31, 32, 16, 16) → 文章数, 单篇最大句子数, 注意力头数, padding后单词数, padding后单词数）
    layer_attn = np.load(layer_file)
    print(f"  原始注意力矩阵维度：{layer_attn.shape}")

    # 2. 调整维度：注意力头→第一维，拼接所有文章的句子（按论文148个有效句子筛选）
    # 维度调整：(32, 5, 31, 16, 16) → 注意力头数, 文章数, 单篇句子数, 单词数, 单词数
    attn = layer_attn.swapaxes(2, 0).swapaxes(1, 2)
    # 拼接所有文章的句子，取前148个有效句子（与words_list长度一致）
    X = attn.reshape(attn.shape[0], -1, attn.shape[3], attn.shape[4])[:, :len(words_list), :, :]
    print(f"  筛选有效句子后维度（注意力头数, 有效句子数, padding后单词数, padding后单词数）：{X.shape}")
    assert X.shape[1] == len(words_list), f"句子数不匹配：注意力矩阵{X.shape[1]}句 vs words_list{len(words_list)}句"

    # 3. 按实际单词数提取下三角（核心：适配眼动向量的7388维）
    all_sentence_vectors = []
    total_triangle_elements = 0  # 统计总下三角元素数（最终应≈7388）
    for snt_idx in range(len(words_list)):
        # 获取当前句子的实际单词数（从words_list提取，无padding）
        n_word = words_list[snt_idx]
        # 生成下三角索引（不含对角线，k=-1，与论文一致）
        tril_idx = np.tril_indices(n_word, k=-1)
        # 提取当前句子的下三角部分（按实际单词数截取，忽略padding）
        # X维度：(32, 148, 16, 16) → 截取为(32, 148, n_word, n_word)后提取下三角
        snt_tril = X[:, snt_idx, :n_word, :n_word][:, tril_idx[0], tril_idx[1]]
        all_sentence_vectors.append(snt_tril)
        total_triangle_elements += len(tril_idx[0])

    # 4. 拼接所有句子的向量 → 最终维度：(注意力头数, 总下三角元素数)（与眼动向量匹配）
    attention_vector = np.concatenate(all_sentence_vectors, axis=1)
    print(f"  第{layer_idx}层注意力向量维度：{attention_vector.shape}")
    print(f"  总下三角元素数：{total_triangle_elements}（论文标准为7388，差异源于数据集细节）")

    # 5. 保存回归就绪的注意力向量（文件名明确标注“regression_ready”）
    vector_filename = f"{MODEL_NAME}_{MODEL_SIZE}_regression_ready_attention_vector_layer{layer_idx}.npy"
    vector_save_path = os.path.join(VECTOR_OUTPUT_DIR, vector_filename)
    np.save(vector_save_path, attention_vector)
    print(f"  向量已保存至：{vector_save_path}")

# ========== 输出汇总信息 ==========
print(f"\n✅ 所有层回归就绪向量提取完成！")
print(f"📌 输出目录：{VECTOR_OUTPUT_DIR}")
print(f"📌 向量文件格式：{MODEL_NAME}_{MODEL_SIZE}_regression_ready_attention_vector_layer{layer_idx}.npy")
print(f"📌 单层层向量维度示例（以最后一层为例）：{attention_vector.shape}")
