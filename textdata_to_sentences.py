import pandas as pd
import csv  # 关键：导入csv模块处理引号
import os
import re

# ========== 核心配置（无需修改） ==========
INPUT_CSV_PATH = "words.csv"
OUTPUT_TXT_PATH = "reading_brain_sentences.txt"

# ========== 步骤1：正确读取CSV（修复QUOTE_ALL报错） ==========
# 关键修改：用csv.QUOTE_ALL替代pd.QUOTE_ALL，处理带双引号的字段
df = pd.read_csv(
    INPUT_CSV_PATH,
    usecols=["ArticleID", "SentenceID", "OriginalText"],  # 严格匹配你的CSV列名
    quoting=csv.QUOTE_ALL,  # 修复：调用csv模块的QUOTE_ALL
    encoding="utf-8"
)

# 数据清洗：去空值、去重
df = df.dropna(subset=["ArticleID", "SentenceID", "OriginalText"])
df = df.drop_duplicates(subset=["SentenceID"])
print(f"✅ 读取成功：共{df['ArticleID'].nunique()}篇文章，{len(df)}个句子")

# ========== 步骤2：清洗句子（去引号/标点/多余空格） ==========
def clean_sentence(sentence):
    """清洗规则：1.去双引号 2.去所有标点 3.规整空格"""
    # 1. 去除句子中的双引号（核心处理CSV里的带引号句子）
    sentence_clean = sentence.replace('"', '')
    # 2. 去除所有标点符号（? . , 等）
    sentence_clean = re.sub(r'[^\w\s]', '', sentence_clean)
    # 3. 去除多余空格（多个连续空格→单个，首尾空格去掉）
    sentence_clean = re.sub(r'\s+', ' ', sentence_clean).strip()
    return sentence_clean

df["clean_sentence"] = df["OriginalText"].apply(clean_sentence)

# ========== 步骤3：按文章+句子序号排序（保证顺序正确） ==========
def extract_sentence_order(sent_id):
    """解析SentenceID的序号（如t.01.10 → 10）"""
    parts = sent_id.split(".")
    return int(parts[2]) if len(parts)>=3 else 0

df["sentence_order"] = df["SentenceID"].apply(extract_sentence_order)
df = df.sort_values(by=["ArticleID", "sentence_order"])  # 按文章+句子序号排序

# ========== 步骤4：生成目标txt格式 ==========
# 按文章分组，句子用\n分隔，文章用\n\n分隔
article_sentences = []
for article_id, group in df.groupby("ArticleID"):
    sentences = group["clean_sentence"].tolist()
    article_text = "\n".join(sentences)
    article_sentences.append(article_text)

final_text = "\n\n".join(article_sentences)

# ========== 步骤5：写入文件 ==========
os.makedirs(os.path.dirname(OUTPUT_TXT_PATH), exist_ok=True)
with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
    f.write(final_text)

# ========== 验证结果 ==========
print(f"\n✅ 转换完成！")
print(f"📌 输入CSV：{INPUT_CSV_PATH}")
print(f"📌 输出TXT：{OUTPUT_TXT_PATH}")
print(f"📊 统计：{len(article_sentences)}篇文章，{len(df)}个句子")

# 预览前5行输出
print("\n📄 输出文件前5行预览：")
with open(OUTPUT_TXT_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i < 5:
            print(line.strip())
        else:
            break