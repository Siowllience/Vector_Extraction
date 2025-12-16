import pandas as pd
import re
import os
import pickle

# ===================== 1. 核心配置（已适配你的工作表结构） =====================
TEXT_DATA_PATH = "text_data.xlsx"
WORDS_CSV_OUTPUT_PATH = "words.csv"
WORDS_LIST_OUTPUT_PATH = "words_list.p"

# 有效文章工作表映射（无需修改）
SHEET_TO_ARTICLE = {
    "Mars_Text_310": "ART-001",
    "Supertanker_Text_302": "ART-002",
    "Math_Text_306": "ART-003",
    "GPS_Text_307": "ART-004",
    "Circuit_Text_302": "ART-005"
}
ARTICLE_TO_NUM = {"ART-001":1, "ART-002":2, "ART-003":3, "ART-004":4, "ART-005":5}

# ===================== 2. 核心工具函数（优化SentenceID提取） =====================
def concat_words_to_sentence(word_series):
    """拼接单词为句子，优化标点处理"""
    sentence = " ".join([str(w).strip() for w in word_series.tolist() if str(w).strip()])
    # 修复常见标点格式（如"word ."→"word."、"word ,"→"word,"）
    sentence = re.sub(r' (\.|,|!|\?)', r'\1', sentence)
    return sentence.strip()

def convert_sentence_id(article_num: int, original_snt_id: str) -> str:
    """优化SentenceID提取，适配更多格式（如"1"、"01"、"s1"、"sent1"）"""
    original_snt_id = str(original_snt_id).strip()
    # 提取所有数字（适配"sent01"→"01"→1，"s_12"→"12"→12）
    snt_nums = re.findall(r'\d+', original_snt_id)
    if not snt_nums:
        raise ValueError(f"无法提取句子编号：原始SentenceID={original_snt_id}（无数字）")
    # 取最后一组数字（避免"t01s05"提取到"01"和"05"，优先取句子序号"05"）
    snt_num = int(snt_nums[-1])
    return f"t.{article_num:02d}.{snt_num:02d}"

# ===================== 3. 主流程：完整读取+调试可视化 =====================
def main():
    # 步骤1：创建输出目录
    output_dir = os.path.dirname(WORDS_CSV_OUTPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录：{output_dir}\n")

    # 步骤2：读取Excel并筛选有效工作表
    try:
        # 强制使用openpyxl引擎读取.xlsx，避免旧引擎数据丢失
        excel_file = pd.ExcelFile(TEXT_DATA_PATH, engine="openpyxl")
        all_sheets = excel_file.sheet_names
        valid_sheets = [s for s in SHEET_TO_ARTICLE.keys() if s in all_sheets]
        if len(valid_sheets) != 5:
            print(f"错误：有效工作表仅{len(valid_sheets)}个（需5个），缺失：{set(SHEET_TO_ARTICLE.keys())-set(all_sheets)}")
            return
        print(f"Excel包含工作表：{all_sheets}")
        print(f"有效工作表（5个）：{valid_sheets}\n")
    except Exception as e:
        print(f"读取Excel失败：{str(e)}（建议检查文件是否损坏或路径是否正确）")
        return

    all_sentences = []
    for sheet_name in valid_sheets:
        article_id = SHEET_TO_ARTICLE[sheet_name]
        article_num = ARTICLE_TO_NUM[article_id]
        print("="*50)
        print(f"处理工作表：{sheet_name}（文章{article_id}，编号{article_num}）")
        print("="*50)

        # 步骤3：完整读取工作表数据（关键优化：避免数据丢失）
        try:
            # header=0：第一行为表头；skiprows=None：不跳过任何行；dtype=str：避免数字格式丢失
            df_sheet = pd.read_excel(
                TEXT_DATA_PATH, 
                sheet_name=sheet_name, 
                engine="openpyxl",
                header=0, 
                skiprows=None,
                dtype=str  # 强制所有字段为字符串，避免SentenceID（如"01"）被转为数字1
            )
            # 重置索引，避免空行导致的索引混乱
            df_sheet = df_sheet.reset_index(drop=True)
            print(f"1. 读取数据：共{len(df_sheet)}行，原始字段名：{list(df_sheet.columns)}")

            # 检查必要字段（不区分大小写，适配"sentenceid"、"SentenceID"等情况）
            df_sheet.columns = [col.strip().lower() for col in df_sheet.columns]  # 字段名转小写
            required_cols = ["sentenceid", "word"]
            missing_cols = [col for col in required_cols if col not in df_sheet.columns]
            if missing_cols:
                print(f"错误：缺少必需字段（需{required_cols}，当前字段：{list(df_sheet.columns)}）")
                continue

            # 步骤4：过滤无效数据（空行、SentenceID/Word为空）
            # 重命名字段为统一格式
            df_sheet = df_sheet.rename(columns={"sentenceid": "SentenceID", "word": "Word"})
            # 过滤空值：SentenceID或Word为空的行
            df_sheet = df_sheet[
                (df_sheet["SentenceID"].notna()) & 
                (df_sheet["SentenceID"].str.strip() != "") & 
                (df_sheet["Word"].notna()) & 
                (df_sheet["Word"].str.strip() != "")
            ].reset_index(drop=True)
            print(f"2. 过滤后数据：共{len(df_sheet)}行（剔除空行/空值）")

            # 查看SentenceID分布（关键调试：确认是否有多个句子）
            unique_snt_ids = df_sheet["SentenceID"].unique()
            print(f"3. 包含句子数量：{len(unique_snt_ids)}个（SentenceID列表前10个：{unique_snt_ids[:10]}）")
            if len(unique_snt_ids) == 0:
                print("错误：无有效SentenceID，跳过该工作表")
                continue

            # 步骤5：按SentenceID分组，拼接句子（核心步骤）
            df_sentences = df_sheet.groupby("SentenceID").agg(
                OriginalText=("Word", concat_words_to_sentence),  # 拼接句子
                WordCount=("Word", lambda x: len([w for w in x if str(w).strip()]))  # 统计有效单词数
            ).reset_index()
            print(f"4. 分组后句子数：{len(df_sentences)}个（每个SentenceID对应1个句子）")

            # 步骤6：添加ArticleID和转换后的SentenceID
            df_sentences["ArticleID"] = article_num
            df_sentences["ConvertedSentenceID"] = df_sentences["SentenceID"].apply(
                lambda x: convert_sentence_id(article_num, x)
            )
            # 筛选有效句子（单词数≥1）
            df_sentences = df_sentences[df_sentences["WordCount"] >= 1].reset_index(drop=True)
            print(f"5. 最终有效句子数：{len(df_sentences)}个（单词数≥1）")
            if len(df_sentences) > 0:
                print(f"   示例句子：{df_sentences.iloc[0]['ConvertedSentenceID']} → {df_sentences.iloc[0]['OriginalText'][:50]}...")

            all_sentences.append(df_sentences)
            print(f"\n✅ 工作表{sheet_name}处理完成！\n")

        except Exception as e:
            print(f"处理工作表{sheet_name}出错：{str(e)}")
            continue

    # 步骤7：合并所有句子并生成words.csv
    if not all_sentences:
        print("\n❌ 未提取到任何有效句子，请检查工作表数据！")
        return

    df_words = pd.concat(all_sentences, ignore_index=True)
    # 调整字段顺序并去重
    df_words = df_words[["ArticleID", "ConvertedSentenceID", "WordCount", "OriginalText"]]
    df_words.columns = ["ArticleID", "SentenceID", "WordCount", "OriginalText"]
    df_words = df_words.drop_duplicates(subset=["SentenceID"], keep="first").reset_index(drop=True)

    # 步骤8：保存words.csv
    df_words.to_csv(WORDS_CSV_OUTPUT_PATH, index=False, encoding="utf-8-sig")  # utf-8-sig兼容Excel打开
    print(f"\n📊 所有文章汇总：")
    print(f"- 总有效句子数：{len(df_words)}个")
    print(f"- 各文章句子数：")
    for art_num in sorted(ARTICLE_TO_NUM.values()):
        art_sent_count = len(df_words[df_words["ArticleID"] == art_num])
        print(f"  文章{art_num}：{art_sent_count}个句子")
    print(f"\n✅ words.csv已保存：{WORDS_CSV_OUTPUT_PATH}")

    # 步骤9：生成words_list.p（按文章顺序的句子长度）
    words_list = []
    for art_num in sorted(ARTICLE_TO_NUM.values()):
        art_sentences = df_words[df_words["ArticleID"] == art_num].copy()
        # 按SentenceID中的句子序号排序（t.01.01→t.01.02→...）
        art_sentences["SntNum"] = art_sentences["SentenceID"].apply(lambda x: int(x.split(".")[-1]))
        art_sentences_sorted = art_sentences.sort_values("SntNum")
        words_list.extend(art_sentences_sorted["WordCount"].tolist())

    with open(WORDS_LIST_OUTPUT_PATH, "wb") as f:
        pickle.dump(words_list, f)
    print(f"✅ words_list.p已保存：{WORDS_LIST_OUTPUT_PATH}（长度：{len(words_list)}）")

if __name__ == "__main__":
    main()