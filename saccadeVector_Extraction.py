import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import zscore
import warnings
import time

# 禁用无关警告，与官方脚本保持一致
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# ===================== 1. 核心参数配置（仅修改为动态路径模板，逻辑不变） =====================
# 固定根路径（替换为你的实际根路径）
ROOT_DS_DIR = "/data/home/jinyuehan/LLMBrain/11.25/reading_brain_datalad/ds003974" #原始数据路径
WORDS_CSV_PATH = "words.csv" #words.csv路径
WORDS_LIST_PATH = "words_list.p" #words_list.p路径
# 向量输出根目录（确保该目录已创建）
OUTPUT_ROOT_DIR = "saccadeVector"  #输出路径

# 官方固定参数（完全复制，无任何修改）
arti_num, snt_max, word_max, n_subj = 5, 31, 16, 51
split_idx = int(148 * 0.9)
snts_list = [31, 31, 28, 28, 30]
cums = np.cumsum(snts_list)

# 定义需要处理的被试ID列表：1-20 + 22-52
TARGET_SUBJ_IDS = list(range(1, 21)) + list(range(22, 53))


# ===================== 2. 核心函数（仅修改路径为动态传入，处理逻辑完全不变） =====================
def load_subject_events(subj, arti_num, subj_events_dir):
    """仅新增subj_events_dir参数，其余逻辑完全不变"""
    events_all = []
    for article in range(1, arti_num + 1):
        for run in range(1, 6):
            # 动态生成当前被试的TSV路径
            fname = os.path.join(subj_events_dir, f"{subj}_task-read_run-{run}_events.tsv")
            if not os.path.exists(fname):
                continue
            df = pd.read_csv(fname, delimiter='\t').dropna().reset_index(drop=True)
            if int(df.SentenceID.iloc[0].split('.')[1]) == article:
                df['article'] = article
                events_all.append(df)
                break
    if not events_all:
        return None  # 无数据时返回None，避免报错中断循环
    return pd.concat(events_all, ignore_index=True)


def extract_subj_saccade_vectors(subj_id, words, words_list):
    """处理逻辑完全不变，仅动态生成路径"""
    # 生成被试名称（sub-01/sub-10等格式）
    subj = 'sub-0%s' % subj_id if subj_id < 10 else 'sub-%s' % subj_id
    print(f"\n{'='*50} 开始处理被试：{subj} {'='*50}")
    start_time = time.time()

    # 动态生成当前被试的眼动数据目录
    subj_events_dir = os.path.join(ROOT_DS_DIR, subj, "func")
    if not os.path.exists(subj_events_dir):
        print(f"❌ 被试{subj}的眼动数据目录不存在：{subj_events_dir}，跳过")
        return None, None

    # 加载眼动数据（逻辑不变）
    events = load_subject_events(subj, arti_num, subj_events_dir)
    if events is None:
        print(f"❌ 被试{subj}未加载到有效眼动数据，跳过")
        return None, None
    print(f"✅ 成功加载眼动数据：共{len(events)}行注视记录")

    # 初始化存储列表（逻辑不变）
    y_num_train, y_num_test = [], []
    y_dur_train, y_dur_test = [], []

    for article in range(1, arti_num + 1):
        print(f"\n处理文章{article}/{arti_num}")
        article_snts = words[words.SentenceID.str.match(f't.0{article}')]
        n_snts = int(article_snts.SentenceID.iloc[-1].split('.')[-1])
        print(f"  文章{article}共{n_snts}个句子")

        for i in range(1, n_snts + 1):
            snt_id = f't.0{article}.0{i}' if i < 10 else f't.0{article}.{i}'
            event = events[events.SentenceID.str.match(snt_id)].copy()
            if event.empty:
                print(f"  句子{snt_id}无眼动数据，跳过")
                continue

            # 句子索引和长度（逻辑不变）
            sid = i - 1 if article == 1 else cums[article - 2] + i - 1
            sent_len = words_list[sid]
            print(f"  句子{snt_id}：sid={sid}，单词数={sent_len}")

            # 初始化矩阵（逻辑不变）
            sac_num = np.zeros((sent_len, sent_len))
            sac_dur = np.zeros((sent_len, sent_len))
            tril_idx = np.tril_indices(sent_len, k=-1)

            # 重复注视处理（修复后的逻辑完全不变）
            if len(event) >= 2:
                event['duplicate'] = event.CURRENT_FIX_INTEREST_AREA_ID.eq(
                    event.CURRENT_FIX_INTEREST_AREA_ID.shift()
                )
                dup_idx = event[event['duplicate']].index

                # 合并重复行时长（逻辑不变）
                for ind in dup_idx:
                    if ind > 0:
                        event.at[ind - 1, 'duration'] += event.at[ind, 'duration']

                # 删除重复行（逻辑不变）
                event = event.drop(dup_idx).reset_index(drop=True)

                # 填充眼动矩阵（逻辑不变）
                for ind in range(len(event) - 1):
                    row = int(event.iloc[ind].CURRENT_FIX_INTEREST_AREA_ID) - 1
                    col = int(event.iloc[ind + 1].CURRENT_FIX_INTEREST_AREA_ID) - 1
                    if 0 <= row < sent_len and 0 <= col < sent_len:
                        sac_num[row, col] += 1
                        sac_dur[row, col] += event.iloc[ind].duration + event.iloc[ind + 1].duration

            # 分训练/测试收集向量（逻辑不变）
            if sid < split_idx:
                y_num_train.extend(sac_num[tril_idx])
                y_dur_train.extend(sac_dur[tril_idx])
            else:
                y_num_test.extend(sac_num[tril_idx])
                y_dur_test.extend(sac_dur[tril_idx])

    # 标准化处理（逻辑不变）
    y_num_full = np.array(y_num_train + y_num_test)
    y_num_full = np.nan_to_num(zscore(y_num_full, nan_policy='omit'))
    y_dur_full = np.array(y_dur_train + y_dur_test)
    y_dur_full = np.nan_to_num(zscore(y_dur_full, nan_policy='omit'))

    # 输出日志（逻辑不变）
    end_time = time.time()
    print(f"\n✅ {subj}向量提取完成！")
    print(f"  次数向量长度：{len(y_num_full)}（预期：~7388）")
    print(f"  时长向量长度：{len(y_dur_full)}（预期：~7388）")
    print(f"  处理耗时：{end_time - start_time:.2f}秒")

    return y_num_full, y_dur_full


# ===================== 3. 主流程（新增循环，处理逻辑不变） =====================
if __name__ == "__main__":
    # 先加载全局的words和words_list（仅加载一次，提升效率）
    try:
        words = pd.read_csv(WORDS_CSV_PATH)
        print(f"✅ 成功加载words.csv：共{len(words)}行句子索引")
        words_list = pickle.load(open(WORDS_LIST_PATH, 'rb'))
        print(f"✅ 成功加载words_list.p：共{len(words_list)}个句子长度（预期：148）")
        if len(words_list) != 148:
            print(f"⚠️ 警告：words_list长度={len(words_list)}，官方预期148，请检查！")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"❌ 未找到文件：{str(e)}，请检查路径！")
    except Exception as e:
        raise ValueError(f"❌ 加载文件失败：{str(e)}")

    # 创建输出目录（避免路径不存在）
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

    # 循环处理所有目标被试
    total_subj = len(TARGET_SUBJ_IDS)
    processed_subj = 0
    failed_subj = []

    for subj_id in TARGET_SUBJ_IDS:
        # 提取当前被试的向量
        num_vec, dur_vec = extract_subj_saccade_vectors(subj_id, words, words_list)
        
        if num_vec is None or dur_vec is None:
            failed_subj.append(subj_id)
            continue
        
        # 动态生成输出路径
        subj = 'sub-0%s' % subj_id if subj_id < 10 else 'sub-%s' % subj_id
        num_vec_path = os.path.join(OUTPUT_ROOT_DIR, f"{subj}_saccade_num_vector.npy")
        dur_vec_path = os.path.join(OUTPUT_ROOT_DIR, f"{subj}_saccade_dur_vector.npy")
        
        # 保存向量
        np.save(num_vec_path, num_vec)
        np.save(dur_vec_path, dur_vec)
        print(f"✅ 向量已保存：")
        print(f"  次数向量：{num_vec_path}")
        print(f"  时长向量：{dur_vec_path}")
        
        processed_subj += 1

    # 输出汇总日志
    print(f"\n{'='*60} 批量处理完成 {'='*60}")
    print(f"总目标被试数：{total_subj}")
    print(f"成功处理被试数：{processed_subj}")
    print(f"失败/跳过被试数：{len(failed_subj)}")
    if failed_subj:
        print(f"失败被试ID：{failed_subj}")