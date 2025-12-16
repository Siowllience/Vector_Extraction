import os
import sys
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore
import warnings
from scipy.stats import ConstantInputWarning
from joblib import Parallel, delayed  # 恢复并行处理（原代码核心逻辑）

# 忽略无关警告
warnings.filterwarnings("ignore", category=ConstantInputWarning)
pd.options.mode.chained_assignment = None

# ===================== 配置参数 =====================
# 基础路径配置
EVENTS_BASE_DIR = '/data/home/jinyuehan/LLMBrain/11.25/reading_brain_datalad/ds003974'
WORDS_BASE_DIR = '/data/home/jinyuehan/LLMBrain/12.03'
GII_BASE_DIR = '/data/home/jinyuehan/LLMBrain/12.11/fmriprep'
SAVE_DIR = '/data/home/jinyuehan/LLMBrain/12.11/fmri_vector'

# 默认参数
DEFAULT_SUBJ = 'sub-01'    # 指定受试者
DEFAULT_GROUP = 'adult'    # 数据集分组（根据实际修改）
DEFAULT_HEM = 'L'          # 半球（L/R，若文件是lh/rh则改为lh/rh）

# ===================== 命令行参数解析 =====================
def parse_args():
    args = sys.argv[1:]
    subj = DEFAULT_SUBJ
    group = DEFAULT_GROUP
    hem = DEFAULT_HEM
    if len(args) >= 1:
        subj = args[0]       # 第一个参数：受试者（如sub-01）
    if len(args) >= 2:
        group = args[1]     # 第二个参数：分组（如adult）
    if len(args) >= 3:
        hem = args[2]       # 第三个参数：半球（L/R/lh/rh）
    return subj, group, hem

# ===================== 核心函数：单顶点BOLD向量处理（完全还原原逻辑）=====================
def process_vertex(v, X_train, X_test, events, surf, words, cums, words_list):
    """为单个脑区顶点v生成对应的BOLD训练/测试向量（原代码逻辑完全保留）"""
    y_train, y_test = [], []
    for article in range(1, 6):
        # 获取当前article的句子数（原逻辑）
        n_snts = int(words[words.SentenceID.str.match(f't.0{article}')].SentenceID.iloc[-1].split('.')[-1])
        for i in range(1, n_snts + 1):
            # 构造句子ID（t.01.01 格式）
            snt_id = f't.0{article}.0{i}' if i < 10 else f't.0{article}.{i}'
            event = events[events.SentenceID.str.match(snt_id)]
            # 计算句子索引sid（原逻辑）
            sid = i - 1 if article == 1 else cums[article - 2] + i - 1   
            # 初始化句子级BOLD矩阵（原逻辑）
            fmri_snt = np.zeros((words_list[sid], words_list[sid]))
            tril_idx = np.tril_indices(words_list[sid], k=-1)
            event = event.reset_index(drop=True) 
            
            # 填充BOLD值（核心修正：恢复原代码的v顶点索引）
            for ind, e in event.iterrows():
                if ind < len(event)-1:
                    row = int(e.CURRENT_FIX_INTEREST_AREA_ID)-1
                    col = int(event.iloc[ind+1].CURRENT_FIX_INTEREST_AREA_ID)-1
                    # 计算scan索引（原公式）
                    scan = int(np.ceil((event.iloc[ind+1].onset/1000 + 5)/0.4))
                    # 防止scan越界（原逻辑）
                    while scan >= surf[article-1].shape[1]:
                        scan -= 1
                    # ✅ 修正：提取第v个顶点在scan时刻的BOLD值（完全匹配原代码）
                    fmri_snt[row, col] = surf[article-1][v, scan]
            
            # 拆分训练/测试集（原逻辑：sid<133为训练）
            if sid < 133:
                y_train.extend(fmri_snt[tril_idx[0], tril_idx[1]])
            else:
                y_test.extend(fmri_snt[tril_idx[0], tril_idx[1]])
    
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return y_train, y_test

# ===================== 主处理函数（完全匹配原逻辑）=====================
def process_bold_vector(subj, group, hem):
    # 1. 解析受试者ID
    subj_id = int(subj.split('-')[1])
    
    # 2. 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 3. 加载原代码依赖的基础文件（原路径适配）
    words = pd.read_csv(os.path.join(WORDS_BASE_DIR, 'words.csv'))
    words_list = pickle.load(open(os.path.join(WORDS_BASE_DIR, 'words_list.p'), 'rb'))
    snts_list = [31, 31, 28, 28, 30]
    cums = np.cumsum(snts_list)
    
    # 4. 加载events和BOLD数据（原逻辑完全保留）
    surf = []
    events_lst = []
    for article in range(1, 6):
        run = 1
        # 加载events.tsv
        events_path = os.path.join(EVENTS_BASE_DIR, group, subj, 'func', 
                                  f'{subj}_task-read_run-{run}_events.tsv')
        events = pd.read_csv(events_path, delimiter='\t')
        events = events.dropna().reset_index(drop=True)
        
        # 匹配当前article对应的run（原逻辑）
        run_article = int(events.SentenceID[0].split('.')[1])
        while run_article != article and run < 5:
            run += 1
            events_path = os.path.join(EVENTS_BASE_DIR, group, subj, 'func', 
                                      f'{subj}_task-read_run-{run}_events.tsv')
            events = pd.read_csv(events_path, delimiter='\t')
            events = events.dropna().reset_index(drop=True)
            run_article = int(events.SentenceID[0].split('.')[1])
        
        events['article'] = [article] * len(events)
        events_lst.append(events)
        
        # 加载fsaverage5空间的BOLD文件（.func.gii）
        gii_path = os.path.join(GII_BASE_DIR, subj, 'func', 
                               f'{subj}_task-read_run-{run}_hemi-{hem}_space-fsaverage5_bold.func.gii')
        fmri_gii = nib.load(gii_path)
        
        # 提取BOLD数据并标准化（原逻辑）
        fmri_data = np.column_stack([arr.data for arr in fmri_gii.darrays])
        fmri_data = np.nan_to_num(zscore(fmri_data, axis=0, nan_policy='omit'))
        surf.append(fmri_data)
    
    # 合并所有events
    events = pd.concat(events_lst).reset_index(drop=True)
    
    # 5. 并行处理每个脑区顶点v（0-10241，原代码逻辑）
    print(f"开始处理{subj}半球{hem}的10242个脑区顶点...")
    # 空数组占位（X_train/X_test维度不影响BOLD提取，仅需传递形状匹配的空数组，原代码X是模型特征，此处仅处理BOLD）
    X_train_placeholder = np.zeros((1, 1))  # 仅为适配process_vertex参数，不影响BOLD提取
    X_test_placeholder = np.zeros((1, 1))
    
    # 并行循环每个顶点v（原代码n_jobs=-1）
    results = Parallel(n_jobs=-1)(
        delayed(process_vertex)(v, X_train_placeholder, X_test_placeholder, events, surf, words, cums, words_list) 
        for v in range(10242)
    )
    
    # 6. 整理所有顶点的BOLD向量
    # results是列表，每个元素是(v的y_train, v的y_test)，需拆分并堆叠
    y_train_all = np.array([res[0] for res in results])  # 形状：(10242, N_train)
    y_test_all = np.array([res[1] for res in results])   # 形状：(10242, N_test)
    
    # 7. 保存BOLD向量（按顶点维度）
    save_prefix = f'{subj}_hemi-{hem}'
    np.save(os.path.join(SAVE_DIR, f'{save_prefix}_y_train_all_vertices.npy'), y_train_all)
    np.save(os.path.join(SAVE_DIR, f'{save_prefix}_y_test_all_vertices.npy'), y_test_all)
    
    # 8. 输出维度信息（核心：体现顶点维度）
    print("="*60)
    print(f"受试者：{subj} | 半球：{hem}")
    print(f"所有顶点的训练集BOLD向量维度：y_train_all.shape = {y_train_all.shape}")
    print(f"所有顶点的测试集BOLD向量维度：y_test_all.shape = {y_test_all.shape}")
    print(f"  - 第一维：10242 = fsaverage5半球顶点数")
    print(f"  - 第二维：N_train/N_test = 每个顶点对应的BOLD样本数")
    print(f"保存路径：{SAVE_DIR}")
    print(f"保存文件：")
    print(f"  - {save_prefix}_y_train_all_vertices.npy")
    print(f"  - {save_prefix}_y_test_all_vertices.npy")
    print("="*60)
    
    return y_train_all, y_test_all

# ===================== 执行入口 =====================
if __name__ == '__main__':
    # 解析参数：python script.py sub-01 adult L
    subj, group, hem = parse_args()
    # 执行BOLD向量处理
    process_bold_vector(subj, group, hem)