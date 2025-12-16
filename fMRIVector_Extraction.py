import os
import sys
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore
import warnings
from scipy.stats import ConstantInputWarning
from joblib import Parallel, delayed  # 恢复顶点并行循环

# 忽略无关警告
warnings.filterwarnings("ignore", category=ConstantInputWarning)
pd.options.mode.chained_assignment = None

# ===================== 配置参数 =====================
# 基础路径配置（移除group层级）
EVENTS_BASE_DIR = '/data/home/jinyuehan/LLMBrain/11.25/reading_brain_datalad/ds003974'  #原始数据集路径。
WORDS_BASE_DIR = ''                                                                     #words.csv路径。
GII_BASE_DIR = 'fmriprep'                                                               #处理好的fMRI数据集路径
SAVE_DIR = 'fmriVector'                                                                 #输出路径

# 默认参数（保留group但无实际作用）
DEFAULT_SUBJ = 'sub-01'    #受试者编号
DEFAULT_GROUP = 'adult'    #路径中不再使用，可忽略
DEFAULT_HEM = 'L'          #L代表左半脑，R代表右半脑

# ===================== 命令行参数解析 =====================
def parse_args():
    args = sys.argv[1:]
    subj = DEFAULT_SUBJ
    group = DEFAULT_GROUP
    hem = DEFAULT_HEM
    if len(args) >= 1:
        subj = args[0]       
    if len(args) >= 2:
        group = args[1]     
    if len(args) >= 3:
        hem = args[2]       
    return subj, group, hem

# ===================== 单顶点BOLD处理（合并train/test为单个向量）=====================
def process_vertex(v, events, surf, words, cums, words_list):
    y_all = []  # 合并所有BOLD值，不再分train/test
    for article in range(1, 6):
        n_snts = int(words[words.SentenceID.str.match(f't.0{article}')].SentenceID.iloc[-1].split('.')[-1])
        for i in range(1, n_snts + 1):
            snt_id = f't.0{article}.0{i}' if i < 10 else f't.0{article}.{i}'
            event = events[events.SentenceID.str.match(snt_id)]
            sid = i - 1 if article == 1 else cums[article - 2] + i - 1   
            fmri_snt = np.zeros((words_list[sid], words_list[sid]))
            tril_idx = np.tril_indices(words_list[sid], k=-1)
            event = event.reset_index(drop=True) 
            
            # 核心：提取对应顶点v的BOLD值（原逻辑）
            for ind, e in event.iterrows():
                if ind < len(event)-1:
                    row = int(e.CURRENT_FIX_INTEREST_AREA_ID)-1
                    col = int(event.iloc[ind+1].CURRENT_FIX_INTEREST_AREA_ID)-1
                    scan = int(np.ceil((event.iloc[ind+1].onset/1000 + 5)/0.4))
                    while scan >= surf[article-1].shape[1]:
                        scan -= 1
                    fmri_snt[row, col] = surf[article-1][v, scan]  # 修正索引错误
            
            # ✅ 关键修改：不再拆分train/test，直接合并到y_all
            y_all.extend(fmri_snt[tril_idx[0], tril_idx[1]])
    
    y_all = np.array(y_all)
    return y_all  # 仅返回合并后的向量

# ===================== 主处理函数 =====================
def process_bold_vector(subj, group, hem):
    # 1. 解析受试者ID
    subj_id = int(subj.split('-')[1])
    
    # 2. 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 3. 加载基础文件
    words = pd.read_csv(os.path.join(WORDS_BASE_DIR, 'words.csv'))
    words_list = pickle.load(open(os.path.join(WORDS_BASE_DIR, 'words_list.p'), 'rb'))
    snts_list = [31, 31, 28, 28, 30]
    cums = np.cumsum(snts_list)
    
    # 4. 加载events和BOLD数据（核心：移除路径中的group层级）
    surf = []
    events_lst = []
    for article in range(1, 6):
        run = 1
        # ✅ 关键修改：去掉group，路径变为 EVENTS_BASE_DIR/subj/func/...
        events_path = os.path.join(EVENTS_BASE_DIR, subj, 'func', 
                                  f'{subj}_task-read_run-{run}_events.tsv')
        events = pd.read_csv(events_path, delimiter='\t')
        events = events.dropna().reset_index(drop=True)
        
        # 匹配当前article对应的run
        run_article = int(events.SentenceID[0].split('.')[1])
        while run_article != article and run < 5:
            run += 1
            # ✅ 同样去掉group层级
            events_path = os.path.join(EVENTS_BASE_DIR, subj, 'func', 
                                      f'{subj}_task-read_run-{run}_events.tsv')
            events = pd.read_csv(events_path, delimiter='\t')
            events = events.dropna().reset_index(drop=True)
            run_article = int(events.SentenceID[0].split('.')[1])
        
        events['article'] = [article] * len(events)
        events_lst.append(events)
        
        # 加载BOLD文件（路径不变）
        gii_path = os.path.join(GII_BASE_DIR, subj, 'func', 
                               f'{subj}_task-read_run-{run}_hemi-{hem}_space-fsaverage5_bold.func.gii')
        fmri_gii = nib.load(gii_path)
        fmri_data = np.column_stack([arr.data for arr in fmri_gii.darrays])
        fmri_data = np.nan_to_num(zscore(fmri_data, axis=0, nan_policy='omit'))
        surf.append(fmri_data)
    
    # 合并所有events
    events = pd.concat(events_lst).reset_index(drop=True)
    
    # 5. 并行处理10242个顶点
    print(f"开始处理{subj}半球{hem}的10242个脑区顶点...")
    results = Parallel(n_jobs=-1)(
        delayed(process_vertex)(v, events, surf, words, cums, words_list) 
        for v in range(10242)
    )
    
    # 6. 整理结果：堆叠所有顶点的合并后BOLD向量
    y_all_all = np.array(results)  # 形状：(10242, N_total)，N_total是所有样本数
    
    # 7. 保存合并后的BOLD向量（仅一个文件）
    save_prefix = f'{subj}_hemi-{hem}'
    save_path = os.path.join(SAVE_DIR, f'{save_prefix}_y_all_vertices.npy')
    np.save(save_path, y_all_all)
    
    # 8. 输出维度信息
    print("="*60)
    print(f"受试者：{subj} | 半球：{hem}")
    print(f"所有顶点的合并BOLD向量维度：y_all_all.shape = {y_all_all.shape}")
    print(f"  - 第一维：10242 = fsaverage5半球顶点数")
    print(f"  - 第二维：{y_all_all.shape[1]} = 每个顶点的全量BOLD样本数（合并train+test）")
    print(f"保存路径：{SAVE_DIR}")
    print(f"保存文件：{save_prefix}_y_all_vertices.npy")
    print("="*60)
    
    return y_all_all

# ===================== 执行入口 =====================
if __name__ == '__main__':
    subj, group, hem = parse_args()
    process_bold_vector(subj, group, hem)