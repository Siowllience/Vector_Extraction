import os
from modelscope.hub.snapshot_download import snapshot_download

# ========== 核心配置（Llama3.1 + LLM-Research仓库 + 专属路径） ==========
ROOT_SAVE_DIR = "/data/home/jinyuehan/LLMBrain/LLMmodel/Llama/llama3"
# LLM-Research旗下Llama3.1的标准仓库ID（ModelScope官方命名）
LLAMA31_MODELS = {
    "8B": "LLM-Research/Meta-Llama-3-8B",    # 对应https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3-8B
    "70B": "LLM-Research/Meta-Llama-3-70B"   # 对应https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3-70B
}

# ========== 循环下载（极简风格，与之前Llama系列下载逻辑一致） ==========
for model_size, model_repo in LLAMA31_MODELS.items():
    # 创建Llama3各版本专属目录
    model_save_dir = os.path.join(ROOT_SAVE_DIR, model_size)
    os.makedirs(model_save_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"开始下载 Llama3-{model_size}")
    print(f"仓库ID：{model_repo}")
    print(f"保存路径：{model_save_dir}")
    print(f"{'='*50}")
    
    try:
        # 极简下载（仅核心参数，适配所有ModelScope版本）
        model_dir = snapshot_download(
            model_id=model_repo,    # LLM-Research下的Llama3.1仓库ID
            cache_dir=model_save_dir # 自定义保存路径
        )
        
        # 验证下载结果
        config_path = os.path.join(model_dir, "config.json")
        print(f"✅ Llama3-{model_size} 下载完成！")
        print(f"📌 模型实际路径：{model_dir}")
        print(f"📂 验证文件：{config_path}（存在：{os.path.exists(config_path)}）")
        
    except Exception as e:
        print(f"❌ Llama3-{model_size} 下载失败！错误：{str(e)}")
        print(f"💡 针对性排查建议：")
        # 区分不同错误类型给出建议
        if "Failed to resolve" in str(e):
            print(f"  1. 网络问题：服务器无法解析modelscope域名，检查DNS/代理配置")
            print(f"  2. 连通性测试：执行 curl https://www.modelscope.cn 验证网络")
        elif "404" in str(e):
            print(f"  1. 仓库ID错误：访问https://www.modelscope.cn/models/{model_repo}核对命名")
            print(f"  2. 版本确认：Llama3仅8B/70B等版本，无其他非标准尺寸")
        else:
            print(f"  1. 权限问题：执行 modelscope login 登录有权限的ModelScope账号")
            print(f"  2. 空间检查：确认{ROOT_SAVE_DIR}路径有足够磁盘空间（70B版本需数百GB）")
        continue

# 最终汇总
print(f"\n{'='*50}")
print(f"所有Llama3版本下载流程结束！")
print(f"根路径：{ROOT_SAVE_DIR}")
for size in LLAMA31_MODELS.keys():
    print(f"  - Llama3-{size} 目标路径：{os.path.join(ROOT_SAVE_DIR, size)}")
print(f"{'='*50}")