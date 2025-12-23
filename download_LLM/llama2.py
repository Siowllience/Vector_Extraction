import os
from modelscope.hub.snapshot_download import snapshot_download

# ========== 核心配置（Llama2 + LLM-Research仓库 + 指定路径） ==========
ROOT_SAVE_DIR = "/data/home/jinyuehan/LLMBrain/LLMmodel/Llama/llama2"
# LLM-Research旗下Llama2的标准仓库ID（ModelScope官方命名）
LLAMA2_MODELS = {
    # "7B": "LLM-Research/llama-2-7b",    # 对应https://www.modelscope.cn/models/LLM-Research/Meta-Llama-2-7B
    # "13B": "LLM-Research/llama-2-13b",  # 对应https://www.modelscope.cn/models/LLM-Research/Meta-Llama-2-13B
    "70B": "AI-ModelScope/Llama-2-70b-hf"   # 对应https://www.modelscope.cn/models/LLM-Research/Meta-Llama-2-70B
}

# ========== 循环下载（极简风格，适配ModelScope） ==========
for model_size, model_repo in LLAMA2_MODELS.items():
    # 创建Llama2各版本专属目录
    model_save_dir = os.path.join(ROOT_SAVE_DIR, model_size)
    os.makedirs(model_save_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"开始下载 Llama2-{model_size}")
    print(f"仓库ID：{model_repo}")
    print(f"保存路径：{model_save_dir}")
    print(f"{'='*50}")
    
    try:
        # 极简下载（仅核心参数，与Llama3/1下载逻辑一致）
        model_dir = snapshot_download(
            model_id=model_repo,    # LLM-Research下的Llama2仓库ID
            cache_dir=model_save_dir # 自定义保存路径
        )
        
        # 验证下载结果
        config_path = os.path.join(model_dir, "config.json")
        print(f"✅ Llama2-{model_size} 下载完成！")
        print(f"📌 模型实际路径：{model_dir}")
        print(f"📂 验证文件：{config_path}（存在：{os.path.exists(config_path)}）")
        
    except Exception as e:
        print(f"❌ Llama2-{model_size} 下载失败！错误：{str(e)}")
        print(f"💡 针对性排查建议：")
        # 区分域名解析错误和仓库不存在错误
        if "Failed to resolve" in str(e):
            print(f"  1. 网络问题：服务器无法解析modelscope域名，检查DNS/代理配置")
            print(f"  2. 连通性测试：执行 curl https://www.modelscope.cn 验证网络")
            print(f"  3. 手动验证：访问https://www.modelscope.cn/models/{model_repo}确认仓库存在")
        else:
            print(f"  1. 仓库ID错误：访问https://www.modelscope.cn/models/{model_repo}核对命名")
            print(f"  2. 权限问题：执行 modelscope login 登录有权限的账号")
            print(f"  3. 版本核对：Llama2仅7B/13B/70B，无30B/65B版本")
        continue

# 最终汇总
print(f"\n{'='*50}")
print(f"所有Llama2版本下载流程结束！")
print(f"根路径：{ROOT_SAVE_DIR}")
for size in LLAMA2_MODELS.keys():
    print(f"  - Llama2-{size} 目标路径：{os.path.join(ROOT_SAVE_DIR, size)}")
print(f"{'='*50}")