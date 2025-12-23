import os
from modelscope.hub.snapshot_download import snapshot_download

# ========== 核心配置（关键：替换为skyline2006的仓库ID） ==========
ROOT_SAVE_DIR = "/data/home/jinyuehan/LLMBrain/LLMmodel/Llama/llama1"
# 仅保留7B/30B/65B，仓库ID严格匹配ModelScope上skyline2006的路径
LLAMA1_MODELS = {
    "7B": "skyline2006/llama-7b",    # 对应https://www.modelscope.cn/models/skyline2006/llama-7b
    "13B": "skyline2006/llama-13b",   # 需确认该仓库存在，若不存在请在ModelScope核对命名
    "30B": "skyline2006/llama-30b",  # 需确认该仓库存在，若不存在请在ModelScope核对命名
    "65B": "skyline2006/llama-65b"   # 需确认该仓库存在，若不存在请在ModelScope核对命名
}

# ========== 循环下载（极简风格，适配ModelScope） ==========
for model_size, model_repo in LLAMA1_MODELS.items():
    # 创建版本专属目录
    model_save_dir = os.path.join(ROOT_SAVE_DIR, model_size)
    os.makedirs(model_save_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"开始下载 Llama1-{model_size}")
    print(f"仓库ID：{model_repo}")
    print(f"保存路径：{model_save_dir}")
    print(f"{'='*50}")
    
    try:
        # 极简下载（仅核心参数，与Llama3下载逻辑一致）
        model_dir = snapshot_download(
            model_id=model_repo,    # skyline2006下的仓库ID
            cache_dir=model_save_dir # 自定义保存路径
        )
        
        # 验证下载结果
        config_path = os.path.join(model_dir, "config.json")
        print(f"✅ Llama1-{model_size} 下载完成！")
        print(f"📌 模型实际路径：{model_dir}")
        print(f"📂 验证文件：{config_path}（存在：{os.path.exists(config_path)}）")
        
    except Exception as e:
        print(f"❌ Llama1-{model_size} 下载失败！错误：{str(e)}")
        print(f"💡 针对性排查建议：")
        # 区分域名解析错误和仓库不存在错误，给出不同建议
        if "Failed to resolve" in str(e):
            print(f"  1. 网络问题：服务器无法解析modelscope.ai域名，需检查DNS配置/网络代理")
            print(f"  2. 临时方案：手动访问https://www.modelscope.cn/models/{model_repo}确认仓库是否存在")
            print(f"  3. 网络测试：执行 curl https://www.modelscope.cn 验证连通性")
        else:
            print(f"  1. 仓库ID错误：访问https://www.modelscope.cn/models/{model_repo}确认仓库是否存在")
            print(f"  2. 权限问题：确认账号已登录ModelScope且有权限下载该模型")
            print(f"  3. 命名问题：核对skyline2006下的模型名（如llama-30b是否为llama-30B等）")
        continue

# 最终汇总
print(f"\n{'='*50}")
print(f"所有Llama1版本下载流程结束！")
print(f"根路径：{ROOT_SAVE_DIR}")
for size in LLAMA1_MODELS.keys():
    print(f"  - Llama1-{size} 目标路径：{os.path.join(ROOT_SAVE_DIR, size)}")
print(f"{'='*50}")