import os
from modelscope.hub.snapshot_download import snapshot_download

# ========== 核心配置（LLaDA-8B-Base + GSAI-ML仓库 + 专属路径） ==========
# 根保存路径：统一放到LLaDA目录下，方便后续注意力提取
ROOT_SAVE_DIR = "/data/home/jinyuehan/LLMBrain/LLMmodel/Llada"
# GSAI-ML旗下LLaDA-8B-Base的标准仓库ID（ModelScope验证通过）
LLADA_MODELS = {
    "8B": "GSAI-ML/LLaDA-8B-Base",    # 对应https://www.modelscope.cn/models/GSAI-ML/LLaDA-8B-Base
}

# ========== 循环下载（与Llama系列逻辑完全一致，仅适配LLaDA） ==========
for model_size, model_repo in LLADA_MODELS.items():
    # 创建LLaDA-8B-Base专属目录
    model_save_dir = os.path.join(ROOT_SAVE_DIR, model_size)
    os.makedirs(model_save_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"开始下载 LLaDA-{model_size}")
    print(f"仓库ID：{model_repo}")
    print(f"保存路径：{model_save_dir}")
    print(f"{'='*50}")
    
    try:
        # 极简下载（核心参数适配ModelScope所有模型，包含LLaDA）
        model_dir = snapshot_download(
            model_id=model_repo,    # GSAI-ML下的LLaDA-8B-Base仓库ID
            cache_dir=model_save_dir, # 自定义保存路径（避免默认缓存分散）
            revision="master"       # 下载主分支（LLaDA-8B-Base默认分支）
        )
        
        # 验证下载结果（关键文件检查）
        config_path = os.path.join(model_dir, "config.json")
        weight_path = os.path.join(model_dir, "pytorch_model.bin") if os.path.exists(os.path.join(model_dir, "pytorch_model.bin")) else os.path.join(model_dir, "model.safetensors")
        print(f"✅ LLaDA-{model_size} 下载完成！")
        print(f"📌 模型实际路径：{model_dir}")
        print(f"📂 验证config.json：{config_path}（存在：{os.path.exists(config_path)}）")
        print(f"📂 验证权重文件：{weight_path}（存在：{os.path.exists(weight_path)}）")
        
    except Exception as e:
        print(f"❌ LLaDA-{model_size} 下载失败！错误：{str(e)}")
        print(f"💡 针对性排查建议：")
        # 区分LLaDA下载的常见错误类型
        if "Failed to resolve" in str(e):
            print(f"  1. 网络问题：服务器无法解析modelscope域名，检查DNS/代理配置")
            print(f"  2. 连通性测试：执行 curl https://www.modelscope.cn 验证网络")
        elif "404" in str(e):
            print(f"  1. 仓库ID错误：访问https://www.modelscope.cn/models/{model_repo}核对命名（确认是GSAI-ML/LLaDA-8B-Base）")
            print(f"  2. 版本确认：LLaDA-8B-Base仅master分支，无其他版本号")
        elif "Permission denied" in str(e):
            print(f"  1. 权限问题：执行 modelscope login 登录ModelScope账号（需官网生成访问令牌）")
            print(f"  2. 路径权限：检查{model_save_dir}是否有读写权限（执行 chmod 755 {model_save_dir}）")
        else:
            print(f"  1. 磁盘空间：检查{ROOT_SAVE_DIR}所在磁盘是否有足够空间（LLaDA-8B约16GB）")
            print(f"  2. 依赖版本：执行 pip install modelscope==1.11.0 升级依赖（适配LLaDA下载）")
            
        continue

# 最终汇总
print(f"\n{'='*50}")
print(f"LLaDA-8B-Base下载流程结束！")
print(f"根路径：{ROOT_SAVE_DIR}")
for size in LLADA_MODELS.keys():
    print(f"  - LLaDA-{size} 目标路径：{os.path.join(ROOT_SAVE_DIR, size)}")
print(f"{'='*50}")