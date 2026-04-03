import os

# 获取当前脚本所在的绝对目录
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'models', 'TimeLLM.py')

print(f"正在修复数据类型问题: {file_path} ...")

if not os.path.exists(file_path):
    print("错误：找不到 models/TimeLLM.py 文件！")
else:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    
    # 目标：找到 x_enc.to(torch.bfloat16) 并替换为 x_enc
    # 这样就移除了强制类型转换，使用系统默认精度(Float32)
    target_str = "x_enc.to(torch.bfloat16)"
    replacement_str = "x_enc"
    
    if target_str in content:
        content = content.replace(target_str, replacement_str)
        print(f"[修复] 发现并移除了强制类型转换: {target_str}")
    else:
        print("[提示] 未发现强制类型转换代码，或者已经被修复过。")

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("-" * 30)
        print("修复完成！现在模型将使用统一的 Float32 精度运行。")
    else:
        print("-" * 30)
        print("文件未发生变化。")