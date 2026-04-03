import os
import re

# 获取当前脚本所在的绝对目录
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'models', 'TimeLLM.py')

print(f"正在诊断并修复文件: {file_path} ...")

if not os.path.exists(file_path):
    print("错误：找不到 models/TimeLLM.py 文件！")
else:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    fixed_count = 0
    gpt2_block_found = False

    for i, line in enumerate(lines):
        original_line = line
        
        # 1. 强力修复判断条件
        # 只要行里包含 elif, configs.llm_model 和 ==，不管后面写什么，都强制改成 GPT2
        # 同时保留原有的缩进 (indent)
        match = re.match(r"^(\s*)elif\s+configs\.llm_model\s*==.*:", line)
        if match:
            indent = match.group(1)
            # 强制替换为标准格式
            new_line = f"{indent}elif configs.llm_model == 'GPT2':\n"
            if new_line != line:
                print(f"[Line {i+1}] 发现逻辑行: {line.strip()}")
                print(f"          -> 强制修复为: {new_line.strip()}")
                line = new_line
                fixed_count += 1
            gpt2_block_found = True

        # 2. 强力修复模型路径
        # 只要包含 from_pretrained，就把里面的模型名强制改成 ./gpt2
        if "from_pretrained" in line:
            # 匹配括号里的第一个字符串参数
            # 替换 'gpt2', "gpt2", 'openai-community/gpt2' 等
            for bad_path in ["'gpt2'", '"gpt2"', "'openai-community/gpt2'", '"openai-community/gpt2"']:
                if bad_path in line:
                    new_line = line.replace(bad_path, "'./gpt2'")
                    if new_line != line:
                        print(f"[Line {i+1}] 发现远程路径: {line.strip()}")
                        print(f"          -> 强制修复为: {new_line.strip()}")
                        line = new_line
                        fixed_count += 1
                        break # 一行修一次即可

        new_lines.append(line)

    if fixed_count > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print("-" * 40)
        print(f"诊断完成！共执行了 {fixed_count} 处强制修复。")
        print("请再次尝试运行训练脚本。")
    elif gpt2_block_found:
        print("-" * 40)
        print("代码逻辑看起来是正确的，未发现需要修改的地方。")
        print("如果依然报错，可能是 run_main.py 传参有问题，请检查 Run_EV_Multi.bat 中的 --llm_model 参数。")
    else:
        print("-" * 40)
        print("警告：未找到 'elif configs.llm_model == ...' 代码块！")
        print("您的 TimeLLM.py 文件结构可能被破坏（例如被意外删除了判断语句）。")