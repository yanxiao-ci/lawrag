import argparse

# 1. 创建解析器
parser = argparse.ArgumentParser(description='简要描述这个脚本的功能')

# 2. 添加参数
parser.add_argument('input', help='输入文件路径')  # 位置参数
parser.add_argument('--output', help='输出文件路径')  # 可选参数
parser.add_argument('--verbose', action='store_true', help='是否输出详细信息')

# 3. 解析参数
args = parser.parse_args()

# 4. 使用参数
print(f"输入文件: {args.input}")
if args.output:
    print(f"输出文件: {args.output}")
if args.verbose:
    print("详细模式已开启")

# ======= 在 cmd 中运行 ======
# 1. 必须提供位置参数 input
# python demo01_argparse.py input.txt
#
# # 2. 提供所有参数
# python demo01_argparse.py input.txt --output output.txt --verbose
#
# # 3. 可选参数可以任意组合
# python demo01_argparse.py input.txt --verbose
# python demo01_argparse.py input.txt --output result.txt