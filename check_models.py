# check_modules.py
import sys

sys.path.append('.')

# 检查models/common.py中定义了哪些模块
with open('models/common.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 查找class定义
import re

class_definitions = re.findall(r'class (\w+)\(nn\.Module\):', content)
print("在common.py中找到的模块类:")
for i, class_name in enumerate(class_definitions):
    print(f"  {i + 1}. {class_name}")

# 检查特定模块是否存在
modules_to_check = ['SPPF', 'SPP', 'Focus', 'C3', 'Conv', 'Bottleneck', 'BottleneckCSP']
print("\n检查特定模块:")
for module in modules_to_check:
    if module in class_definitions:
        print(f"  ✓ {module} 存在")
    else:
        print(f"  ✗ {module} 不存在")