# complete_label_checker.py
"""
完整的YOLOv5标签文件检查脚本
会正确处理相对路径和绝对路径
"""

import os
import yaml
import sys
from pathlib import Path


def check_labels():
    """
    检查标签文件的完整脚本
    """
    # 数据配置文件路径
    data_yaml_path = 'data/rice_disease.yaml'

    if not os.path.exists(data_yaml_path):
        print(f"错误: 数据配置文件不存在 - {data_yaml_path}")
        return

    print("YOLOv5标签文件完整检查")
    print("=" * 60)

    # 读取数据配置
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)

    print(f"数据配置:")
    print(f"  path: {data_config.get('path', '未指定')}")
    print(f"  train: {data_config.get('train', '未指定')}")
    print(f"  val: {data_config.get('val', '未指定')}")
    print(f"  nc: {data_config.get('nc', '未指定')}")
    print(f"  names: {data_config.get('names', '未指定')}")
    print()

    # 构建完整路径
    base_path = data_config.get('path', '')
    train_relative = data_config.get('train', '')
    val_relative = data_config.get('val', '')

    if not base_path:
        print("警告: 配置文件中没有指定path，使用当前目录作为根目录")
        base_path = '.'

    # 构建完整路径
    train_images_dir = os.path.join(base_path, train_relative)
    val_images_dir = os.path.join(base_path, val_relative)

    # 标签目录通常是images替换为labels
    train_labels_dir = train_images_dir.replace('images', 'labels')
    val_labels_dir = val_images_dir.replace('images', 'labels')

    print(f"计算出的路径:")
    print(f"  训练集图像目录: {train_images_dir}")
    print(f"  训练集标签目录: {train_labels_dir}")
    print(f"  验证集图像目录: {val_images_dir}")
    print(f"  验证集标签目录: {val_labels_dir}")
    print()

    # 检查目录是否存在
    def check_directory(dir_path, dir_type):
        if os.path.exists(dir_path):
            print(f"✓ {dir_type} 目录存在: {dir_path}")

            # 统计文件
            if 'images' in dir_path:
                img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
                files = [f for f in os.listdir(dir_path)
                         if any(f.lower().endswith(ext) for ext in img_exts)]
                print(f"  找到 {len(files)} 个图像文件")
                if files:
                    print(f"  示例: {files[:5]}")
            else:
                txt_files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
                print(f"  找到 {len(txt_files)} 个标签文件")
                if txt_files:
                    print(f"  示例: {txt_files[:5]}")
            return True
        else:
            print(f"✗ {dir_type} 目录不存在: {dir_path}")
            return False

    # 检查所有目录
    print("检查目录存在性:")
    print("-" * 40)

    train_images_exist = check_directory(train_images_dir, "训练集图像")
    train_labels_exist = check_directory(train_labels_dir, "训练集标签")
    val_images_exist = check_directory(val_images_dir, "验证集图像")
    val_labels_exist = check_directory(val_labels_dir, "验证集标签")

    # 如果目录不存在，尝试自动创建
    print("\n" + "=" * 60)
    print("标签文件格式检查")
    print("=" * 60)

    # 检查标签文件格式
    def check_label_format(label_dir, dataset_type):
        if not os.path.exists(label_dir):
            print(f"✗ {dataset_type} 标签目录不存在: {label_dir}")
            return

        txt_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        if not txt_files:
            print(f"✗ {dataset_type} 标签目录中没有 .txt 文件: {label_dir}")
            return

        print(f"\n检查 {dataset_type} 标签文件格式 ({len(txt_files)} 个文件):")

        issues = []
        total_lines = 0
        valid_lines = 0

        for txt_file in txt_files[:50]:  # 只检查前50个文件以节省时间
            filepath = os.path.join(label_dir, txt_file)

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for i, line in enumerate(lines):
                    total_lines += 1
                    line = line.strip()

                    if not line:
                        continue  # 跳过空行

                    # 分割值
                    values = line.split()

                    # 检查值数量
                    if len(values) != 5:
                        issues.append(f"{txt_file} 第{i + 1}行: 有 {len(values)} 个值 (应为5个)")
                        continue

                    # 检查数据类型
                    try:
                        class_id = int(values[0])
                        x_center = float(values[1])
                        y_center = float(values[2])
                        width = float(values[3])
                        height = float(values[4])
                    except ValueError:
                        issues.append(f"{txt_file} 第{i + 1}行: 数据类型错误")
                        continue

                    # 检查类别ID
                    nc = data_config.get('nc', 0)
                    if not (0 <= class_id < nc):
                        issues.append(f"{txt_file} 第{i + 1}行: 类别ID {class_id} 超出范围 (应为0-{nc - 1})")
                        continue

                    # 检查坐标范围
                    coords = [x_center, y_center, width, height]
                    coord_names = ['x_center', 'y_center', 'width', 'height']

                    for coord, name in zip(coords, coord_names):
                        if not (0 <= coord <= 1):
                            issues.append(f"{txt_file} 第{i + 1}行: {name} {coord} 超出范围 (应为0-1)")
                            break
                    else:
                        valid_lines += 1

            except Exception as e:
                issues.append(f"{txt_file}: 读取错误 - {str(e)}")

        # 打印结果
        if issues:
            print(f"  发现 {len(issues)} 个问题:")
            for issue in issues[:10]:  # 只显示前10个问题
                print(f"    - {issue}")
            if len(issues) > 10:
                print(f"    ... 还有 {len(issues) - 10} 个问题未显示")
        else:
            print(f"  ✓ 所有检查的文件格式正确")

        if total_lines > 0:
            print(f"  有效行数: {valid_lines}/{total_lines} ({valid_lines / total_lines * 100:.1f}%)")

    # 检查标签格式
    check_label_format(train_labels_dir, "训练集")
    check_label_format(val_labels_dir, "验证集")

    # 检查图像和标签文件对应关系
    print("\n" + "=" * 60)
    print("检查图像和标签文件对应关系")
    print("=" * 60)

    def check_correspondence(images_dir, labels_dir, dataset_type):
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"✗ {dataset_type}: 图像或标签目录不存在")
            return

        # 获取图像文件列表（不含扩展名）
        img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files = []
        for f in os.listdir(images_dir):
            for ext in img_exts:
                if f.lower().endswith(ext):
                    image_files.append(os.path.splitext(f)[0])
                    break

        # 获取标签文件列表（不含扩展名）
        label_files = [os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith('.txt')]

        print(f"\n{dataset_type}:")
        print(f"  图像文件数: {len(image_files)}")
        print(f"  标签文件数: {len(label_files)}")

        # 检查对应关系
        image_set = set(image_files)
        label_set = set(label_files)

        missing_labels = image_set - label_set
        missing_images = label_set - image_set

        if missing_labels:
            print(f"  ✗ 有 {len(missing_labels)} 个图像没有对应的标签文件:")
            for img in list(missing_labels)[:5]:
                print(f"    - {img}")
            if len(missing_labels) > 5:
                print(f"    ... 还有 {len(missing_labels) - 5} 个")

        if missing_images:
            print(f"  ✗ 有 {len(missing_images)} 个标签文件没有对应的图像:")
            for lbl in list(missing_images)[:5]:
                print(f"    - {lbl}")
            if len(missing_images) > 5:
                print(f"    ... 还有 {len(missing_images) - 5} 个")

        if not missing_labels and not missing_images:
            print(f"  ✓ 所有图像和标签文件一一对应")

        # 计算匹配率
        matched = image_set & label_set
        if image_files:
            match_rate = len(matched) / len(image_files) * 100
            print(f"  匹配率: {match_rate:.1f}%")

    # 检查对应关系
    check_correspondence(train_images_dir, train_labels_dir, "训练集")
    check_correspondence(val_images_dir, val_labels_dir, "验证集")

    # 总结和建议
    print("\n" + "=" * 60)
    print("总结与建议")
    print("=" * 60)

    # 检查结果汇总
    all_good = True
    issues = []

    if not train_images_exist:
        issues.append("训练集图像目录不存在")
        all_good = False

    if not train_labels_exist:
        issues.append("训练集标签目录不存在")
        all_good = False

    if not val_images_exist:
        issues.append("验证集图像目录不存在")
        all_good = False

    if not val_labels_exist:
        issues.append("验证集标签目录不存在")
        all_good = False

    if all_good:
        print("✓ 所有必要的目录都存在")

        # 检查是否能直接运行训练
        print("\n您可以尝试以下命令进行训练:")
        print(f"python train.py \\")
        print(f"  --weights weights/yolov5s.pt \\")
        print(f"  --cfg models/rice_yolov5s.yaml \\")
        print(f"  --data data/rice_disease.yaml \\")
        print(f"  --epochs 1 \\")  # 先试1个epoch
        print(f"  --batch-size 1 \\")  # 减小批次大小
        print(f"  --device 0 \\")
        print(f"  --workers 0 \\")  # 不使用多进程
        print(f"  --hyp data/hyp.scratch-low.yaml")

        # 如果使用知识蒸馏
        print("\n如果要使用知识蒸馏:")
        print(f"python train.py \\")
        print(f"  --weights weights/yolov5s.pt \\")
        print(f"  --cfg models/rice_yolov5s.yaml \\")
        print(f"  --data data/rice_disease.yaml \\")
        print(f"  --epochs 1 \\")
        print(f"  --batch-size 1 \\")
        print(f"  --device 0 \\")
        print(f"  --workers 0 \\")
        print(f"  --hyp data/hyp.scratch-low.yaml \\")
        print(f"  --t_weights yolov5m.pt \\")
        print(f"  --distill")
    else:
        print("✗ 发现以下问题:")
        for issue in issues:
            print(f"  - {issue}")

        print("\n建议解决方案:")

        if not train_images_exist:
            print(f"  1. 创建训练集图像目录: {train_images_dir}")
            print(f"     或修改 data/rice_disease.yaml 中的 train 路径")

        if not train_labels_exist:
            print(f"  2. 创建训练集标签目录: {train_labels_dir}")
            print(f"     或确保标签文件在正确的位置")

        print(f"\n  3. 检查数据配置文件 {data_yaml_path}")
        print(f"     当前配置:")
        print(f"       path: {base_path}")
        print(f"       train: {train_relative}")
        print(f"       val: {val_relative}")


if __name__ == "__main__":
    check_labels()