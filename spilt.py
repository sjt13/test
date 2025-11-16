import os
import shutil
import random
from tqdm import tqdm


def split_dataset(images_folder, labels_folder, output_folder,
                  train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    随机划分数据集，并按照指定目录结构保存

    参数:
    images_folder: 原始图片文件夹路径
    labels_folder: 原始标签文件夹路径
    output_folder: 输出文件夹路径（kunkun文件夹）
    train_ratio: 训练集比例（默认0.7）
    val_ratio: 验证集比例（默认0.2）
    test_ratio: 测试集比例（默认0.1）
    """

    # 检查比例总和是否为1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        print("错误：训练集、验证集和测试集的比例之和必须等于1")
        return

    # 检查原始文件夹是否存在
    if not os.path.exists(images_folder):
        print(f"错误：图片文件夹不存在 - {images_folder}")
        return

    if not os.path.exists(labels_folder):
        print(f"错误：标签文件夹不存在 - {labels_folder}")
        return

    # 创建输出文件夹结构
    folders_to_create = [
        os.path.join(output_folder, "images", "train"),
        os.path.join(output_folder, "images", "val"),
        os.path.join(output_folder, "images", "test"),
        os.path.join(output_folder, "labels", "train"),
        os.path.join(output_folder, "labels", "val"),
        os.path.join(output_folder, "labels", "test")
    ]

    for folder in folders_to_create:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"创建文件夹: {folder}")

    # 获取所有图片文件（只处理.jpg文件）
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]

    if not image_files:
        print("在图片文件夹中没有找到.jpg文件！")
        return

    print(f"找到 {len(image_files)} 个图片文件")

    # 检查对应的标签文件是否存在
    valid_pairs = []
    for image_file in image_files:
        # 获取图片文件名（不含扩展名）
        base_name = os.path.splitext(image_file)[0]
        label_file = base_name + ".txt"
        label_path = os.path.join(labels_folder, label_file)

        # 如果对应的标签文件存在，则添加到有效对列表中
        if os.path.exists(label_path):
            valid_pairs.append((image_file, label_file))
        else:
            print(f"警告：图片 {image_file} 没有对应的标签文件 {label_file}，已跳过")

    print(f"找到 {len(valid_pairs)} 个有效的图片-标签对")

    if len(valid_pairs) == 0:
        print("没有找到有效的图片-标签对，无法划分数据集")
        return

    # 随机打乱数据
    random.shuffle(valid_pairs)

    # 计算各数据集的数量
    total_count = len(valid_pairs)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count

    print(f"\n数据集划分:")
    print(f"  训练集: {train_count} 个样本 ({train_ratio * 100:.1f}%)")
    print(f"  验证集: {val_count} 个样本 ({val_ratio * 100:.1f}%)")
    print(f"  测试集: {test_count} 个样本 ({test_ratio * 100:.1f}%)")

    # 划分数据集
    train_pairs = valid_pairs[:train_count]
    val_pairs = valid_pairs[train_count:train_count + val_count]
    test_pairs = valid_pairs[train_count + val_count:]

    # 复制训练集文件
    print("\n正在复制训练集文件...")
    for image_file, label_file in tqdm(train_pairs, desc="训练集"):
        # 复制图片
        src_image_path = os.path.join(images_folder, image_file)
        dst_image_path = os.path.join(output_folder, "images", "train", image_file)
        shutil.copy2(src_image_path, dst_image_path)

        # 复制标签
        src_label_path = os.path.join(labels_folder, label_file)
        dst_label_path = os.path.join(output_folder, "labels", "train", label_file)
        shutil.copy2(src_label_path, dst_label_path)

    # 复制验证集文件
    print("\n正在复制验证集文件...")
    for image_file, label_file in tqdm(val_pairs, desc="验证集"):
        # 复制图片
        src_image_path = os.path.join(images_folder, image_file)
        dst_image_path = os.path.join(output_folder, "images", "val", image_file)
        shutil.copy2(src_image_path, dst_image_path)

        # 复制标签
        src_label_path = os.path.join(labels_folder, label_file)
        dst_label_path = os.path.join(output_folder, "labels", "val", label_file)
        shutil.copy2(src_label_path, dst_label_path)

    # 复制测试集文件
    print("\n正在复制测试集文件...")
    for image_file, label_file in tqdm(test_pairs, desc="测试集"):
        # 复制图片
        src_image_path = os.path.join(images_folder, image_file)
        dst_image_path = os.path.join(output_folder, "images", "test", image_file)
        shutil.copy2(src_image_path, dst_image_path)

        # 复制标签
        src_label_path = os.path.join(labels_folder, label_file)
        dst_label_path = os.path.join(output_folder, "labels", "test", label_file)
        shutil.copy2(src_label_path, dst_label_path)

    # 显示总结信息
    print(f"\n数据集划分完成！")
    print(f"输出目录结构:")
    print(f"  {output_folder}/")
    print(f"    images/")
    print(f"      train/ - {len(train_pairs)} 个图片文件")
    print(f"      val/   - {len(val_pairs)} 个图片文件")
    print(f"      test/  - {len(test_pairs)} 个图片文件")
    print(f"    labels/")
    print(f"      train/ - {len(train_pairs)} 个标签文件")
    print(f"      val/   - {len(val_pairs)} 个标签文件")
    print(f"      test/  - {len(test_pairs)} 个标签文件")

    # 显示一些示例文件
    print(f"\n示例文件:")
    if train_pairs:
        print(f"  训练集示例: {train_pairs[0][0]} -> {train_pairs[0][1]}")
    if val_pairs:
        print(f"  验证集示例: {val_pairs[0][0]} -> {val_pairs[0][1]}")
    if test_pairs:
        print(f"  测试集示例: {test_pairs[0][0]} -> {test_pairs[0][1]}")


# 主程序
if __name__ == "__main__":
    # 设置文件夹路径
    images_folder = r"D:\Program Files (x86)\deeplearning\make_dataset\images"
    labels_folder = r"D:\Program Files (x86)\deeplearning\make_dataset\labels"
    output_folder = r"D:\Program Files (x86)\deeplearning\make_dataset\kunkun"

    # 设置划分比例（总和必须为1）
    train_ratio = 0.7  # 训练集70%
    val_ratio = 0.2  # 验证集20%
    test_ratio = 0.1  # 测试集10%

    # 调用函数开始划分数据集
    split_dataset(
        images_folder=images_folder,
        labels_folder=labels_folder,
        output_folder=output_folder,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )