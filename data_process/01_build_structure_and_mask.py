import os
import json
import shutil
import cv2
import numpy as np
from pathlib import Path

# 脚本说明：
# 这个脚本用于从原始数据集中重组新的数据结构，并根据 COCO 格式的 JSON 文件生成对应的 Mask 图像。
# 原始数据集架构：
# dataset
#  |-------case1
#  |-------|------high
#  |-------|-------|------1.avi
#  |-------|-------|------1-1.png
#  |-------|-------|------1-1.txt
#  |-------|-------|------1-2.png
#  |-------|-------|------1-2.txt
#  |-------|-------|------label.json
#  |-------|------low
#  |-------|-------|------1.avi
#  |-------|-------|------1-1.png
#  |-------|-------|------1-1.txt
#  |-------|-------|------label.json
# 处理后数据集架构：
# clip-dataset
#  |-------case1
#  |-------|------normal
#  |-------|-------|-------high
#  |-------|-------|-------|------video.avi  固定命名为 video.avi
#  |-------|-------|-------|------fetal ultrasound, fetal eyes.
#  |-------|-------|-------|-------|------imgs
#  |-------|-------|-------|-------|------|-------1-1.png
#  |-------|-------|-------|-------|------masks
#  |-------|-------|-------|-------|------|-------1-1_mask.png
#  |-------|-------|-------|-------|------1-1.txt
#  |-------|-------|-------|------fetal ultrasound, hard palate.
#  |-------|-------|-------|-------|------imgs
#  |-------|-------|-------|-------|------|-------1-2.png
#  |-------|-------|-------|-------|------masks
#  |-------|-------|-------|-------|------|-------1-2_mask.png
#  |-------|-------|-------|-------|------1-2.txt
#  |-------|-------|-------|------label.json  固定命名为 label.json
#  |-------|-------|-------low
#  |-------|-------|-------|------video.avi
#  |-------|-------|-------|------label.json
#  |-------|------abnormal
#
# 使用方式：直接运行该脚本，确保 SRC_DIR 和 DST_DIR 路径正确设置。脚本会自动处理所有 case，并生成新的数据结构和 Mask 图像。

# ==========================================
# 配置参数
# ==========================================
SRC_DIR = Path(r"C:\AI_data\ultrasound\final_data\raw-preprocess")            # 原始数据集路径
DST_DIR = Path(r"C:\AI_data\ultrasound\final_data\clip-dataset")       # 输出数据集路径

def parse_coco_json(json_path):
    """
    解析标准的 COCO 格式 JSON，建立文件名到多边形坐标的映射。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 1. 建立 file_name -> image_id 的映射
    img_name_to_id = {}
    for img in data.get('images', []):
        img_name_to_id[img['file_name']] = img['id']
        
    # 2. 建立 image_id -> list of segmentations 的映射
    id_to_polygons = {}
    for ann in data.get('annotations', []):
        img_id = ann['image_id']
        # COCO 的 segmentation 通常是 [[x1, y1, x2, y2...]]
        segs = ann.get('segmentation', [])
        
        if img_id not in id_to_polygons:
            id_to_polygons[img_id] = []
        id_to_polygons[img_id].extend(segs)
        
    return img_name_to_id, id_to_polygons

# def generate_mask_from_polygons(image_path, polygons, output_mask_path):
#     """
#     将一维多边形坐标渲染为二值化 Mask。
#     """
#     img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         print(f"[警告] 无法读取图像，跳过 Mask 生成: {image_path}")
#         return False
    
#     h, w = img.shape
#     mask = np.zeros((h, w), dtype=np.uint8)
    
#     if polygons:
#         formatted_polygons = []
#         for poly in polygons:
#             # 将 [x1, y1, x2, y2...] 转换为 numpy 数组，并 reshape 为 (N, 2)
#             pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
#             formatted_polygons.append(pts)
            
#         # 填充多边形 (值为 255)
#         cv2.fillPoly(mask, formatted_polygons, 255)
#     else:
#         print(f"[提示] {image_path.name} 未找到多边形标注，生成全黑 Mask。")
    
#     cv2.imwrite(str(output_mask_path), mask)
#     return True

def generate_mask_from_polygons(image_path, polygons, output_mask_path):
    """
    支持中文路径的多边形 Mask 生成函数。
    """
    # --- 关键修改点 1: 使用 numpy 读取中文路径图像 ---
    try:
        # np.fromfile 可以正常读取包含中文的路径
        img_array = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"[错误] 读取文件异常: {image_path}, 原因: {e}")
        return False

    if img is None:
        print(f"[警告] 无法读取图像，跳过 Mask 生成: {image_path}")
        return False
    
    h, w = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if polygons:
        formatted_polygons = []
        for poly in polygons:
            # 将 [x1, y1, x2, y2...] 转换为 (N, 2)
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            formatted_polygons.append(pts)
            
        # 填充多边形 (值为 255)
        cv2.fillPoly(mask, formatted_polygons, 255)
    else:
        print(f"[提示] {image_path.name} 未找到多边形标注，生成全黑 Mask。")
    
    # --- 关键修改点 2: 使用 numpy 编码并写入中文路径 ---
    try:
        # 先将图像编码为 png 格式的内存缓冲区
        _, buffer = cv2.imencode('.png', mask)
        # 将缓冲区数据写入文件
        buffer.tofile(str(output_mask_path))
    except Exception as e:
        print(f"[错误] 写入 Mask 失败: {output_mask_path}, 原因: {e}")
        return False

    return True

def process_dataset():
    if not SRC_DIR.exists():
        print(f"[错误] 源目录 {SRC_DIR} 不存在！")
        return

    for case_dir in SRC_DIR.iterdir():
        if not case_dir.is_dir():
            continue
            
        case_name = case_dir.name
        print(f"========== 开始处理: {case_name} ==========")
        
        for quality in ['high', 'low']:
            quality_dir = case_dir / quality
            if not quality_dir.exists():
                continue
                
            dst_quality_dir = DST_DIR / case_name / "normal" / quality
            dst_quality_dir.mkdir(parents=True, exist_ok=True)
            
            # 查找第一个 JSON 文件
            json_files = list(quality_dir.glob("*.json"))
            
            if json_files:
                # 获取第一个 JSON 文件
                json_path = json_files[0]
                img_name_to_id = {}
                id_to_polygons = {}
                
                # 解析 COCO JSON
                img_name_to_id, id_to_polygons = parse_coco_json(json_path)
                
                # 将第一个 JSON 文件复制到目标目录
                shutil.copy2(json_path, dst_quality_dir / "label.json")
            else:
                print(f"[警告] {quality_dir} 下未找到 JSON 文件")
                
            # 复制视频文件
            video_path = quality_dir / "video_1.avi"
            if video_path.exists():
                shutil.copy2(video_path, dst_quality_dir / "video.avi")
                
            # 处理图像、文本和 Mask
            for txt_file in quality_dir.glob("*.txt"):
                img_name = txt_file.stem + ".png"
                img_path = quality_dir / img_name
                
                if not img_path.exists():
                    print(f"[警告] 找到 {txt_file.name} 但缺失图像 {img_name}")
                    continue
                    
                with open(txt_file, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                
                if not first_line:
                    continue
                    
                category_name = first_line
                
                # 构建目录
                category_dir = dst_quality_dir / category_name
                imgs_dir = category_dir / "imgs"
                masks_dir = category_dir / "masks"
                
                imgs_dir.mkdir(parents=True, exist_ok=True)
                masks_dir.mkdir(parents=True, exist_ok=True)
                
                # 迁移 txt 和 png
                shutil.copy2(txt_file, category_dir / txt_file.name)
                shutil.copy2(img_path, imgs_dir / img_name)
                
                # 提取对应的多边形并生成 Mask
                polygons = []
                # 检查 JSON 中是否存在该文件名
                if img_name in img_name_to_id:
                    img_id = img_name_to_id[img_name]
                    polygons = id_to_polygons.get(img_id, [])
                else:
                    print(f"[提示] JSON 中未匹配到文件名: {img_name}，将生成全黑 Mask。")
                
                dst_mask_name = f"{txt_file.stem}_mask.png"
                dst_mask_path = masks_dir / dst_mask_name
                
                generate_mask_from_polygons(img_path, polygons, dst_mask_path)
                
        print(f"========== {case_name} 处理完成 ==========\n")

if __name__ == "__main__":
    process_dataset()
    print("所有数据结构重组与 Mask 生成已完成！")