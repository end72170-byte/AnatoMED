import os
import cv2
import shutil
import numpy as np
from pathlib import Path

# 这个脚本用于在 clip-dataset / sam-dataset 中针对特定类别（如硬腭、牙槽骨）生成极其逼真的多特征异常样本，模拟真实的裂隙缺损。
# 主要步骤包括：
# 1. 生成形态不规则的裂隙 Mask，模拟不同类型的断裂位置和形态。
# 2. 根据原图中背景暗区的统计特征，生成逼真的散斑噪声，并注入裂隙区域。
# 3. 局部模糊处理，模拟容积效应和边缘模糊。
# 4. 施加声学后方增强效应，模拟超声穿透增强。
# 5. 生成高亮 Overlay 可视化，突出显示健康组织和断裂区域。
# 6. 保存生成的异常图像、更新后的 Mask 和 Overlay 可视化图。

# ==========================================
# 配置参数
# ==========================================
# 你可以在这里切换是处理 clip-dataset 还是 sam-dataset
DATASET_DIR = Path(r"C:\AI_data\ultrasound\final_data\clip-dataset") 

TARGET_CATEGORIES = [
    "fetal ultrasound, hard palate",
    "fetal ultrasound, upper alveolar ridge",
    "fetal ultrasound, lower alveolar ridge"
]

OVERLAY_ALPHA = 0.5  # 可视化透明度

# ==========================================
# 中文路径安全读取工具
# ==========================================
def cv2_imread_chinese(filepath, flags=cv2.IMREAD_COLOR):
    try:
        img_array = np.fromfile(str(filepath), dtype=np.uint8)
        return cv2.imdecode(img_array, flags)
    except Exception as e:
        print(f"[错误] 读取异常: {filepath}, {e}")
        return None

def cv2_imwrite_chinese(filepath, img):
    try:
        _, ext = os.path.splitext(str(filepath))
        _, buffer = cv2.imencode(ext, img)
        buffer.tofile(str(filepath))
        return True
    except Exception as e:
        print(f"[错误] 写入异常: {filepath}, {e}")
        return False

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ==========================================
# 核心伪造算法
# ==========================================
class CleftSynthesizer:
    @staticmethod
    def get_background_speckle_stats(img_gray, mask):
        """获取原图中背景暗区（非骨质区域）的统计学特征，用于生成逼真的散斑"""
        # 寻找图像中偏暗且不在 Mask 内的区域
        bg_pixels = img_gray[(mask == 0) & (img_gray > 5) & (img_gray < 80)]
        if len(bg_pixels) < 100:
            # 如果背景太干净，给定一个默认的超声散斑分布
            return 20.0, 10.0
        return np.mean(bg_pixels), np.std(bg_pixels)

    @staticmethod
    def generate_cleft_mask(mask, category):
        """生成形态不规则的裂隙 Mask"""
        h, w = mask.shape
        y_indices, x_indices = np.where(mask > 0)
        
        if len(x_indices) == 0:
            return np.zeros_like(mask)
            
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        width = max_x - min_x
        
        # 1. 类别微调：确定裂隙起始的 X 坐标
        if "hard palate" in category.lower():
            # 硬腭：正中偏离一点 (正态分布)
            center_x = min_x + width / 2
            cut_x = int(np.random.normal(center_x, width * 0.1))
        else:
            # 牙槽骨：双侧侧切牙位置 (随机选择左侧或右侧的 1/4 到 1/3 处)
            if np.random.rand() > 0.5:
                cut_x = int(np.random.normal(min_x + width * 0.25, width * 0.05))
            else:
                cut_x = int(np.random.normal(min_x + width * 0.75, width * 0.05))
                
        # 防止越界
        cut_x = np.clip(cut_x, min_x + 5, max_x - 5)
        
        gap_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 2. 随机游走生成裂隙路径 (模拟不规则边缘)
        current_x = cut_x
        # 从 mask 最上缘上方一点，延伸到最下缘下方一点
        start_y = max(0, min_y - 15)
        end_y = min(h, max_y + 15)
        
        for y in range(start_y, end_y, 5):
            # 随机左右游走，产生弯曲感
            current_x += np.random.randint(-4, 5)
            # 裂隙宽度随机变化 (模拟 3-10 pixel 的物理缺损)
            radius = np.random.randint(3, 10)
            cv2.circle(gap_mask, (int(current_x), int(y)), radius, 255, -1)
            
        # 3. 使用高斯模糊和阈值处理，让裂隙的边缘变得圆滑和虫蚀样
        gap_mask = cv2.GaussianBlur(gap_mask, (11, 11), 0)
        _, gap_mask = cv2.threshold(gap_mask, 127, 255, cv2.THRESH_BINARY)
        
        return gap_mask

    @staticmethod
    def apply_acoustic_enhancement(img_bgr, gap_mask, mask, intensity=25):
        """模拟声学穿透增强效应 (Acoustic Enhancement)"""
        h, w, _ = img_bgr.shape
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # 找到裂隙最下方的坐标
        y_indices, x_indices = np.where((gap_mask > 0) & (mask > 0))
        if len(y_indices) == 0:
            return img_bgr
            
        bottom_y = np.max(y_indices)
        center_x = int(np.mean(x_indices))
        
        # 在裂隙下方生成一个梯度的亮区
        enhancement_mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(enhancement_mask, (center_x, bottom_y + 40), (25, 60), 0, 0, 180, 1.0, -1)
        enhancement_mask = cv2.GaussianBlur(enhancement_mask, (51, 51), 0)
        
        # 提升亮度通道 (V)
        v_channel = img_hsv[:, :, 2].astype(np.float32)
        v_channel += enhancement_mask * intensity
        img_hsv[:, :, 2] = np.clip(v_channel, 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def synthesize(img_bgr, mask, category):
        """综合执行伪造流水线"""
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. 生成裂隙形态
        gap_mask = CleftSynthesizer.generate_cleft_mask(mask, category)
        
        # 2. 计算需要被破坏的有效区域 (原 mask 与裂隙的交集)
        effective_gap = cv2.bitwise_and(mask, gap_mask)
        if cv2.countNonZero(effective_gap) < 20:
            # 如果交集太小，说明随机切断偏了，重试一次或返回空
            return None, None, None
            
        # 3. 获取背景散斑特征
        bg_mean, bg_std = CleftSynthesizer.get_background_speckle_stats(img_gray, mask)
        
        # 4. 生成散斑噪声并注入裂隙
        noise = np.random.normal(bg_mean, bg_std, img_gray.shape).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (3, 3), 0) # 轻微模糊匹配超声质感
        
        synth_img = img_bgr.copy()
        # 将 BGR 转换为独立通道进行替换
        for i in range(3):
            channel = synth_img[:, :, i].astype(np.float32)
            channel[effective_gap > 0] = noise[effective_gap > 0]
            synth_img[:, :, i] = np.clip(channel, 0, 255).astype(np.uint8)
            
        # 5. 局部模糊边缘 (模拟部分容积效应)
        blurred_img = cv2.GaussianBlur(synth_img, (7, 7), 0)
        edge_mask = cv2.dilate(effective_gap, np.ones((5, 5), np.uint8), iterations=1)
        synth_img[edge_mask > 0] = blurred_img[edge_mask > 0]
        
        # 6. 施加声学后方增强
        synth_img = CleftSynthesizer.apply_acoustic_enhancement(synth_img, effective_gap, mask)
        
        # 7. 更新 Mask
        new_mask = cv2.bitwise_and(mask, cv2.bitwise_not(gap_mask))
        
        return synth_img, new_mask, effective_gap

# ==========================================
# 主流程控制
# ==========================================
def process_anomalies():
    if not DATASET_DIR.exists():
        print(f"[错误] 数据集目录 {DATASET_DIR} 不存在！")
        return

    print(f"========== 开始针对 {DATASET_DIR.name} 执行多特征异常伪造 ==========")
    
    total_synthesized = 0

    for case_dir in DATASET_DIR.iterdir():
        if not case_dir.is_dir():
            continue
            
        print(f"\n[处理 Case] {case_dir.name}")
        
        # 遍历 normal/high 和 normal/low
        for quality in ['high', 'low']:
            quality_dir = case_dir / "normal" / quality
            if not quality_dir.exists():
                continue
                
            for category in TARGET_CATEGORIES:
                src_category_dir = quality_dir / category
                if not src_category_dir.exists():
                    continue
                    
                src_imgs_dir = src_category_dir / "imgs"
                src_masks_dir = src_category_dir / "masks"
                
                # 构建 abnormal 目标路径
                dst_category_dir = case_dir / "abnormal" / quality / category
                dst_imgs_dir = dst_category_dir / "imgs"
                dst_masks_dir = dst_category_dir / "masks"
                dst_overlay_dir = dst_category_dir / "overlay"
                
                ensure_dir(dst_imgs_dir)
                ensure_dir(dst_masks_dir)
                ensure_dir(dst_overlay_dir)
                
                # 复制并重命名 txt 文本 (这里暂时原样复制，等 Step 5 交给 LLM 篡改)
                src_txt = list(src_category_dir.glob("*.txt"))
                if src_txt:
                    shutil.copy2(src_txt[0], dst_category_dir / src_txt[0].name)
                
                # 开始逐图伪造
                for img_path in src_imgs_dir.glob("*.png"):
                    img_name = img_path.name
                    mask_name = f"{img_path.stem}_mask.png"
                    mask_path = src_masks_dir / mask_name
                    
                    if not mask_path.exists():
                        continue
                        
                    img_bgr = cv2_imread_chinese(img_path)
                    mask_gray = cv2_imread_chinese(mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    # 运行合成引擎
                    synth_img, new_mask, gap_mask = CleftSynthesizer.synthesize(img_bgr, mask_gray, category)
                    
                    if synth_img is None:
                        # 生成失败（交集太小），跳过此图
                        continue
                        
                    # --- 生成高亮 Overlay 可视化 ---
                    overlay = synth_img.copy().astype(np.float32)
                    
                    # 健康组织 (new_mask) 标为半透明绿色
                    green = np.array([0, 255, 0], dtype=np.float32)
                    m_green = new_mask > 0
                    overlay[m_green] = (1 - OVERLAY_ALPHA) * overlay[m_green] + OVERLAY_ALPHA * green
                    
                    # 切断的裂隙区 (gap_mask) 标为半透明红色（醒目）
                    red = np.array([0, 0, 255], dtype=np.float32)
                    m_red = gap_mask > 0
                    overlay[m_red] = (1 - OVERLAY_ALPHA) * overlay[m_red] + OVERLAY_ALPHA * red
                    
                    overlay_bgr = np.clip(overlay, 0, 255).astype(np.uint8)
                    
                    # 保存结果
                    cv2_imwrite_chinese(dst_imgs_dir / img_name, synth_img)
                    cv2_imwrite_chinese(dst_masks_dir / mask_name, new_mask)
                    cv2_imwrite_chinese(dst_overlay_dir / img_name, overlay_bgr)
                    
                    total_synthesized += 1

    print(f"\n========== 伪造完成！成功生成 {total_synthesized} 组断裂样本。 ==========")

if __name__ == "__main__":
    process_anomalies()