import os
import cv2
import json
import tempfile
import shutil
import numpy as np
from pathlib import Path

# 这个脚本用于扫描 clip-dataset 中的所有 PNG 文件，提取它们在对应视频中的时间节点（秒数.帧数），并保存到一个 JSON 文件中。
# 例如： "case17/high/1-1-00.04.27.png" -> "00.04.25" 
# 帧数以视频的实际帧率为准，使用感知哈希和颜色直方图进行匹配，确保时间节点的准确性。

# ==========================================
# 配置参数
# ==========================================
DST_DIR = Path(r"C:\AI_data\ultrasound\final_data\clip-dataset")       # 已经处理好的 clip-dataset 路径
OUTPUT_JSON = DST_DIR / "time.json"  # 最终保存的时间戳字典
W_HASH = 0.7                         # 感知哈希权重
W_HIST = 0.3                         # 颜色直方图权重
REFINE_SEC = 0.5                     # 在粗匹配结果前后 0.5 秒内逐帧精细搜索

# ==========================================
# 中文路径安全读取工具 (图像 & 视频)
# ==========================================
def cv2_imread_chinese(filepath):
    """安全读取包含中文路径的图像，返回 BGR 格式（与 cv2.imread 行为一致）"""
    try:
        img_array = np.fromfile(str(filepath), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[错误] 读取图像异常: {filepath}, 原因: {e}")
        return None

class SafeVideoCapture:
    """
    上下文管理器：安全读取包含中文路径的视频。
    如果 cv2 无法直接打开，会临时拷贝到 ASCII 路径下读取，结束后自动清理。
    """
    def __init__(self, path):
        self.path = str(path)
        self.temp_path = None
        self.cap = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.path)
        # 如果读取失败，大概率是因为中文路径
        if not self.cap.isOpened():
            _, ext = os.path.splitext(self.path)
            # 创建临时 ASCII 文件
            fd, self.temp_path = tempfile.mkstemp(suffix=ext)
            os.close(fd)
            shutil.copy2(self.path, self.temp_path)
            self.cap = cv2.VideoCapture(self.temp_path)
        return self.cap

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
        if self.temp_path and os.path.exists(self.temp_path):
            try:
                os.remove(self.temp_path)
            except:
                pass

# ==========================================
# 核心匹配算法
# ==========================================
def get_video_info(video_path: str):
    with SafeVideoCapture(video_path) as cap:
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return width, height, fps, total

def compute_descriptor(bgr: np.ndarray, hash_size: int = 8, hist_bins: int = 64):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    dhash = diff.astype(np.uint8).flatten()

    hist = cv2.calcHist([gray], [0], None, [hist_bins], [0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return dhash, hist

def score_descriptor(key_hash, key_hist, frame_hash, frame_hist, w_hash, w_hist):
    hamming = np.count_nonzero(key_hash != frame_hash) / float(key_hash.size)
    hist_corr = cv2.compareHist(key_hist, frame_hist, cv2.HISTCMP_CORREL)
    hist_dist = (1.0 - hist_corr) * 0.5
    return w_hash * hamming + w_hist * hist_dist

def build_frame_index(video_path: str, sample_fps: float = 25.0):
    width, height, fps, total = get_video_info(video_path)
    used_fps = min(fps, sample_fps) if sample_fps > 0 else fps
    step = max(1, int(round(fps / used_fps)))

    indices, hashes, hists = [], [], []
    idx = 0
    
    with SafeVideoCapture(video_path) as cap:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                h, hist = compute_descriptor(frame)
                indices.append(idx)
                hashes.append(h)
                hists.append(hist)
            idx += 1

    return {
        "indices": np.array(indices, dtype=np.int32),
        "hashes": np.stack(hashes, axis=0),
        "hists": np.stack(hists, axis=0),
        "fps": fps,
        "total": total,
        "step": step,
    }

def find_best_match(index_data, key_bgr, w_hash, w_hist):
    key_hash, key_hist = compute_descriptor(key_bgr)
    best_score = float("inf")
    best_idx = -1

    for i in range(index_data["indices"].shape[0]):
        score = score_descriptor(key_hash, key_hist, index_data["hashes"][i], 
                                 index_data["hists"][i], w_hash, w_hist)
        if score < best_score:
            best_score = score
            best_idx = int(index_data["indices"][i])

    return best_idx, best_score

def refine_match(video_path, center_idx, fps, refine_sec, key_bgr, w_hash, w_hist):
    if refine_sec <= 0:
        return center_idx
        
    _, _, _, total = get_video_info(video_path)
    start_idx = max(0, int(round(center_idx - refine_sec * fps)))
    end_idx = min(total - 1, int(round(center_idx + refine_sec * fps)))

    key_hash, key_hist = compute_descriptor(key_bgr)
    best_score = float("inf")
    best_idx = center_idx

    with SafeVideoCapture(video_path) as cap:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        cur = start_idx
        while cur <= end_idx:
            ok, frame = cap.read()
            if not ok:
                break
            frame_hash, frame_hist = compute_descriptor(frame)
            score = score_descriptor(key_hash, key_hist, frame_hash, frame_hist, w_hash, w_hist)
            if score < best_score:
                best_score = score
                best_idx = cur
            cur += 1

    return best_idx

# ==========================================
# 主流程：扫描并生成 time.json
# ==========================================
def process_time_nodes():
    if not DST_DIR.exists():
        print(f"[错误] 数据集目录 {DST_DIR} 不存在！请先执行 01 脚本。")
        return

    time_dict = {}

    # 遍历所有 case
    for case_dir in DST_DIR.iterdir():
        if not case_dir.is_dir():
            continue
            
        print(f"\n========== 正在索引: {case_dir.name} ==========")
        
        for quality in ['high', 'low']:
            quality_dir = case_dir / "normal" / quality
            video_path = quality_dir / "video.avi"

            if not quality_dir.exists():
                print(f"[警告] {quality_dir} 不存在，跳过该质量级别")
                continue
            
            if not video_path.exists():
                print(f"[警告] {quality_dir} 下未找到视频文件")
                continue
                
            # 查找该视频下的所有关键帧 png
            # 路径结构类似于: clip-dataset/case1/normal/high/双眼/imgs/1-1.png
            png_files = list(quality_dir.glob("*/imgs/*.png"))
            if not png_files:
                print(f"[警告] {quality_dir} 下未找到 PNG 文件")
                continue
                
            print(f"[INFO] 找到视频 {quality}/video.avi，开始建立特征索引库...")
            try:
                index_data = build_frame_index(str(video_path))
            except Exception as e:
                print(f"[错误] 视频 {video_path} 索引失败: {e}")
                continue
                
            fps = index_data['fps']
            fps_int = int(round(fps)) # 用于计算取模时的整数帧率

            for img_path in png_files:
                img_bgr = cv2_imread_chinese(img_path)
                if img_bgr is None:
                    continue
                    
                # 粗匹配 + 细匹配
                coarse_idx, _ = find_best_match(index_data, img_bgr, W_HASH, W_HIST)
                final_idx = refine_match(str(video_path), coarse_idx, fps, REFINE_SEC, img_bgr, W_HASH, W_HIST)
                
                # 计算 MM.FF (秒数.帧数)
                # 帧数不能超过实际 FPS - 1
                mm = int(final_idx // fps_int)
                ff = int(final_idx % fps_int)
                time_str = f"{mm:02d}.{ff:02d}"
                
                # 使用相对于 clip-dataset 的路径作为 Key，防止不同 case 下图片同名导致覆盖
                rel_path = str(img_path.relative_to(DST_DIR)).replace("\\", "/")
                
                time_dict[rel_path] = {
                    "time_format": time_str,
                    "frame_idx": int(final_idx),
                    "fps": round(fps, 2)
                }
                print(f"[OK] {img_path.name} -> {time_str} (总帧索引: {final_idx})")

    # 保存总的 time.json 到 clip-dataset 根目录
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(time_dict, f, indent=4, ensure_ascii=False)
    
    print(f"\n========== 所有时间节点提取完毕！已保存至 {OUTPUT_JSON} ==========")

if __name__ == "__main__":
    process_time_nodes()