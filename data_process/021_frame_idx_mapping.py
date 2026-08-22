import os
import cv2
import re
import json

# 这个脚本用于生成一个 JSON 文件，记录每个 PNG 文件的新时间节点映射
# 例如： "case17/high/1-1-00.04.27.png" -> "00.04.25" (假设标注帧率为 30 FPS，实际视频帧率为 25 FPS)
# 纠正时间节点，使其与实际视频帧率相匹配，避免索引越界问题

def generate_frame_map(root_dir, output_json, target_fps=30):
    time_map = {}
    
    print("开始处理目录，正在计算时间映射...")

    for subdir, dirs, files in os.walk(root_dir):
        # 1. 获取当前目录下的视频帧率
        video_file = next((f for f in files if f.endswith(('.avi', '.mp4'))), None)
        if not video_file:
            continue
            
        video_path = os.path.join(subdir, video_file)
        cap = cv2.VideoCapture(video_path)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if actual_fps <= 0:
            print(f"[警告] 无法读取视频帧率: {video_path}")
            continue

        # 2. 遍历该目录下的所有 PNG 文件
        png_files = [f for f in files if f.endswith('.png')]
        for png in png_files:
            # 正则匹配：提取 (秒数).(旧帧索引)
            # 兼容格式如：1-1-00.04.27.png 或 低11-3-04.27.png
            match = re.search(r'(\d{1,2})\.(\d{2})\.png$', png)
            if match:
                seconds = int(match.group(1))
                old_frame_idx = int(match.group(2))
                
                # 3. 计算重映射后的新帧索引
                # 使用 round 确保最接近物理帧，min 确保不越界
                new_frame_idx = int(round(old_frame_idx * (actual_fps / target_fps)))
                new_frame_idx = min(new_frame_idx, int(actual_fps) - 1)
                
                # 4. 生成新的时间节点字符串 (保持两位数格式)
                new_timestamp = f"{seconds:02d}.{new_frame_idx:02d}"
                
                # 5. 获取文件的相对路径作为 Key
                rel_path = os.path.relpath(os.path.join(subdir, png), root_dir)
                time_map[rel_path] = new_timestamp

    # 保存为 JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(time_map, f, indent=4, ensure_ascii=False)
    
    print(f"\n处理完成！映射表已保存至: {output_json}")
    print(f"总计映射样本数: {len(time_map)}")

if __name__ == "__main__":
    # 配置你的路径
    DATASET_ROOT = r'C:\AI_data\ultrasound\final_data\raw-preprocess'
    OUTPUT_FILE = 'frame_remapping_log.json'
    
    generate_frame_map(DATASET_ROOT, OUTPUT_FILE)