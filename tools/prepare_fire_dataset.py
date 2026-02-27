#!/usr/bin/env python3
"""
火灾检测数据集准备工具
=====================
功能：
1. 将 D-Fire 数据集（YOLO格式）转换为 COCO JSON 格式
2. 将 gengyanlei 数据集（VOC XML格式）转换为 COCO JSON 格式
3. 合并两个数据集并按比例划分训练集/验证集
4. 支持加入背景负样本（关键：可将误检率从 11% 降至 1.1%）

许可证合规：
- D-Fire Dataset：CC0 1.0（无任何限制）
- gengyanlei 数据集：MIT（需保留版权声明）

使用方法：
    # 准备数据
    python prepare_fire_dataset.py \
        --dfire_dir /path/to/DFireDataset \
        --gengyanlei_dir /path/to/fire-smoke-detect-yolov4 \
        --output_dir /path/to/PaddleDetection/dataset/fire_detection \
        --background_dir /path/to/background_images \
        --val_ratio 0.1
"""

import os
import sys
import json
import shutil
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional

# 类别定义（与训练配置保持一致）
CATEGORIES = [
    {"id": 1, "name": "fire",  "supercategory": "fire_smoke"},
    {"id": 2, "name": "smoke", "supercategory": "fire_smoke"},
]

CLASS_NAME_TO_ID = {cat["name"]: cat["id"] for cat in CATEGORIES}


def yolo_to_coco_bbox(yolo_bbox, img_w, img_h):
    """YOLO 归一化坐标 → COCO [x_min, y_min, width, height]"""
    cx, cy, w, h = yolo_bbox
    cx, cy, w, h = cx * img_w, cy * img_h, w * img_w, h * img_h
    x_min = cx - w / 2
    y_min = cy - h / 2
    return [x_min, y_min, w, h]


def load_dfire_dataset(dfire_dir: str) -> List[Dict]:
    """
    加载 D-Fire 数据集（YOLO 格式）
    
    D-Fire 目录结构：
    dfire_dir/
    ├── images/
    │   ├── train/  ← 图片
    │   └── test/
    └── labels/
        ├── train/  ← YOLO 格式标注（.txt）
        └── test/
    
    类别 ID（D-Fire 原始）：
    0 = fire, 1 = smoke
    """
    from PIL import Image  # pip install Pillow

    dfire_path = Path(dfire_dir)
    samples = []

    for split in ["train", "test"]:
        img_dir = dfire_path / "images" / split
        lbl_dir = dfire_path / "labels" / split

        if not img_dir.exists():
            # 尝试扁平结构
            img_dir = dfire_path / "images"
            lbl_dir = dfire_path / "labels"

        if not img_dir.exists():
            print(f"  [跳过] D-Fire {split} 目录不存在: {img_dir}")
            continue

        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
        print(f"  D-Fire {split}: 找到 {len(img_files)} 张图片")

        for img_path in img_files:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            bboxes = []

            if lbl_path.exists():
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cls_id = int(parts[0])
                        # D-Fire: 0=fire, 1=smoke → 映射到 COCO id (1=fire, 2=smoke)
                        coco_cls_id = cls_id + 1
                        if coco_cls_id not in (1, 2):
                            continue
                        coords = [float(x) for x in parts[1:]]
                        bboxes.append((coco_cls_id, coords))

            try:
                with Image.open(img_path) as img:
                    w, h = img.size
            except Exception:
                print(f"  [警告] 无法读取图片：{img_path}")
                continue

            samples.append({
                "img_path": str(img_path),
                "width": w,
                "height": h,
                "bboxes": bboxes,  # [(coco_cls_id, [cx, cy, w, h] normalized)]
                "source": "dfire"
            })

    print(f"D-Fire 共加载 {len(samples)} 个样本")
    return samples


def load_gengyanlei_dataset(gengyanlei_dir: str) -> List[Dict]:
    """
    加载 gengyanlei 数据集（VOC XML 格式）
    
    gengyanlei 目录结构（通常）：
    gengyanlei_dir/
    ├── images/  ← 图片
    └── Annotations/  ← VOC XML 标注
    """
    gengyanlei_path = Path(gengyanlei_dir)
    samples = []

    # 寻找图片和标注目录
    img_dirs = [gengyanlei_path / "images", gengyanlei_path / "JPEGImages", gengyanlei_path]
    anno_dirs = [gengyanlei_path / "Annotations", gengyanlei_path / "annotations"]

    img_dir = next((d for d in img_dirs if d.exists() and any(d.glob("*.jpg"))), None)
    anno_dir = next((d for d in anno_dirs if d.exists()), None)

    if img_dir is None or anno_dir is None:
        print(f"  [警告] gengyanlei 目录结构不匹配，跳过: {gengyanlei_dir}")
        return []

    xml_files = list(anno_dir.glob("*.xml"))
    print(f"  gengyanlei: 找到 {len(xml_files)} 个 XML 标注")

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            continue

        # 查找图片文件
        filename = root.findtext("filename", "")
        img_path = img_dir / filename
        if not img_path.exists():
            # 尝试不同扩展名
            for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
                candidate = img_dir / (xml_path.stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            else:
                continue

        size_node = root.find("size")
        if size_node is not None:
            w = int(size_node.findtext("width", 0))
            h = int(size_node.findtext("height", 0))
        else:
            from PIL import Image
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
            except Exception:
                continue

        if w == 0 or h == 0:
            continue

        bboxes = []
        for obj in root.iter("object"):
            name = obj.findtext("name", "").lower().strip()
            # 规范化类别名称
            if name in ("fire", "flame"):
                coco_cls_id = 1
            elif name in ("smoke", "smog"):
                coco_cls_id = 2
            else:
                continue

            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            xmin = float(bnd.findtext("xmin", 0))
            ymin = float(bnd.findtext("ymin", 0))
            xmax = float(bnd.findtext("xmax", w))
            ymax = float(bnd.findtext("ymax", h))

            box_w = xmax - xmin
            box_h = ymax - ymin
            if box_w <= 0 or box_h <= 0:
                continue

            # 转为 YOLO 归一化格式（统一存储，后续再转 COCO）
            cx = (xmin + box_w / 2) / w
            cy = (ymin + box_h / 2) / h
            bboxes.append((coco_cls_id, [cx, cy, box_w / w, box_h / h]))

        samples.append({
            "img_path": str(img_path),
            "width": w,
            "height": h,
            "bboxes": bboxes,
            "source": "gengyanlei"
        })

    print(f"gengyanlei 共加载 {len(samples)} 个样本")
    return samples


def load_background_samples(background_dir: str) -> List[Dict]:
    """加载背景负样本（无火焰无烟雾）"""
    if not background_dir or not Path(background_dir).exists():
        return []

    bg_path = Path(background_dir)
    img_files = (list(bg_path.glob("*.jpg")) + list(bg_path.glob("*.png")) +
                 list(bg_path.glob("*.jpeg")) + list(bg_path.rglob("*.jpg")))

    samples = []
    from PIL import Image
    for img_path in img_files[:2000]:  # 最多 2000 张背景图
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception:
            continue
        samples.append({
            "img_path": str(img_path),
            "width": w,
            "height": h,
            "bboxes": [],  # 空标注 = 背景图
            "source": "background"
        })

    print(f"背景负样本 共加载 {len(samples)} 个样本")
    return samples


def build_coco_json(samples: List[Dict], img_dst_dir: str, split_name: str) -> Dict:
    """将样本列表转换为 COCO JSON 格式，并复制图片到目标目录"""
    os.makedirs(img_dst_dir, exist_ok=True)

    coco = {
        "info": {
            "description": "Fire and Smoke Detection Dataset",
            "version": "1.0",
            "year": 2026,
            "licenses": [
                {"id": 1, "name": "CC0 1.0 (D-Fire)", "url": "https://creativecommons.org/publicdomain/zero/1.0/"},
                {"id": 2, "name": "MIT (gengyanlei)", "url": "https://opensource.org/licenses/MIT"},
            ],
            "classes": ["fire", "smoke"],
            "note": "Background images (empty annotations) are included to reduce false positive rate."
        },
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    ann_id = 1
    for img_id, sample in enumerate(samples, start=1):
        src_path = Path(sample["img_path"])
        ext = src_path.suffix.lower()
        dst_filename = f"{split_name}_{img_id:06d}{ext}"
        dst_path = Path(img_dst_dir) / dst_filename

        # 复制图片
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)

        coco["images"].append({
            "id": img_id,
            "file_name": dst_filename,
            "width": sample["width"],
            "height": sample["height"],
            "source": sample.get("source", "unknown"),
        })

        for coco_cls_id, (cx, cy, bw, bh) in [(b[0], b[1]) for b in sample["bboxes"]]:
            x_min, y_min, box_w, box_h = yolo_to_coco_bbox(
                [cx, cy, bw, bh], sample["width"], sample["height"]
            )
            # 边界检查
            x_min = max(0.0, x_min)
            y_min = max(0.0, y_min)
            box_w = min(box_w, sample["width"] - x_min)
            box_h = min(box_h, sample["height"] - y_min)
            if box_w <= 1 or box_h <= 1:
                continue
            area = box_w * box_h

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": coco_cls_id,
                "bbox": [round(x_min, 2), round(y_min, 2), round(box_w, 2), round(box_h, 2)],
                "area": round(area, 2),
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1

    fire_count = sum(1 for a in coco["annotations"] if a["category_id"] == 1)
    smoke_count = sum(1 for a in coco["annotations"] if a["category_id"] == 2)
    bg_count = sum(1 for s in samples if not s["bboxes"])
    print(f"  {split_name}: {len(samples)} 图片, {fire_count} fire框, {smoke_count} smoke框, {bg_count} 背景图")

    return coco


def main():
    parser = argparse.ArgumentParser(description="火灾检测数据集准备工具")
    parser.add_argument("--dfire_dir", type=str, default="",
                        help="D-Fire 数据集目录（CC0，推荐主力数据集）")
    parser.add_argument("--gengyanlei_dir", type=str, default="",
                        help="gengyanlei 数据集目录（MIT，PaddlePaddle 官方使用）")
    parser.add_argument("--background_dir", type=str, default="",
                        help="背景负样本目录（无标注图片，强烈推荐，可将误检率从 11% → 1.1%）")
    parser.add_argument("--output_dir", type=str,
                        default="dataset/fire_detection",
                        help="输出目录（PaddleDetection/dataset/fire_detection）")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集比例（默认 10%）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（保证可复现性）")
    args = parser.parse_args()

    print("=" * 60)
    print("🔥 火灾检测数据集准备工具")
    print("=" * 60)

    random.seed(args.seed)

    # 1. 加载所有数据集
    all_samples = []

    if args.dfire_dir:
        print("\n[1/3] 加载 D-Fire 数据集（CC0 1.0）...")
        dfire_samples = load_dfire_dataset(args.dfire_dir)
        all_samples.extend(dfire_samples)
    else:
        print("\n[1/3] 跳过 D-Fire（未指定目录）")

    if args.gengyanlei_dir:
        print("\n[2/3] 加载 gengyanlei 数据集（MIT）...")
        gengyanlei_samples = load_gengyanlei_dataset(args.gengyanlei_dir)
        all_samples.extend(gengyanlei_samples)
    else:
        print("\n[2/3] 跳过 gengyanlei（未指定目录）")

    if args.background_dir:
        print("\n[3/3] 加载背景负样本...")
        bg_samples = load_background_samples(args.background_dir)
        # 背景图只加入训练集
        bg_train = bg_samples
    else:
        print("\n[3/3] 无背景负样本（建议添加以降低误检率）")
        bg_train = []

    if not all_samples:
        print("\n⚠️  没有加载到任何样本，请检查目录路径")
        sys.exit(1)

    print(f"\n总计：{len(all_samples)} 个有标注样本（+ {len(bg_train)} 背景图）")

    # 2. 划分训练集/验证集（背景图只放训练集）
    random.shuffle(all_samples)
    val_size = max(1, int(len(all_samples) * args.val_ratio))
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:] + bg_train

    print(f"训练集：{len(train_samples)} 张（含 {len(bg_train)} 背景图）")
    print(f"验证集：{len(val_samples)} 张")

    # 3. 构建 COCO JSON 并复制图片
    output_path = Path(args.output_dir)
    anno_dir = output_path / "annotations"
    os.makedirs(anno_dir, exist_ok=True)

    print("\n[构建训练集 COCO JSON...]")
    train_coco = build_coco_json(
        train_samples,
        str(output_path / "images" / "train"),
        "train"
    )

    print("[构建验证集 COCO JSON...]")
    val_coco = build_coco_json(
        val_samples,
        str(output_path / "images" / "val"),
        "val"
    )

    # 4. 保存 JSON
    train_json_path = anno_dir / "train.json"
    val_json_path = anno_dir / "val.json"

    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(train_coco, f, ensure_ascii=False, indent=2)

    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_coco, f, ensure_ascii=False, indent=2)

    # 5. 生成 label_list.txt
    label_list_path = output_path / "label_list.txt"
    with open(label_list_path, "w") as f:
        f.write("fire\nsmoke\n")

    print(f"\n✅ 数据集准备完成！")
    print(f"   输出目录：{output_path.resolve()}")
    print(f"   训练集标注：{train_json_path}")
    print(f"   验证集标注：{val_json_path}")
    print(f"   类别文件：{label_list_path}")
    print(f"\n下一步：")
    print(f"   cd PaddleDetection")
    print(f"   python tools/train.py \\")
    print(f"       -c configs/ppyoloe/application/ppyoloe_plus_crn_s_80e_fire_detection.yml \\")
    print(f"       -o weights=ppyoloe_plus_crn_s_80e_coco.pdparams \\")
    print(f"       --eval --amp")


if __name__ == "__main__":
    main()
