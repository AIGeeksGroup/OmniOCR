import os
import shutil
import json
import logging
import random
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("omni_ocr_selection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OmniOCRSelector:
    """
    def __init__(self, dataset_dir, output_dir, selected_categories=None, max_samples_per_class=None):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        # 选取的 30 个类别
        self.selected_categories = selected_categories 
        self.max_samples_per_class = max_samples_per_class # 用于样本均衡
        os.makedirs(self.output_dir, exist_ok=True)

    def get_image_files(self, cat_dir):
        return [
            f for f in os.listdir(cat_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]

    def select_and_copy(self):
        if not self.selected_categories:
            logger.info("未指定类别列表，将按样本量排序并应用均衡化策略...")
            cat_counts = []
            for d in os.listdir(self.dataset_dir):
                d_path = os.path.join(self.dataset_dir, d)
                if os.path.isdir(d_path):
                    count = len(self.get_image_files(d_path))
                    if count > 0:
                        cat_counts.append((d, count))
            
            cat_counts.sort(key=lambda x: x[1], reverse=True)
            self.selected_categories = [x[0] for x in cat_counts[:30]]

        logger.info(f"开始处理选定的 {len(self.selected_categories)} 个代表性类别...")

        report = {}
        for category in tqdm(self.selected_categories, desc="处理类别"):
            src_dir = os.path.join(self.dataset_dir, category)
            if not os.path.exists(src_dir):
                logger.warning(f"跳过不存在的类别目录: {category}")
                continue

            dst_dir = os.path.join(self.output_dir, category)
            os.makedirs(dst_dir, exist_ok=True)

            all_imgs = self.get_image_files(src_dir)
            
            # 如果样本数超过阈值，进行随机采样以模拟“均衡写者分布”
            if self.max_samples_per_class and len(all_imgs) > self.max_samples_per_class:
                selected_imgs = random.sample(all_imgs, self.max_samples_per_class)
            else:
                selected_imgs = all_imgs

            for img_file in selected_imgs:
                shutil.copy2(
                    os.path.join(src_dir, img_file),
                    os.path.join(dst_dir, img_file)
                )
            
            report[category] = len(selected_imgs)
            
        result = {
            "top30_categories": top30,
            "samples_count": top30_samples,
            "output_dir": self.output_dir
        }
        with open("top30_most_samples.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"筛选完成！样本数最多的30个类别已保存到：{self.output_dir}")
        return top30, top30_samples

        # 保存筛选元数据
        with open("omni_selection_metadata.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"筛选完成。数据集已存至: {self.output_dir}")

if __name__ == "__main__":
    SELECTED_30 = None  # 如果为 None，代码将自动选取前30个高频类别
    
    DATASET_DIR = "../datasets/AncientYi"
    OUTPUT_DIR = "../datasets/OmniOCR_Yi_Balanced"

    selector = OmniOCRSelector(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        selected_categories=SELECTED_30,
    )

    selector.select_and_copy()
