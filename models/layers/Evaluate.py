import os
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report
import logging
import json
from accelerate import Accelerator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("lora_model_test_log.txt"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 模型和数据集路径
model_path = "./RolmOCR_lora_shui"
dataset_path = "../datasets/shui_datasets"

# 处理视觉信息
def process_vision_info(messages):
    image_inputs = []
    for message in messages:
        if message["role"] == "user" and isinstance(message["content"], list):
            for content in message["content"]:
                if content["type"] == "image":
                    image_inputs.append(content["image"])
    return image_inputs

# 自定义数据集类
class ShuiDataset(Dataset):
    def __init__(self, dataset_path, splits, split_name, processor, class_names, image_size=128):
        self.dataset_path = dataset_path
        self.splits = splits[split_name]
        self.processor = processor
        self.class_names = class_names
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.data = []
        for cls in self.splits:
            for img_name in self.splits[cls]:
                self.data.append((os.path.join(dataset_path, cls, img_name), cls))
   
    def __len__(self):
        return len(self.data)
   
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return {"image": image, "label": int(label)}
        except Exception as e:
            logger.error(f"无法加载图像 {img_path}: {e}")
            return None

# 数据集划分
def split_dataset(dataset_path, train_size=250, test_size=150, val_size=40):
    classes = [str(i) for i in range(12)]
    splits = {"test": {}}
   
    for cls in classes:
        class_path = os.path.join(dataset_path, cls)
        if not os.path.exists(class_path):
            raise FileNotFoundError(f"类别文件夹 {class_path} 不存在")
       
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(images) < train_size + test_size + val_size:
            logger.error(f"类别 {cls} 图片数量不足，实际: {len(images)}，需要: {train_size + test_size + val_size}")
            raise ValueError(f"类别 {cls} 图片数量不足")
       
        splits["test"][cls] = images[train_size:train_size + test_size]
        logger.info(f"类别 {cls}: 测试 {len(splits['test'][cls])}")
       
    return splits

# 加载模型和处理器
def load_model_and_processor(model_path):
    logger.info(f"检查模型目录: {os.listdir(model_path)}")
   
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map={"": "cuda:0"},
            torch_dtype=torch.bfloat16
        )
        model = torch.compile(model)
        model.eval()
        logger.info("模型已加载（bfloat16）")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise
   
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )
        logger.info(f"处理器加载成功，配置: {processor.__dict__}")
    except Exception as e:
        logger.error(f"处理器加载失败: {e}")
        raise
   
    config_path = os.path.join(model_path, "preprocessor_config.json")
    image_size = 128
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            image_size = config.get("image_size", 128)
            logger.info(f"从配置文件加载图像分辨率: {image_size}")
   
    return model, processor, image_size

# 分类函数
def classify_image(model, processor, image_path, class_names, image_size=128, max_retries=3):
    logger.info(f"开始分类图片: {image_path}")
   
    for attempt in range(max_retries):
        try:
            image = Image.open(image_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)
           
            prompt = f"请返回水族文字的类别编号（0-11）。"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
           
            text = [processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True) for msg in messages]
            image_inputs = process_vision_info(messages)
           
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=None,
                padding=True,
                return_tensors="pt"
            ).to(next(model.parameters()).device)
           
            logger.info(f"输入形状: {inputs.input_ids.shape}, 图像输入: {len(image_inputs)}")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :len(class_names)]
                logger.info(f"输出logits形状: {logits.shape}, expected: [1, {len(class_names)}]")
                if logits.shape != (1, len(class_names)):
                    logger.error(f"分类logits形状错误: {logits.shape}")
                    return None
                pred_idx = torch.argmax(logits, dim=-1).item()
                result = str(pred_idx)
           
            if result not in class_names:
                logger.warning(f"预测结果 {result} 无效，使用默认类别")
                result = class_names[0]
           
            logger.info(f"图片 {image_path} 分类完成，预测类别: {result}")
            return result
           
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"尝试 {attempt + 1}/{max_retries} 显存不足: {e}")
            torch.cuda.empty_cache()
            if attempt == max_retries - 1:
                return None
        except Exception as e:
            logger.error(f"尝试 {attempt + 1}/{max_retries} 失败: {e}")
            if attempt == max_retries - 1:
                return None
        finally:
            if 'inputs' in locals():
                del inputs
            if 'outputs' in locals():
                del outputs
            if 'logits' in locals():
                del logits
            torch.cuda.empty_cache()
   
    return None

# 评估函数
def evaluate_model(model, processor, dataset_path, splits, split_name="test", image_size=128):
    model.eval()
    class_names = [str(i) for i in range(12)]
    predictions = []
    true_labels = []
   
    with torch.no_grad():
        for cls in class_names:
            for img_name in splits[split_name][cls]:
                img_path = os.path.join(dataset_path, cls, img_name)
                pred = classify_image(model, processor, img_path, class_names, image_size)
                if pred in class_names:
                    predictions.append(pred)
                    true_labels.append(cls)
                    logger.info(f"Class {cls}, Image {img_name}, Predicted: {pred}, True: {cls}")
                else:
                    logger.warning(f"图片 {img_path} 预测结果 {pred} 不在类别列表中")
   
    if predictions:
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, zero_division=0, output_dict=True)
        macro_recall = report['macro avg']['recall']
        macro_f1 = report['macro avg']['f1-score']
        
        logger.info(f"{split_name}集准确率: {accuracy:.4f}")
        logger.info(f"{split_name}集宏平均召回率: {macro_recall:.4f}")
        logger.info(f"{split_name}集宏平均F1分数: {macro_f1:.4f}")
        logger.info(f"分类报告:\n{classification_report(true_labels, predictions, zero_division=0)}")
        
        return accuracy, macro_recall, macro_f1
    else:
        logger.error("无有效预测结果")
        return 0.0, 0.0, 0.0

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
   
    try:
        splits = split_dataset(dataset_path)
    except Exception as e:
        logger.error(f"数据集划分失败: {e}")
        return
   
    try:
        model, processor, image_size = load_model_and_processor(model_path)
    except Exception as e:
        logger.error(f"模型或处理器加载失败: {e}")
        return
   
    class_names = [str(i) for i in range(12)]
    test_dataset = ShuiDataset(dataset_path, splits, "test", processor, class_names, image_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                           collate_fn=lambda x: [item for item in x if item is not None])
   
    accelerator = Accelerator(mixed_precision="bf16")
    model = accelerator.prepare(model)
   
    try:
        test_accuracy, test_recall, test_f1 = evaluate_model(model, processor, dataset_path, splits,
                                                            split_name="test", image_size=image_size)
        logger.info(f"测试集最终结果 - 准确率: {test_accuracy:.4f}, 召回率: {test_recall:.4f}, F1分数: {test_f1:.4f}")
    except Exception as e:
        logger.error(f"测试失败: {e}")
        logger.info(f"显存分配详情:\n{torch.cuda.memory_summary()}")
   
    del model, processor
    torch.cuda.empty_cache()
    logger.info("显存已清理")

if __name__ == "__main__":
    main()
