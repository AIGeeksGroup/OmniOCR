import os
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import random
from sklearn.metrics import accuracy_score
import logging
import time
import math
import re

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("lora_model_test_log.txt"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 模型和数据集路径
model_path = "./RolmOCR"
dataset_path = "./datasets/TibetanMNIST"

# 检查数据集目录
def check_dataset_integrity(dataset_path):
    classes = [str(i) for i in range(10)]  # 类别0-9
    for cls in classes:
        class_path = os.path.join(dataset_path, cls)
        if not os.path.exists(class_path):
            logger.error(f"类别文件夹 {class_path} 不存在")
            continue
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        logger.info(f"类别 {cls} 包含 {len(images)} 张图片")
        if len(images) == 0:
            logger.error(f"类别 {cls} 文件夹为空")
        elif len(images) < 3:  # 至少需要3张图片以支持分割
            logger.warning(f"类别 {cls} 图片数量不足，至少需要3张，实际: {len(images)}")

# 修复视频处理器配置文件
def fix_video_processor_config(path):
    preprocessor_old = os.path.join(path, "preprocessor.json")
    preprocessor_new = os.path.join(path, "video_preprocessor.json")
    if os.path.exists(preprocessor_old) and not os.path.exists(preprocessor_new):
        logger.info("迁移旧版配置文件到video_preprocessor.json...")
        try:
            os.rename(preprocessor_old, preprocessor_new)
            logger.info("配置文件迁移完成")
        except Exception as e:
            logger.error(f"配置文件迁移失败: {e}")
    elif os.path.exists(preprocessor_old):
        logger.warning("preprocessor.json存在但video_preprocessor.json已存在，跳过迁移")
    else:
        logger.info("无需迁移配置文件")

# 数据集划分
def split_dataset(dataset_path):
    classes = [str(i) for i in range(10)]  # 类别0-9
    splits = {"train": {}, "test": {}, "val": {}}
    
    for cls in classes:
        class_path = os.path.join(dataset_path, cls)
        if not os.path.exists(class_path):
            raise FileNotFoundError(f"类别文件夹 {class_path} 不存在")
        
        # 获取所有图片文件
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(images)
        if total_images < 3:
            logger.error(f"类别 {cls} 图片数量不足，实际: {total_images}，需要至少3张")
            raise ValueError(f"类别 {cls} 图片数量不足")
        
        # 计算分割大小
        train_size = math.floor(total_images * 0.6)  # 60% 训练集
        test_size = math.floor(total_images * 0.3)   # 30% 测试集
        val_size = total_images - train_size - test_size  # 剩余的为验证集（约10%）
        
        # 确保每个分割至少有1张图片
        if train_size < 1 or test_size < 1 or val_size < 1:
            logger.error(f"类别 {cls} 图片数量 {total_images} 不足以分配到所有分割")
            raise ValueError(f"类别 {cls} 图片数量不足以分配")
        
        # 随机打乱
        random.seed(42)  # 固定种子以确保可重复性
        random.shuffle(images)
        
        # 划分数据集
        splits["train"][cls] = images[:train_size]
        splits["test"][cls] = images[train_size:train_size + test_size]
        splits["val"][cls] = images[train_size + test_size:train_size + test_size + val_size]
        
        logger.info(f"类别 {cls}: 训练 {len(splits['train'][cls])}, 测试 {len(splits['test'][cls])}, 验证 {len(splits['val'][cls])}")
        
    return splits

# 加载模型和处理器
def load_model_and_processor(model_path):
    fix_video_processor_config(model_path)
    
    # 配置4位量化参数以减少显存占用
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 使用4位量化
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",  # 使用NF4量化
        bnb_4bit_use_double_quant=True  # 启用双重量化
    )
    
    # 加载模型
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config,
            offload_buffers=True,
            torch_dtype=torch.float16,  # 使用float16
            llm_int8_enable_fp32_cpu_offload=True  # 启用CPU卸载
        )
        logger.info("模型已使用4位量化加载")
    except Exception as e:
        logger.warning(f"4位量化加载失败，使用默认方式加载: {e}")
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            offload_buffers=True,
            torch_dtype=torch.float16
        )
    
    # 加载处理器
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )
        logger.info("处理器加载成功")
    except Exception as e:
        logger.error(f"处理器加载失败: {e}")
        raise
    
    return model, processor

# Zero-shot分类
def classify_image(model, processor, image_path, class_names):
    logger.info(f"开始分类图片: {image_path}")
    start_time = time.time()
    
    try:
        image = Image.open(image_path).convert("RGB").resize((512, 512))  # 降低分辨率
    except Exception as e:
        logger.error(f"无法加载图像 {image_path}: {e}")
        return None
    
    # 构建分类提示
    prompt = f"请仅返回藏语数字的类别编号（0-9），不要包含其他文字。"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_path}}
            ]
        }
    ]
    
    # 生成输入
    try:
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt_text], images=[image], return_tensors="pt").to(model.device)
    except Exception as e:
        logger.error(f"处理输入失败 {image_path}: {e}")
        return None
    
    # 推理
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)  # 减少max_new_tokens
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"推理时显存不足 {image_path}: {e}")
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        logger.error(f"推理失败 {image_path}: {e}")
        return None
    
    # 解码输出
    try:
        prompt_len = inputs["input_ids"].shape[1]
        text = processor.tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        result = text[0].strip()
        
        # 提取类别编号（0-9）
        match = re.search(r'\b([0-9])\b', result)
        if match:
            result = match.group(1)  # 提取匹配的数字
        else:
            logger.warning(f"无法从输出 '{result}' 中提取有效类别编号，使用默认类别")
            result = class_names[0]  # 默认类别
    except Exception as e:
        logger.error(f"解码输出失败 {image_path}: {e}")
        return None
    
    # 验证结果是否在类别列表中
    if result not in class_names:
        logger.warning(f"预测结果 {result} 无效，使用默认类别")
        result = class_names[0]
    
    # 清理显存
    del inputs, outputs
    torch.cuda.empty_cache()
    
    # 记录推理耗时和结果
    end_time = time.time()
    inference_time = end_time - start_time
    logger.info(f"图片 {image_path} 分类完成，预测类别: {result}, 耗时: {inference_time:.3f}秒")
    
    return result

# 评估函数
def evaluate_model(model, processor, dataset_path, splits, split_name="test"):
    class_names = [str(i) for i in range(10)]  # 类别0-9
    predictions = []
    true_labels = []
    
    for cls in class_names:
        for img_name in splits[split_name][cls]:
            img_path = os.path.join(dataset_path, cls, img_name)
            pred = classify_image(model, processor, img_path, class_names)
            if pred in class_names:
                predictions.append(pred)
                true_labels.append(cls)
            else:
                logger.warning(f"图片 {img_path} 预测结果 {pred} 不在类别列表中")
    
    # 计算准确率
    if predictions:
        accuracy = accuracy_score(true_labels, predictions)
        logger.info(f"{split_name}集准确率: {accuracy:.4f}")
        return accuracy
    else:
        logger.error("无有效预测结果")
        return 0.0

# 主函数
def main():
    # 设置PyTorch内存管理环境变量
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 检查数据集完整性
    try:
        check_dataset_integrity(dataset_path)
    except Exception as e:
        logger.error(f"数据集检查失败: {e}")
        return
    
    # 划分数据集
    try:
        splits = split_dataset(dataset_path)
    except Exception as e:
        logger.error(f"数据集划分失败: {e}")
        return
    
    # 加载模型和处理器
    try:
        model, processor = load_model_and_processor(model_path)
    except Exception as e:
        logger.error(f"模型或处理器加载失败: {e}")
        return
    
    # 评估模型（使用测试集）
    try:
        accuracy = evaluate_model(model, processor, dataset_path, splits, split_name="test")
    except Exception as e:
        logger.error(f"评估失败: {e}")
    
    # 清理显存
    del model, processor
    torch.cuda.empty_cache()
    logger.info("显存已清理")

if __name__ == "__main__":
    main()
