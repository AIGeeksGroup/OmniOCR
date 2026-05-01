
import os
import base64
import random
import math
import time
import logging
from sklearn.metrics import accuracy_score
from PIL import Image
import backoff
import json
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("deepseek_model_test_log.txt"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 模型API配置
DEEPSEEK_API_KEY = "************************************"  # 请确保这是有效的API密钥
DEEPSEEK_BASE_URL = "***********"
MODEL_NAME = "***************"

# 数据集配置
dataset_path = "../datasets/TibetanMNIST"
DATASET_SCALE_FACTOR = 1/20

# 初始化OpenAI客户端
def get_openai_client():
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "请在此处输入你的API密钥":
        logger.error("请先设置有效的API密钥")
        raise ValueError("未设置有效的API密钥")
    
    client = OpenAI(
        base_url=DEEPSEEK_BASE_URL,
        api_key=DEEPSEEK_API_KEY
    )
    # 验证模型支持图像输入
    try:
        models = client.models.list()
        logger.info(f"可用模型: {[model.id for model in models.data]}")
        if MODEL_NAME not in [model.id for model in models.data]:
            logger.warning(f"模型 {MODEL_NAME} 可能不受支持，建议检查API文档")
    except Exception as e:
        logger.error(f"无法获取模型列表: {str(e)}")
    return client

# 检查数据集完整性
def check_dataset_integrity(dataset_path):
    classes = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f)) and f != '.ipynb_checkpoints']
    if not classes:
        logger.error("数据集目录中没有找到任何有效类别文件夹")
        raise FileNotFoundError("数据集目录中没有有效类别文件夹")
    
    for cls in classes:
        class_path = os.path.join(dataset_path, cls)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        scaled_size = max(1, math.floor(len(images) * DATASET_SCALE_FACTOR))
        logger.info(f"类别 {cls} 原始图片数: {len(images)}, 缩减后保留: {scaled_size}")
        if len(images) == 0:
            logger.error(f"类别 {cls} 文件夹为空")
        elif scaled_size < 1:
            logger.warning(f"类别 {cls} 图片数量过少，缩减后不足1张，将强制保留1张")

# 数据集划分
def split_dataset(dataset_path):
    classes = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f)) and f != '.ipynb_checkpoints']
    if not classes:
        raise FileNotFoundError("数据集目录中没有有效类别文件夹")
    
    splits = {"train": {}, "test": {}, "val": {}}
    
    for cls in classes:
        class_path = os.path.join(dataset_path, cls)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(images)
        
        scaled_total = max(3, math.floor(total_images * DATASET_SCALE_FACTOR))
        if scaled_total > total_images:
            scaled_total = total_images
        random.seed(42)
        scaled_images = random.sample(images, scaled_total)
        
        train_size = math.floor(scaled_total * 0.6)
        test_size = math.floor(scaled_total * 0.3)
        val_size = scaled_total - train_size - test_size
        
        if train_size < 1:
            train_size = 1
            test_size = max(1, math.floor((scaled_total - train_size) * 0.5))
            val_size = scaled_total - train_size - test_size
        if test_size < 1:
            test_size = 1
            val_size = scaled_total - train_size - test_size
        if val_size < 1:
            val_size = 1
            test_size = scaled_total - train_size - val_size
        
        random.shuffle(scaled_images)
        
        splits["train"][cls] = scaled_images[:train_size]
        splits["test"][cls] = scaled_images[train_size:train_size + test_size]
        splits["val"][cls] = scaled_images[train_size + test_size:]
        
        logger.info(f"类别 {cls} 划分结果 - 训练: {len(splits['train'][cls])}, 测试: {len(splits['test'][cls])}, 验证: {len(splits['val'][cls])}")
    
    return splits

# 图片转base64
def image_to_base64(image_path, max_size=(256, 256)):
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img.thumbnail(max_size)
            
            import io
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=80)  # 提高图像质量
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            logger.debug(f"图片 {image_path} Base64 长度: {len(base64_str)}")
            return base64_str
    except Exception as e:
        logger.error(f"图片转换失败 {image_path}: {e}")
        return None

# 重试条件
def is_retryable_exception(e):
    if isinstance(e, (APIConnectionError, RateLimitError)):
        return True
    if isinstance(e, APIError):
        return e.status_code == 422 or (500 <= e.status_code < 600)
    return False

# 调用DeepSeek API进行分类
@backoff.on_exception(
    backoff.expo,
    (APIError, APIConnectionError, RateLimitError),
    max_tries=3,
    base=2,
    factor=1,
    giveup=lambda e: not is_retryable_exception(e)
)
def classify_image(client, image_path, class_names):
    logger.info(f"开始分类图片: {image_path}")
    start_time = time.time()
    
    base64_image = image_to_base64(image_path)
    if not base64_image:
        logger.error(f"图片 {image_path} 无法转换为base64")
        return None
    
    messages = [
        {
            "role": "system",
            "content": """You are an expert in recognizing Tibetan digits (0-9). The Tibetan digits correspond to Arabic digits as follows:
- ༠ → 0
- ༡ → 1
- ༢ → 2
- ༣ → 3
- ༤ → 4
- ༥ → 5
- ༦ → 6
- ༧ → 7
- ༨ → 8
- ༩ → 9
Return only the corresponding Arabic digit (0-9) for the given image, with no additional text, explanation, or punctuation."""
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Identify the Tibetan digit in the image and return only the Arabic digit (0-9)."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,  # 增加少量随机性以提高鲁棒性
            timeout=30
        )
        
        logger.debug(f"API完整响应: {response.to_dict()}")
        
        if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
            logger.error(f"API返回无效响应结构: {str(response)}")
            return None
        
        content = response.choices[0].message.content.strip()
        
        if not content:
            logger.error(f"API返回空响应: {image_path}")
            return None
        
        if content in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            pred = content
        else:
            logger.warning(f"响应 '{content}' 不是有效的数字: {image_path}")
            return None
    
    except Exception as e:
        logger.error(f"API请求失败 {image_path}: {str(e)}, 类型: {type(e).__name__}")
        try:
            if isinstance(e, str) or (hasattr(e, 'message') and isinstance(e.message, str)):
                error_msg = e if isinstance(e, str) else e.message
                logger.error(f"API错误消息: {error_msg}")
                try:
                    error_json = json.loads(error_msg)
                    logger.error(f"解析的错误信息: {error_json}")
                except:
                    pass
        except:
            pass
        return None
    
    end_time = time.time()
    inference_time = end_time - start_time
    logger.info(f"图片 {image_path} 分类完成，预测数字: {pred}, 耗时: {inference_time:.3f}秒")
    
    return pred

# 评估函数
def evaluate_model(dataset_path, splits, split_name="test"):
    class_names = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f)) and f != '.ipynb_checkpoints']
    predictions = []
    true_labels = []
    
    try:
        client = get_openai_client()
    except ValueError:
        return 0.0
    
    total_samples = sum(len(images) for images in splits[split_name].values())
    logger.info(f"开始评估 {split_name} 集，缩减后总样本数: {total_samples}")
    
    for cls in class_names:
        for img_name in splits[split_name][cls]:
            img_path = os.path.join(dataset_path, cls, img_name)
            try:
                pred = classify_image(client, img_path, class_names)
            except Exception as e:
                logger.error(f"处理图片 {img_path} 最终失败: {str(e)}")
                pred = None
            
            if pred in class_names:
                predictions.append(pred)
                true_labels.append(cls)
            else:
                logger.warning(f"跳过无效预测 {img_path}: 预测结果 {pred}")
    
    if predictions:
        accuracy = accuracy_score(true_labels, predictions)
        logger.info(f"{split_name}集准确率: {accuracy:.4f} (基于 {len(predictions)} 个样本)")
        return accuracy
    else:
        logger.error("无有效预测结果")
        return 0.0

# 主函数
def main():
    try:
        import backoff
    except ImportError:
        logger.info("安装backoff库...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "backoff"])
        import backoff
    
    try:
        check_dataset_integrity(dataset_path)
    except Exception as e:
        logger.error(f"数据集检查失败: {e}")
        return
    
    try:
        splits = split_dataset(dataset_path)
    except Exception as e:
        logger.error(f"数据集划分失败: {e}")
        return
    
    try:
        accuracy = evaluate_model(dataset_path, splits, split_name="test")
    except Exception as e:
        logger.error(f"评估失败: {e}")
    
    logger.info("测试完成")

if __name__ == "__main__":
    main()
