import os
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import random
from sklearn.metrics import accuracy_score, classification_report
import logging
import time
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import json
from accelerate import Accelerator
from torch.profiler import profile, record_function, ProfilerActivity
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("full_fine_tune_tibetan_digit_log.txt"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 模型和数据集路径
model_path = "../RolmOCR"
dataset_path = "../datasets/TibetanMNIST"
output_dir = "./TibetanDigit_fully_tuned"

# 处理视觉信息
def process_vision_info(messages):
    image_inputs = []
    video_inputs = []
    for message in messages:
        if message["role"] == "user" and isinstance(message["content"], list):
            for content in message["content"]:
                if content["type"] == "image":
                    image_inputs.append(content["image"])
                elif content["type"] == "video":
                    video_inputs.append(content["video"])
    return image_inputs, video_inputs

# 检查数据集完整性
def check_dataset_integrity(dataset_path, expected_num_images=440):
    classes = [str(i) for i in range(10)]
    for cls in classes:
        class_path = os.path.join(dataset_path, cls)
        if not os.path.exists(class_path):
            raise FileNotFoundError(f"类别文件夹 {class_path} 不存在")
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        logger.info(f"类别 {cls} 包含 {len(images)} 张图片")
        if len(images) == 0:
            raise ValueError(f"类别 {cls} 文件夹为空")
        elif len(images) != expected_num_images:
            logger.warning(f"类别 {cls} 图片数量为 {len(images)}，预期为 {expected_num_images}")
        for img in images:
            try:
                Image.open(os.path.join(class_path, img)).convert("RGB")
            except Exception as e:
                logger.error(f"图像 {img} 加载失败: {e}")

# 自定义数据集类
class TibetanDigitDataset(Dataset):
    def __init__(self, dataset_path, splits, split_name, processor, class_names, image_size=128):
        self.dataset_path = dataset_path
        self.splits = splits[split_name]
        self.processor = processor
        self.class_names = class_names
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) if split_name == "train" else transforms.Compose([
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
    classes = [str(i) for i in range(10)]
    splits = {"train": {}, "test": {}, "val": {}}

    for cls in classes:
        class_path = os.path.join(dataset_path, cls)
        if not os.path.exists(class_path):
            raise FileNotFoundError(f"类别文件夹 {class_path} 不存在")

        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(images) < train_size + test_size + val_size:
            logger.error(f"类别 {cls} 图片数量不足，实际: {len(images)}，需要: {train_size + test_size + val_size}")
            raise ValueError(f"类别 {cls} 图片数量不足")

        random.seed(42)
        random.shuffle(images)

        splits["train"][cls] = images[:train_size]
        splits["test"][cls] = images[train_size:train_size + test_size]
        splits["val"][cls] = images[train_size + test_size:train_size + test_size + val_size]

        logger.info(f"类别 {cls}: 训练 {len(splits['train'][cls])}, 测试 {len(splits['test'][cls])}, 验证 {len(splits['val'][cls])}")

    return splits

# 加载模型和处理器（全微调版本）
def load_model_and_processor(model_path):
    logger.info(f"检查模型目录: {os.listdir(model_path)}")

    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map={"": "cuda:0"},
            torch_dtype=torch.bfloat16
        )
        model.gradient_checkpointing_enable()

        for param in model.parameters():
            param.requires_grad = True

        logger.info("模型已加载（全微调模式，所有参数可训练）")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"可训练参数总数: {total_params:,}")

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

# 检查梯度有效性
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                logger.warning(f"发现无效梯度（NaN或inf）在参数 {name}")
                return False
    return True

# 训练函数（全微调版本）
def train_model(model, processor, train_loader, val_loader, splits, class_names, num_epochs=3, learning_rate=2e-6, accum_steps=4):
    accelerator = Accelerator(mixed_precision="bf16")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    class_weights = torch.tensor([1.0] * 10).to(accelerator.device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    best_val_accuracy = 0
    patience = 2
    patience_counter = 0
    training_successful = False
    valid_batch_count = 0
    smoothed_loss = 0
    alpha = 0.2

    def get_batch_samples(dataset, class_names):
        samples = []
        labels = []
        random.seed(time.time())
        for cls in class_names:
            cls_indices = [i for i, (path, label) in enumerate(dataset.data) if label == cls]
            if cls_indices:
                idx = random.choice(cls_indices)
                sample = dataset[idx]
                if sample is not None:
                    samples.append(sample["image"])
                    labels.append(sample["label"])
        return samples, torch.tensor(labels, dtype=torch.long).to(accelerator.device)

    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0
        batch_count = 0
        valid_batch_count = 0
        optimizer.zero_grad()

        model.train()
        total_batches = len(train_loader)
        for batch_idx in range(total_batches):
            try:
                images, labels = get_batch_samples(train_loader.dataset, class_names)
                if len(images) != 10 or len(labels) != 10:
                    logger.warning(f"批次 {batch_idx} 样本数量不足: {len(images)}，跳过")
                    continue

                prompt = f"""你是藏语数字识别专家，需要识别图片中展示的藏语数字（对应阿拉伯数字0-9）。
注意事项：
1. 藏语数字与阿拉伯数字的对应关系为：
   ༠ → 0，༡ → 1，༢ → 2，༣ → 3，༤ → 4，
   ༥ → 5，༦ → 6，༧ → 7，༨ → 8，༩ → 9
2. 请仔细分析图片中的藏语数字字符，忽略背景和干扰元素
3. 仅返回对应的阿拉伯数字（0-9中的一个），不添加任何额外文字、解释或标点符号"""
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt}
                        ]
                    } for img in images
                ]

                loss = None
                try:
                    if batch_idx == 0 and epoch == 0:
                        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                            with record_function("model_inference"):
                                image_inputs, video_inputs = process_vision_info(messages)
                                if not image_inputs or len(image_inputs) != 10:
                                    logger.error(f"批次 {batch_idx} 无有效图像输入或数量不正确: {len(image_inputs)}")
                                    continue

                                text = [processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True) for msg in messages]
                                inputs = processor(
                                    text=text,
                                    images=image_inputs,
                                    videos=None,
                                    padding=True,
                                    return_tensors="pt"
                                ).to(accelerator.device)

                                logger.info(f"输入形状: {inputs.input_ids.shape}, 图像输入: {len(image_inputs)}, 标签: {labels.tolist()}")
                                outputs = model(**inputs)
                                logits = outputs.logits[:, -1, :len(class_names)]
                                logger.info(f"Batch {batch_idx}, logits shape: {logits.shape}, expected: [10, {len(class_names)}], logits range: {logits.min().item():.4f} to {logits.max().item():.4f}")
                                if logits.shape[0] != 10 or logits.shape[1] != len(class_names):
                                    logger.error(f"Invalid logits shape: {logits.shape}")
                                    continue
                                loss = loss_fn(logits, labels) / accum_steps
                        logger.info(prof.key_averages().table(sort_by="cuda_memory_usage"))
                        logger.info(f"显存分配详情:\n{torch.cuda.memory_summary()}")
                    else:
                        image_inputs, video_inputs = process_vision_info(messages)
                        if not image_inputs or len(image_inputs) != 10:
                            logger.error(f"批次 {batch_idx} 无有效图像输入或数量不正确: {len(image_inputs)}")
                            continue

                        text = [processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True) for msg in messages]
                        inputs = processor(
                            text=text,
                            images=image_inputs,
                            videos=None,
                            padding=True,
                            return_tensors="pt"
                        ).to(accelerator.device)

                        logger.info(f"输入形状: {inputs.input_ids.shape}, 图像输入: {len(image_inputs)}, 标签: {labels.tolist()}")
                        outputs = model(**inputs)
                        logits = outputs.logits[:, -1, :len(class_names)]
                        logger.info(f"Batch {batch_idx}, logits shape: {logits.shape}, expected: [10, {len(class_names)}], logits range: {logits.min().item():.4f} to {logits.max().item():.4f}")
                        if logits.shape[0] != 10 or logits.shape[1] != len(class_names):
                            logger.error(f"Invalid logits shape: {logits.shape}")
                            continue
                        loss = loss_fn(logits, labels) / accum_steps

                    if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                        loss.backward()
                        valid_batch_count += 1

                        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

                        if not check_gradients(model):
                            logger.warning(f"批次 {batch_idx} 梯度无效，清零梯度")
                            optimizer.zero_grad()
                            valid_batch_count = 0
                            continue

                        if batch_count == 0:
                            smoothed_loss = loss.item() * accum_steps
                        else:
                            smoothed_loss = (1 - alpha) * smoothed_loss + alpha * (loss.item() * accum_steps)

                        total_loss += loss.item() * accum_steps
                        batch_count += 1
                        training_successful = True

                        if valid_batch_count >= accum_steps or batch_idx + 1 == total_batches:
                            optimizer.step()
                            optimizer.zero_grad()
                            valid_batch_count = 0
                            logger.info(f"优化器更新，当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

                        logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{total_batches}, 损失: {loss.item() * accum_steps:.4f}, 平滑损失: {smoothed_loss:.4f}, 显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB, 最大显存: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")

                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"批次 {batch_idx} 显存不足: {e}")
                    logger.info(f"显存分配详情:\n{torch.cuda.memory_summary()}")
                    logger.error("全微调显存需求较高，尝试减小批次大小或降低图像分辨率")
                    return False
                except Exception as e:
                    logger.error(f"训练批次 {batch_idx} 失败: {e}")
                    continue
                finally:
                    if 'inputs' in locals():
                        del inputs
                    if 'outputs' in locals():
                        del outputs
                    if 'logits' in locals():
                        del logits
                    if 'loss' in locals() and loss is not None:
                        del loss
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

            except Exception as e:
                logger.error(f"批次 {batch_idx} 采样失败: {e}")
                continue

        if batch_count > 0:
            avg_loss = total_loss / batch_count
            logger.info(f"Epoch {epoch+1}/{num_epochs}, 平均损失: {avg_loss:.4f}, 耗时: {time.time() - start_time:.2f}秒")
        else:
            logger.error("所有批次均失败")
            return False

        val_accuracy = evaluate_model(accelerator.unwrap_model(model), processor, dataset_path, splits, split_name="val")
        logger.info(f"验证集准确率: {val_accuracy:.4f}")

        scheduler.step(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            os.makedirs(output_dir, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            logger.info(f"保存最佳全微调模型，验证准确率: {best_val_accuracy:.4f}")
            logger.info(f"检查点保存到: {os.listdir(output_dir)}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"早停：验证准确率未提升 {patience} 个epoch")
                break

    return training_successful

# 分类函数
def classify_image(model, processor, image_path, class_names, image_size=128, max_retries=3):
    logger.info(f"开始分类图片: {image_path}")
    start_time = time.time()

    for attempt in range(max_retries):
        try:
            image = Image.open(image_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)

            prompt = f"""你是藏语数字识别专家，需要识别图片中展示的藏语数字（对应阿拉伯数字0-9）。
注意事项：
1. 藏语数字与阿拉伯数字的对应关系为：
   ༠ → 0，༡ → 1，༢ → 2，༣ → 3，༤ → 4，
   ༥ → 5，༦ → 6，༧ → 7，༨ → 8，༩ → 9
2. 请仔细分析图片中的藏语数字字符，忽略背景和干扰元素
3. 仅返回对应的阿拉伯数字（0-9中的一个），不添加任何额外文字、解释或标点符号"""
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
            image_inputs, video_inputs = process_vision_info(messages)

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

            logger.info(f"图片 {image_path} 分类完成，原始 logits: {logits.tolist()}, 预测类别: {result}, 耗时: {time.time() - start_time:.3f}秒")
            return result

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"尝试 {attempt + 1}/{max_retries} 显存不足: {e}")
            logger.info(f"显存分配详情:\n{torch.cuda.memory_summary()}")
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
    class_names = [str(i) for i in range(10)]
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
        logger.info(f"{split_name}集准确率: {accuracy:.4f}")
        logger.info(f"分类报告:\n{classification_report(true_labels, predictions, zero_division=0)}")
        return accuracy
    else:
        logger.error("无有效预测结果")
        return 0.0

# 主函数
def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
        model, processor, image_size = load_model_and_processor(model_path)
    except Exception as e:
        logger.error(f"模型或处理器加载失败: {e}")
        return

    class_names = [str(i) for i in range(10)]
    train_dataset = TibetanDigitDataset(dataset_path, splits, "train", processor, class_names, image_size)
    val_dataset = TibetanDigitDataset(dataset_path, splits, "val", processor, class_names, image_size)
    test_dataset = TibetanDigitDataset(dataset_path, splits, "test", processor, class_names, image_size)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda x: [item for item in x if item is not None])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: [item for item in x if item is not None])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: [item for item in x if item is not None])

    training_successful = False
    try:
        training_successful = train_model(model, processor, train_loader, val_loader, splits, class_names, num_epochs=5, learning_rate=2e-6, accum_steps=4)
    except Exception as e:
        logger.error(f"训练失败: {e}")
        training_successful = False

    if training_successful:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            logger.info(f"加载最佳全微调模型从 {output_dir}")
            test_model = AutoModelForVision2Seq.from_pretrained(
                output_dir,
                trust_remote_code=True,
                device_map={"": "cuda:0"},
                torch_dtype=torch.bfloat16,
                assign=True
            )
            test_model = torch.compile(test_model)
            test_model.eval()
            logger.info(f"测试模型显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
            test_accuracy = evaluate_model(test_model, processor, dataset_path, splits, split_name="test", image_size=image_size)
            logger.info(f"测试集最终准确率: {test_accuracy:.4f}")
        except Exception as e:
            logger.error(f"测试失败: {e}")
            logger.info(f"显存分配详情:\n{torch.cuda.memory_summary()}")

    del model, processor
    if 'test_model' in locals():
        del test_model
    torch.cuda.empty_cache()
    logger.info("显存已清理")

if __name__ == "__main__":
    main()
