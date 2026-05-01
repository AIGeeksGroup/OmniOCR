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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import torch.nn.functional as F
from torch.cuda.amp import autocast

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("codyra_fine_tune_rolmocr_log.txt"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 模型和数据集路径
model_path = "../RolmOCR"
dataset_path = "../datasets/TibetanMNIST"
output_dir = "./RolmOCR_codyra_zang"

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
def check_dataset_integrity(dataset_path):
    classes = [str(i) for i in range(10)]  # 类别为0-9
    for cls in classes:
        class_path = os.path.join(dataset_path, cls)
        if not os.path.exists(class_path):
            raise FileNotFoundError(f"类别文件夹 {class_path} 不存在")
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        logger.info(f"类别 {cls} 包含 {len(images)} 张图片")
        if len(images) == 0:
            raise ValueError(f"类别 {cls} 文件夹为空")
        if len(images) < 3:
            raise ValueError(f"类别 {cls} 图片数量不足，至少需要3张，实际: {len(images)}")
        for img in images:
            try:
                Image.open(os.path.join(class_path, img)).convert("RGB")
            except Exception as e:
                logger.error(f"图像 {img} 加载失败: {e}")

# 自定义数据集类
class ShuiDataset(Dataset):
    def __init__(self, dataset_path, splits, split_name, processor, class_names, image_size=48):
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
def split_dataset(dataset_path):
    classes = [str(i) for i in range(10)]  # 类别为0-9
    splits = {"train": {}, "test": {}, "val": {}}
    for cls in classes:
        class_path = os.path.join(dataset_path, cls)
        if not os.path.exists(class_path):
            raise FileNotFoundError(f"类别文件夹 {class_path} 不存在")
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(images)
        if total_images < 3:
            logger.error(f"类别 {cls} 图片数量不足，实际: {total_images}，需要至少3张")
            raise ValueError(f"类别 {cls} 图片数量不足")
        train_size = math.floor(total_images * 0.6)  # 60% 训练集
        test_size = math.floor(total_images * 0.3)  # 30% 测试集
        val_size = total_images - train_size - test_size  # 剩余的为验证集（约10%）
        if train_size < 1 or test_size < 1 or val_size < 1:
            logger.error(f"类别 {cls} 图片数量 {total_images} 不足以分配到所有分割")
            raise ValueError(f"类别 {cls} 图片数量不足以分配")
        random.seed(42)
        random.shuffle(images)
        splits["train"][cls] = images[:train_size]
        splits["test"][cls] = images[train_size:train_size + test_size]
        splits["val"][cls] = images[train_size + test_size:train_size + test_size + val_size]
        logger.info(f"类别 {cls}: 训练 {len(splits['train'][cls])}, 测试 {len(splits['test'][cls])}, 验证 {len(splits['val'][cls])}")
    return splits

# CoDyRA 模块（动态秩选择 + 可合并）
class CoDyRAModule(nn.Module):
    def __init__(self, in_features, out_features, initial_rank=8, lora_alpha=16, lora_dropout=0.0):
        super(CoDyRAModule, self).__init__()
        self.initial_rank = initial_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.lora_A = nn.Parameter(torch.randn(out_features, initial_rank) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(initial_rank, in_features) * 0.01)
        self.rank_weights = nn.Parameter(torch.ones(initial_rank))
        self.scaling = lora_alpha / initial_rank
        self.register_buffer("pretrained_weight", torch.zeros(out_features, in_features))

    def forward(self, x):
        rank_weights = F.softshrink(self.rank_weights, lambd=0.005)
        rank_mask = (rank_weights != 0).float()
        lora_delta = (self.lora_A @ (self.lora_B * rank_mask.unsqueeze(1))) * self.scaling
        lora_delta = self.lora_dropout(lora_delta)
        return x @ (self.pretrained_weight + lora_delta).T

    def merge_weights(self):
        with torch.no_grad():
            rank_weights = F.softshrink(self.rank_weights, lambd=0.005)
            rank_mask = (rank_weights != 0).float()
            lora_delta = (self.lora_A @ (self.lora_B * rank_mask.unsqueeze(1))) * self.scaling
            self.pretrained_weight.data.copy_(self.pretrained_weight + lora_delta)

# 替换模型中的线性层以支持 CoDyRA
def apply_codyra(model, target_modules, initial_rank=8, lora_alpha=16, lora_dropout=0.0):
    for name, module in model.named_modules():
        if any(t in name for t in target_modules) and isinstance(module, nn.Linear):
            codyra_module = CoDyRAModule(
                in_features=module.in_features,
                out_features=module.out_features,
                initial_rank=initial_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
            codyra_module.pretrained_weight.copy_(module.weight)
            parent_name = name.rsplit(".", 1)[0]
            parent_module = model.get_submodule(parent_name)
            module_name = name.split(".")[-1]
            setattr(parent_module, module_name, codyra_module)
            logger.info(f"替换 {name} 为 CoDyRA 模块")
    return model

# 加载模型和处理器
def load_model_and_processor(model_path):
    logger.info(f"检查模型目录: {os.listdir(model_path)}")
    with torch.no_grad():
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map={"": "cuda:0"},
                torch_dtype=torch.bfloat16
            )
            logger.info("模型已加载（bfloat16）")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "mlp.fc1", "mlp.fc2"]
    model = apply_codyra(model, target_modules, initial_rank=8, lora_alpha=16, lora_dropout=0.0)
    logger.info("CoDyRA 适配器已应用，目标模块: {}".format(target_modules))
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
    image_size = 48
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            image_size = config.get("image_size", 48)
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

# 训练函数
def train_model(model, processor, train_loader, val_loader, splits, class_names, num_epochs=1, learning_rate=5e-6, accum_steps=2):
    accelerator = Accelerator(mixed_precision="bf16")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
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

    # 手动采样1个样本（随机选择1个类别）
    def get_batch_samples(dataset, class_names, batch_size=1):
        samples = []
        labels = []
        random.seed(time.time())
        selected_classes = random.sample(class_names, min(len(class_names), batch_size))
        for cls in selected_classes:
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
                images, labels = get_batch_samples(train_loader.dataset, class_names, batch_size=1)
                if len(images) < 1 or len(labels) < 1:
                    logger.warning(f"批次 {batch_idx} 样本数量不足: {len(images)}，跳过")
                    continue
                prompt = "返回藏语数字类别（0-9）"  # 简化 prompt
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
                    with autocast():
                        image_inputs, video_inputs = process_vision_info(messages)
                        if not image_inputs or len(image_inputs) < 1:
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
                        logger.info(f"Batch {batch_idx}, logits shape: {logits.shape}, expected: [{len(image_inputs)}, {len(class_names)}], logits range: {logits.min().item():.4f} to {logits.max().item():.4f}")
                        if logits.shape[0] != len(image_inputs) or logits.shape[1] != len(class_names):
                            logger.error(f"Invalid logits shape: {logits.shape}")
                            continue
                        loss = loss_fn(logits, labels) / accum_steps
                    if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                        accelerator.backward(loss)
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
                    torch.cuda.synchronize()
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"批次 {batch_idx} 显存不足: {e}")
                    logger.info(f"显存分配详情:\n{torch.cuda.memory_summary()}")
                    torch.cuda.empty_cache()
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
                    torch.cuda.synchronize()
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
            for name, module in model.named_modules():
                if isinstance(module, CoDyRAModule):
                    module.merge_weights()
                    logger.info(f"合并 CoDyRA 更新于 {name}")
            accelerator.unwrap_model(model).save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            logger.info(f"保存最佳 CoDyRA 模型，验证准确率: {best_val_accuracy:.4f}")
            logger.info(f"检查点保存到: {os.listdir(output_dir)}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"早停：验证准确率未提升 {patience} 个epoch")
                break
    return training_successful

# 分类函数
def classify_image(model, processor, image_path, class_names, image_size=48, max_retries=3):
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
            prompt = "返回藏语数字类别（0-9）"  # 简化 prompt
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
            with torch.no_grad(), autocast():
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
            torch.cuda.synchronize()
    return None

# 评估函数
def evaluate_model(model, processor, dataset_path, splits, split_name="test", image_size=48):
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
    train_dataset = ShuiDataset(dataset_path, splits, "train", processor, class_names, image_size)
    val_dataset = ShuiDataset(dataset_path, splits, "val", processor, class_names, image_size)
    test_dataset = ShuiDataset(dataset_path, splits, "test", processor, class_names, image_size)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda x: [item for item in x if item is not None])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: [item for item in x if item is not None])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: [item for item in x if item is not None])
    training_successful = False
    try:
        training_successful = train_model(model, processor, train_loader, val_loader, splits, class_names, num_epochs=1, learning_rate=5e-6, accum_steps=2)
    except Exception as e:
        logger.error(f"训练失败: {e}")
        training_successful = False
    if training_successful:
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info(f"加载最佳 CoDyRA 模型从 {output_dir}")
            test_model = AutoModelForVision2Seq.from_pretrained(
                output_dir,
                trust_remote_code=True,
                device_map={"": "cuda:0"},
                torch_dtype=torch.bfloat16,
                assign=True
            )
            test_model.eval()
            logger.info(f"测试模型显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
            test_accuracy = evaluate_model(test_model, processor, dataset_path, splits, split_name="test", image_size=image_size)
            logger.info(f"测试集最终准确率: {test_accuracy:.4f}")
        except Exception as e:
            logger.error(f"测试失败: {e}")
            logger.info(f"显存分配详情:\n{torch.cuda.memory_summary()}")
            training_successful = False
    del model, processor
    if 'test_model' in locals():
        del test_model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    logger.info("显存已清理")

if __name__ == "__main__":
    main()
