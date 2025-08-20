import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
from argparse import Namespace
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
class XRFWiFiDataset(Dataset):
    """WiFi数据集 (通道数270，时序长度1200)"""

    def __init__(self, is_train=True):
        self.root = './dataset/Raw_dataset/WiFi1/'
        self.is_train = is_train
        self.meta_file = 'dml_train.txt' if is_train else 'dml_val.txt'

        with open(f'./dataset/XRF_dataset/{self.meta_file}', 'r') as f:
            lines = f.readlines()
            self.files = [line.split(',')[0].strip() for line in lines]
            self.labels = [int(line.split(',')[2].strip()) - 1 for line in lines]

        # 计算数据集统计量
        if is_train and not os.path.exists('./wifi_stats.npz'):
            self._compute_stats()

        stats = np.load('./wifi_stats.npz')
        self.mean = stats['mean']
        self.std = stats['std']

    def _compute_stats(self):
        sum_ = np.zeros((270, 1), dtype=np.float32)
        sum_sq = np.zeros((270, 1), dtype=np.float32)
        count = 0

        for fn in self.files:
            data = np.load(os.path.join(self.root, f"{fn}.npy"))  # [270, 1000]
            sum_ += np.sum(data, axis=1, keepdims=True)
            sum_sq += np.sum(data ** 2, axis=1, keepdims=True)
            count += data.shape[1]

        mean = sum_ / count
        std = np.sqrt(sum_sq / count - mean ** 2)
        np.savez('./wifi_stats.npz', mean=mean, std=std)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.root, f"{self.files[idx]}.npy"))
        data = (data - self.mean) / (self.std + 1e-6)

        if self.is_train:  # 修复属性访问
            # 时序遮蔽
            if np.random.rand() < 0.3:
                mask_len = np.random.randint(30, 100)
                start = np.random.randint(0, 1200 - mask_len)
                data[:, start:start + mask_len] = 0

            # 通道遮蔽
            if np.random.rand() < 0.2:
                channels = np.random.choice(270, np.random.randint(5, 20), replace=False)
                data[channels] *= np.random.uniform(0.3, 0.7)
        if self.is_train and np.random.rand() < 0.3:
            alpha = np.random.uniform(0.8, 1.2)
            target_length = int(1200 * alpha)

            # 使用线性插值保持维度
            original_length = data.shape[1]
            x = np.linspace(0, 1, original_length)
            new_x = np.linspace(0, 1, target_length)
            data = np.apply_along_axis(
                lambda y: np.interp(new_x, x, y),
                axis=1,
                arr=data
            )

            # 截断或填充保持1000长度
            if target_length > 1200:
                data = data[:, :1200]
            else:
                pad_width = 1200 - target_length
                data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
            data = data[:, :1200] if data.shape[1] >= 1200 else np.pad(data, ((0, 0), (0, 1200 - data.shape[1])))
            assert data.shape == (270, 1200), f"插值后维度应为(270,1200)，实际为{data.shape}"
            # 通道相关性增强修正
        # 修改后的通道相关性增强代码
        if self.is_train and np.random.rand() < 0.2:
            cov_matrix = np.cov(data)
            reg_coeff = 1e-3  # 增大正则化系数
            cov_matrix += reg_coeff * np.eye(cov_matrix.shape[0])
            data = cov_matrix @ data
            # 添加数据标准化
            data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-6)

        return torch.FloatTensor(data).permute(1, 0), self.labels[idx]

class RFDataset(Dataset):
    """RFID数据集 (通道数23，时序长度148)"""

    def __init__(self, is_train=True):
        self.is_train = is_train
        self.root = './dataset/Raw_dataset/rfid ges/rfid ges/'

        self.meta_file = 'dml_train.txt' if is_train else 'dml_val.txt'

        with open(f'./dataset/XRF_dataset/{self.meta_file}', 'r') as f:
            lines = f.readlines()
            self.files = [line.split(',')[0].strip() for line in lines]
            self.labels = [int(line.split(',')[2].strip()) - 1 for line in lines]

        # 计算数据集统计量
        if is_train and not os.path.exists('./rfid_stats.npz'):
            self._compute_stats()

        stats = np.load('./rfid_stats.npz')
        self.mean = stats['mean']
        self.std = stats['std']

    def _compute_stats(self):
        sum_ = np.zeros((24, 1), dtype=np.float32)
        sum_sq = np.zeros((24, 1), dtype=np.float32)
        count = 0

        for fn in self.files:
            data = np.load(os.path.join(self.root, f"{fn}.npy"))  # [23, 148]
            sum_ += np.sum(data, axis=1, keepdims=True)
            sum_sq += np.sum(data ** 2, axis=1, keepdims=True)
            count += data.shape[1]

        mean = sum_ / count
        std = np.sqrt(sum_sq / count - mean ** 2)
        np.savez('./rfid_stats.npz', mean=mean, std=std)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.root, f"{self.files[idx]}.npy"))
        data = (data - self.mean) / (self.std + 1e-6)

        if self.is_train:  # 添加训练判断
            # 时序遮蔽
            if np.random.rand() < 0.3:
                mask_len = np.random.randint(10, 30)
                start = np.random.randint(0, 150 - mask_len)
                data[:, start:start + mask_len] = 0

            # 通道缩放
            if np.random.rand() < 0.2:
                channels = np.random.choice(24, 3, replace=False)
                data[channels] *= np.random.uniform(0.3, 0.7)

        return torch.FloatTensor(data).permute(1, 0), self.labels[idx]

# -------------------- 改进对比损失 --------------------
# -------------------- 修正后的对比损失 --------------------
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        """
        z_i, z_j: 形状为 [B, D] 的配对特征向量
        """
        batch_size = z_i.size(0)

        # 特征归一化
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # 拼接所有特征
        features = torch.cat([z_i, z_j], dim=0)  # [2B, D]

        # 计算相似度矩阵
        sim_matrix = torch.mm(features, features.T) / self.temperature  # [2B, 2B]

        # 构建正样本标签（对角线对称位置）
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0) # [2B]
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [2B, 2B]
        labels = labels.to(z_i.device)

        # 排除自身相似度
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z_i.device)
        labels = labels[~mask].view(2 * batch_size, -1)  # [2B, 2B-1]
        sim_matrix = sim_matrix[~mask].view(2 * batch_size, -1)  # [2B, 2B-1]

        # 选择正样本
        positives = sim_matrix[labels.bool()].view(2 * batch_size, -1)  # [2B, 1]

        # 计算交叉熵损失
        logits = torch.cat([positives, sim_matrix], dim=1)  # [2B, 2B]
        labels = torch.zeros(2 * batch_size, dtype=torch.long).to(z_i.device)

        return self.criterion(logits, labels)


# -------------------- 增强MoE结构 --------------------
class MoE(nn.Module):
    def __init__(self, wifi_dim=256, rfid_dim=256, num_experts=16, hidden_dim=1024, num_classes=20, topk=4):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk

        # 专家网络 (增加深度)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(wifi_dim + rfid_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3)
            ) for _ in range(num_experts)
        ])

        # 门控网络 (增加复杂度)
        self.gate = nn.Sequential(
            nn.Linear(wifi_dim + rfid_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, num_experts),
            nn.Softmax(dim=1)
        )

        # 最终分类适配
        self.final = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, wifi_feat, rfid_feat):
        combined = torch.cat([wifi_feat, rfid_feat], dim=1)

        # 门控机制 (Top-K选择)
        gate = self.gate(combined)
        topk_gate, topk_indices = gate.topk(self.topk, dim=1)
        topk_gate = topk_gate.softmax(dim=1)

        # 专家输出
        expert_outputs = torch.stack([e(combined) for e in self.experts], dim=1)  # [B, E, D]

        # 稀疏组合
        batch_indices = torch.arange(expert_outputs.size(0)).unsqueeze(1).expand(-1, self.topk)
        selected_experts = expert_outputs[batch_indices, topk_indices]  # [B, K, D]
        fused = torch.einsum('bkd,bk->bd', selected_experts, topk_gate)

        return self.final(fused)


# -------------------- 模型定义 --------------------
class AdvancedWiFiModel(nn.Module):
    """WiFi分支模型（保持原始结构）"""

    def __init__(self, args):
        super().__init__()
        # 输入处理
        self.embed = nn.Sequential(
            nn.Conv1d(270, 256, 7, padding=3),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # 特征提取
        self.stage1 = nn.Sequential(
            *[nn.Sequential(
                EfficientBlock(256),
                MultiScaleFusion(256)
            ) for _ in range(5)],
            nn.Conv1d(256, 512, 3, stride=2, padding=1)
        )

        # 深层特征
        self.stage2 = nn.Sequential(
            *[nn.Sequential(
                EfficientBlock(512),
                HybridAttention(512, reduction=8)
            ) for _ in range(6)],
            nn.AdaptiveAvgPool1d(1)
        )

        self.head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, args.num_classes)
        )
        self.proj = nn.Linear(512, 256)  # 特征投影

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B,270,1000]
        x = self.embed(x)
        x = self.stage1(x)  # [B,512,500]
        x = self.stage2(x)  # [B,512,1]
        feat = x.squeeze(-1)
        return self.head(feat), self.proj(feat)


class AdvancedRFIDModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 输入处理
        self.embed = nn.Sequential(
            nn.Conv1d(24, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # 特征提取
        self.stage1 = nn.Sequential(
            *[self._build_block(64) for _ in range(3)],
            nn.Conv1d(64, 128, 3, stride=2, padding=1)
        )

        # 深层特征
        self.stage2 = nn.Sequential(
            *[self._build_block(128) for _ in range(2)],
            HybridAttention(128),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, args.num_classes)
        )
        self.proj = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 256)
        )

    def _build_block(self, channels):
        return nn.Sequential(
            EfficientBlock(channels),
            HybridAttention(channels)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
        x = self.embed(x)        # [B,64,148]
        x = self.stage1(x)       # [B,128,74]
        x = self.stage2(x)       # [B,128,1]
        feat = x.squeeze(-1)     # [B,128]
        return self.classifier(feat), self.proj(feat)


class HybridAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // reduction, 1),
            nn.GELU(),
            nn.Conv1d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        self.temporal_att = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1),
            nn.GELU(),
            nn.Conv1d(in_channels // reduction, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_att(x)
        ta = self.temporal_att(x)
        return x * ca * ta + x

class EfficientBlock(nn.Module):
    def __init__(self, in_channels, expansion=4):
        super().__init__()
        hidden_dim = in_channels * expansion
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 3, padding=1, groups=in_channels),
            nn.Dropout(0.2),
            HybridAttention(hidden_dim),
            nn.Conv1d(hidden_dim, in_channels, 1),
            nn.BatchNorm1d(in_channels)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.conv(x))


class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, in_channels, 3, dilation=d, padding=d),
                nn.BatchNorm1d(in_channels),
                nn.GELU()
            ) for d in [1, 2, 4]
        ])
        self.fusion = nn.Conv1d(3 * in_channels, in_channels, 1)

    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        return self.fusion(torch.cat(features, dim=1)) + x

class StochasticDepth(nn.Module):

    def __init__(self, p=0.5, mode='batch'):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"概率值必须在[0,1]区间，当前为{p}")
        self.p = p
        self.mode = mode

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        # 根据模式生成丢弃掩码
        if self.mode == 'batch':
            mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        elif self.mode == 'layer':
            mask_shape = (1,) * x.ndim
        else:
            raise ValueError(f"不支持的模式类型: {self.mode}")

        # 生成伯努利掩码
        keep_prob = 1 - self.p
        mask = torch.empty(mask_shape, device=x.device).bernoulli_(keep_prob)

        # 训练阶段应用丢弃并缩放
        return x * mask / keep_prob

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"

class MultimodalDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def train_dataloader(self):
        return DataLoader(
            MultimodalDataset(is_train=True),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            MultimodalDataset(is_train=False),
            batch_size=self.args.batch_size,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True
        )


class MultimodalDataset(Dataset):
    def __init__(self, is_train=True):
        self.wifi_ds = XRFWiFiDataset(is_train=is_train)
        self.rfid_ds = RFDataset(is_train=is_train)

        # 验证数据集对齐
        assert len(self.wifi_ds) == len(self.rfid_ds), "数据集长度不匹配!"
        for i in range(min(len(self.wifi_ds), len(self.rfid_ds))):
            _, label1 = self.wifi_ds[i]
            _, label2 = self.rfid_ds[i]
            assert label1 == label2, f"标签不匹配 index {i}"

    def __len__(self):
        return len(self.wifi_ds)

    def __getitem__(self, idx):
        wifi, label1 = self.wifi_ds[idx]
        rfid, label2 = self.rfid_ds[idx]
        assert label1 == label2, f"标签不一致: idx={idx}, {label1} vs {label2}"
        return (wifi, rfid), label1


# -------------------- 训练模块 --------------------
class MultimodalSystem(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        # 双模态分支
        self.wifi_net = AdvancedWiFiModel(args)
        self.rfid_net = AdvancedRFIDModel(args)
        self.contrastive_loss = NTXentLoss(temperature=0.05)
        self.moe = MoE(num_experts=16, topk=4)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.2)

        # 修改指标初始化方式
        self.train_acc = Accuracy(task='multiclass', num_classes=args.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=args.num_classes)

        # 添加变量用于存储预测结果
        self.validation_outputs = []

    def forward(self, wifi, rfid):
        wifi_logits, wifi_feat = self.wifi_net(wifi)
        rfid_logits, rfid_feat = self.rfid_net(rfid)
        fused_feat = self.moe(wifi_feat, rfid_feat)
        return wifi_logits, rfid_logits, fused_feat, wifi_feat, rfid_feat

    def training_step(self, batch, batch_idx):
        (wifi, rfid), labels = batch
        wifi_logits, rfid_logits, final_logits, wifi_feat, rfid_feat = self(wifi, rfid)

        # 对比学习
        contrast_loss = self.contrastive_loss(wifi_feat, rfid_feat)

        # 分类损失
        loss_wifi = self.ce_loss(wifi_logits, labels)
        loss_rfid = self.ce_loss(rfid_logits, labels)
        loss_final = self.ce_loss(final_logits, labels)

        # 动态权重调整
        t = self.current_epoch / self.trainer.max_epochs
        alpha = 0.5 * (1 + np.cos(np.pi * t))

        total_loss = (
                alpha * (loss_wifi + loss_rfid) +
                (1 - alpha) * loss_final +
                0.3 * contrast_loss * (1 + t)
        )

        # 训练指标
        self.train_acc.update(final_logits, labels)
        self.log_dict({
            'train_loss': total_loss,
            'train_acc': self.train_acc,  # 自动处理epoch聚合
            'contrast_loss': contrast_loss,
            'alpha': alpha
        }, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        (wifi, rfid), labels = batch
        _, _, final_logits, _, _ = self(wifi, rfid)
        loss = self.ce_loss(final_logits, labels)

        # 修改指标记录方式
        preds = torch.argmax(final_logits, dim=1)
        self.val_acc.update(final_logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)

        # 保存预测结果用于混淆矩阵
        self.validation_outputs.append({
            "preds": preds.cpu().numpy(),
            "labels": labels.cpu().numpy()
        })

        return loss

    def on_validation_epoch_start(self):
        """在验证的每个epoch开始前，清空上一轮的输出结果。"""
        self.validation_outputs = []

    def on_validation_epoch_end(self):
        """
        在验证的每个epoch结束后，此函数被调用。
        我们保留这个函数为空，以防止在trainer.validate()期间意外清除结果。
        """
        pass

    def configure_optimizers(self):
        # 创建AdamW优化器
        optimizer = torch.optim.AdamW(
            self.parameters(),  # 优化所有模型参数
            lr=3e-4,  # 初始学习率
            weight_decay=0.05  # 权重衰减
        )

        # 计算总步数（每个epoch的batch数乘以总epoch数）
        total_steps = self.trainer.estimated_stepping_batches

        # 创建OneCycleLR调度器
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=3e-3,  # 峰值学习率
                total_steps=total_steps,  # 总训练步数
                pct_start=0.3,  # 学习率上升阶段占比
                anneal_strategy='cos'  # 余弦退火策略
            ),
            'interval': 'step'  # 按步更新学习率
        }

        return [optimizer], [scheduler]

    def generate_final_report(self):
        """
        生成並保存顯示百分比的混淆矩陣, 同時打印分類報告。
        (終極修正版): 放棄 seaborn 的自動註解，採用 matplotlib 手動繪製文字，確保萬無一失。
        """
        if not self.validation_outputs:
            print("错误: 验证输出为空。无法生成报告。")
            print("请确保 'trainer.validate()' 在此函数之前被成功调用。")
            return

        all_preds = np.concatenate([out["preds"] for out in self.validation_outputs])
        all_labels = np.concatenate([out["labels"] for out in self.validation_outputs])
        class_names = [f"Class {i+1}" for i in range(self.hparams.args.num_classes)]

        print("="*50)
        print("             Classification Report (Best Model)              ")
        print("="*50)
        report = classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            digits=4
        )
        print(report)
        print("="*50)

        cm = confusion_matrix(all_labels, all_preds)

        # --- 這是新的修改部分 ---
        # 1. 計算百分比 (和之前一樣)
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_percent = cm.astype('float') / cm_sum * 100
        cm_percent = np.nan_to_num(cm_percent)

        df_cm_percent = pd.DataFrame(cm_percent, index=class_names, columns=class_names)
        plt.figure(figsize=(18, 15))

        # 2. 只繪製熱力圖背景，完全不使用 annot 功能
        sns.heatmap(df_cm_percent,
                    annot=False,  # 關鍵：關閉自動註解
                    cmap="Blues",
                    cbar=True) # 可以保留顏色條

        # 3. 手動遍歷數據並在每個單元格上繪製文字
        for i in range(cm_percent.shape[0]):
            for j in range(cm_percent.shape[1]):
                value = cm_percent[i, j]
                # 只在值大於 0.1% 時顯示文字，避免 0.0 佔滿螢幕
                if value > 0.1:
                    # 判斷文字顏色：背景色深則用白色，背景色淺則用黑色
                    text_color = "white" if value > 50 else "black"
                    # 使用 plt.text 手動在中心位置添加文字
                    plt.text(j + 0.5, i + 0.5, f'{value:.1f}',
                             ha="center", va="center",
                             color=text_color,
                             fontsize=8)
        # --- 修改結束 ---

        plt.title("Confusion Matrix (Row-wise %)", fontsize=16)
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)

        os.makedirs("results", exist_ok=True)
        save_path = "results/confusion_matrix_percent_best.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n已生成新的百分比混淆矩陣, 並保存到: {save_path}")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    args = Namespace(
        num_classes=20,
        batch_size=64,
        epochs=200
    )

    system = MultimodalSystem(args)
    dm = MultimodalDataModule(args)

    # 添加模型检查点回调，监控val_acc并保存最佳模型
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        filename='best-model-epoch{epoch:02d}-val_acc{val_acc:.4f}',
        save_last=True
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=args.epochs,
        precision='16-mixed',
        callbacks=[
            checkpoint_callback,
            pl.callbacks.LearningRateMonitor(),
        ],
        gradient_clip_val=0.5,
        accumulate_grad_batches=2
    )

    # 训练模型
    print("开始模型训练...")
    trainer.fit(system, dm)
    print("模型训练完成。")

    # 加载验证准确率最高的最佳模型
    best_model_path = checkpoint_callback.best_model_path
    print(f"\n训练结束。加载最佳模型进行最终评估: {best_model_path}")
    best_model = MultimodalSystem.load_from_checkpoint(
        checkpoint_path=best_model_path,
        args=args
    )

    # 在验证集上运行最佳模型以收集最终的预测结果
    print("使用最佳模型在验证集上进行评估...")
    trainer.validate(best_model, datamodule=dm)

    # 生成并显示最终报告（包含混淆矩阵、Precision, Recall, F1-score）
    best_model.generate_final_report()

    print("\n评估完成! 分类报告已打印，混淆矩阵已保存。")

