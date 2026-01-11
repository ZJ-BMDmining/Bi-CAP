import os
import sys
import logging
import torch
sys.path.append('../')
sys.path.append('../scripts')
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model.ResNet import resnet18
import skimage.transform as skTrans
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from itertools import chain
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from scripts.utils import CrossAttentionFusion, DualCodebookVQ, model_save, ConcaModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class SimulatedDataset(Dataset):
    def __init__(self, csv_path=None, split='train', val_split=0.2, random_state=42):
        if csv_path is None:
            csv_path = 'dataset/Clinic.csv'

        selected_columns = ['age','gender','education','hispanic','race','apoe','mmse','cdr','cdrSum','Tesla','trailA',
        'trailB','lm_imm','lm_del','boston','animal','vege','digitB','digitBL','digitF','digitFL','npiq_DEL','npiq_HALL',
        'npiq_AGIT','npiq_DEPD','npiq_ANX','npiq_ELAT','npiq_APA','npiq_DISN','npiq_IRR','npiq_MOT','npiq_NITE','npiq_APP',
        'faq_BILLS','faq_TAXES','faq_SHOPPING','faq_GAMES','faq_STOVE','faq_MEALPREP','faq_EVENTS','faq_PAYATTN','faq_REMDATES',
        'faq_TRAVEL','his_CVHATT','his_PSYCDIS','his_Alcohol','his_SMOKYRS','his_PACKSPER','his_NACCFAM','his_CBSTROKE','his_HYPERTEN',
        'his_DEPOTHR','gds','moca']
        df = pd.read_csv(csv_path)

        if df['COG'].dtype == 'object':
            label_mapping = {'NC': 0, 'MCI': 1, 'DE': 2, 'AD': 2}  # 根据数据格式调整
            df['COG'] = df['COG'].map(label_mapping)

        # 先对完整数据集进行one-hot编码，以获取所有可能的类别
        df_full_dummies = pd.get_dummies(df[selected_columns], drop_first=False)
        self.dummy_columns = df_full_dummies.columns.tolist()  # 保存所有列名

        # 划分训练集和验证集
        if split == 'train':
            df, _ = train_test_split(df, test_size=val_split, random_state=random_state, stratify=df['COG'])
            df = df.reset_index(drop=True)  # 重置索引
        elif split == 'valid':
            _, df = train_test_split(df, test_size=val_split, random_state=random_state, stratify=df['COG'])
            df = df.reset_index(drop=True)  # 重置索引

        # 使用相同的列名进行one-hot编码，确保训练集和验证集有相同数量的特征
        df_dummies = pd.get_dummies(df[selected_columns], drop_first=False)
        # 确保所有列都存在，如果某些类别在子集中不存在，添加缺失的列（值为0）
        for col in self.dummy_columns:
            if col not in df_dummies.columns:
                df_dummies[col] = 0

        # 按原始顺序重新排列列
        df_dummies = df_dummies[self.dummy_columns]

        self.scaler = StandardScaler()
        self.scaler.fit(df_dummies)

        # 标准化
        self.text_data = self.scaler.transform(df_dummies)
        self.text_data = np.nan_to_num(self.text_data, nan=0.0)
        self.file_name = df['filename']
        self.true_label = df['COG']

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        data = nib.load(f'/home/wcj/data/old_MRI/{self.file_name[idx]}').get_fdata()
        data_cnn = (data - np.mean(data)) / (np.std(data) + 1e-7)
        data1 = skTrans.resize(data_cnn[(data.shape[0]-1)//2,:,:], (224, 224), order=1, preserve_range=True)
        data2 = skTrans.resize(data_cnn[:,(data.shape[1]-1)//2,:], (224, 224), order=1, preserve_range=True)
        data3 = skTrans.resize(data_cnn[:,:,(data.shape[2]-1)//2], (224, 224), order=1, preserve_range=True)
        data_cnn = torch.tensor(np.array([data1,data2,data3]), dtype=torch.float32)
        data_resnet = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        return torch.tensor(self.text_data[idx], dtype=torch.float32), self.true_label[idx], data_resnet, data_cnn

def train_model(csv_path='dataset/Clinic.csv'):
    """训练模型，直接使用Clinic数据集"""
    print("Training model on Clinic dataset")

    batch_size = 16
    train_dataset = SimulatedDataset(csv_path, split='train', val_split=0.2)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 创建验证数据集
    valid_dataset = SimulatedDataset(csv_path, split='valid', val_split=0.2)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = ConcaModel().to(device)
    vq_layer = DualCodebookVQ(n_embeddings=256, embedding_dim=256).to(device)
    classify_fusion_LC = CrossAttentionFusion().to(device)
    classify_fusion_LG = CrossAttentionFusion().to(device)

    # 计算类别权重来处理类别不平衡
    train_labels = train_dataset.true_label.values
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Using class weights: {class_weights}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(chain(
        model.parameters(),
        vq_layer.parameters(),
        classify_fusion_LC.parameters(),
        classify_fusion_LG.parameters()
    ), lr=1e-4, weight_decay=0.01, eps=1e-8)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-4,
        total_steps=1000*len(train_dataloader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    start_epoch = 0
    scaler = GradScaler()
    best_acc = 0.0
    checkpoint_dir = "/home/wcj/Desktop/Bi-CAMP/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 早停相关变量
    best_lg_f1 = 0.0
    patience = 20  # 增加耐心值
    patience_counter = 0
    model_saved = False  # 标记是否已经保存过模型

    for epoch in range(start_epoch, 1001):
        model.train()
        correct_LC = 0
        correct_LG = 0
        correct_clip = 0
        total = 0

        for batch in train_dataloader:
            optimizer.zero_grad(set_to_none=True)
            texts, labels, img1, img2 = [x.to(device, non_blocking=True) for x in batch]

            with autocast("cuda"):
                Vision_outputs, Text_outputs = model(img1, img2, texts, pre = True)
                Vision_outputs = F.layer_norm(Vision_outputs, (Vision_outputs.size(-1),))
                Text_outputs = F.layer_norm(Text_outputs, (Text_outputs.size(-1),))

                mri_G_quant, mri_L_quant, clinic_quant, vq_loss = vq_layer(Vision_outputs, Text_outputs)

                mri_L_quant = mri_L_quant + Vision_outputs[:, 256:]
                mri_G_quant = mri_G_quant + Vision_outputs[:, :256]
                clinic_quant = clinic_quant + Text_outputs

                pred_LC, fusion_LC = classify_fusion_LC(mri_L_quant, clinic_quant)
                pred_LG, fusion_LG = classify_fusion_LG(mri_G_quant, mri_L_quant)

                LC_loss = loss_fn(pred_LC, labels)
                LG_loss = loss_fn(pred_LG, labels)

                fusion_LG = F.normalize(fusion_LG, p=2, dim=-1)
                fusion_LC = F.normalize(fusion_LC, p=2, dim=-1)
                logit_scale = model.logit_scale.exp().clamp(max=100)

                # KNN-based relaxed cross-modal alignment (RCA)
                batch_size_current = fusion_LG.size(0)
                # Create similarity matrix
                similarity_matrix = torch.matmul(fusion_LG, fusion_LC.t()) * logit_scale

                # Create relaxed alignment loss - only allow matching within same label
                logits_per_LG = torch.zeros_like(similarity_matrix)
                logits_per_LC = torch.zeros_like(similarity_matrix)

                # For each sample, only consider samples with the same label
                for i in range(batch_size_current):
                    # Find samples with same label as sample i
                    same_label_mask = (labels == labels[i])
                    same_label_indices = torch.where(same_label_mask)[0]

                    if len(same_label_indices) > 1:  # Need at least 2 samples for meaningful comparison
                        # Get similarities only for same-label samples
                        same_label_similarities = similarity_matrix[i, same_label_indices]

                        # Find top-k similar samples within same label (k=5 or number of same-label samples)
                        k = min(5, len(same_label_similarities))
                        _, top_k_local_indices = torch.topk(same_label_similarities, k, dim=0)

                        # Convert local indices back to global indices
                        top_k_global_indices = same_label_indices[top_k_local_indices]

                        # Set logits for top-k similar samples with same label
                        logits_per_LG[i, top_k_global_indices] = similarity_matrix[i, top_k_global_indices]

                        # For LC logits (text-to-image), set corresponding entries
                        # Since logits_per_LC should be transpose of logits_per_LG for symmetric contrastive loss
                        logits_per_LC[top_k_global_indices, i] = similarity_matrix[i, top_k_global_indices]
                    else:
                        # If only one sample with this label, create self-matching
                        logits_per_LG[i, i] = similarity_matrix[i, i]
                        logits_per_LC[i, i] = similarity_matrix[i, i]

                # Relaxed contrastive loss
                loss_LG = F.cross_entropy(logits_per_LG, torch.arange(len(logits_per_LG), device=device))
                loss_LC = F.cross_entropy(logits_per_LC, torch.arange(len(logits_per_LC), device=device))
                contrastive_loss = 0.5*(loss_LG + loss_LC) + F.kl_div(
                    F.log_softmax(logits_per_LG, dim=1),
                    F.softmax(logits_per_LC.T, dim=1),
                    reduction='batchmean'
                )

                preds_LG = logits_per_LG.argmax(dim=1)
                preds_LC = logits_per_LC.argmax(dim=1)
                clip_labels = torch.arange(len(logits_per_LC), device=device)
                batch_clip_correct = ((preds_LG == clip_labels) & (preds_LC == clip_labels)).sum().item()
                correct_clip += batch_clip_correct

                # 调整损失权重，更关注分类任务
                total_loss = 0.25 * contrastive_loss + 0.6 * (LC_loss + LG_loss) + 0.15 * vq_loss
                total_loss = torch.clamp(total_loss, max=1e3)

            scaler.scale(total_loss).backward()
            # 降低梯度裁剪阈值，允许更大的梯度更新
            clip_grad_norm_(
                chain(
                    model.parameters(),
                    vq_layer.parameters(),
                    classify_fusion_LC.parameters(),
                    classify_fusion_LG.parameters()
                ), 1.0)

            if torch.isnan(total_loss):
                raise FloatingPointError("NaN loss detected")

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            _, predicted_LC = torch.max(pred_LC, 1)
            correct_LC += (predicted_LC == labels).sum().item()
            _, predicted_LG = torch.max(pred_LG, 1)
            correct_LG += (predicted_LG == labels).sum().item()
            total += labels.size(0)

        # 验证阶段
        model.eval()
        valid_correct_LC = 0
        valid_correct_LG = 0
        valid_correct_clip = 0
        valid_total = 0

        # 用于计算f1分数的列表
        all_labels = []
        all_preds_LG = []

        with torch.no_grad():
            for batch in valid_dataloader:
                texts, labels, img1, img2 = [x.to(device) for x in batch]

                Vision_outputs, Text_outputs = model(img1, img2, texts, pre=True)
                Vision_outputs = F.layer_norm(Vision_outputs, (Vision_outputs.size(-1),))
                Text_outputs = F.layer_norm(Text_outputs, (Text_outputs.size(-1),))

                mri_G_quant, mri_L_quant, clinic_quant, _ = vq_layer(Vision_outputs, Text_outputs)

                mri_L_quant = mri_L_quant + Vision_outputs[:, 256:]
                mri_G_quant = mri_G_quant + Vision_outputs[:, :256]
                clinic_quant = clinic_quant + Text_outputs

                pred_LC, fusion_LC = classify_fusion_LC(mri_L_quant, clinic_quant)
                pred_LG, fusion_LG = classify_fusion_LG(mri_G_quant, mri_L_quant)

                fusion_LG = F.normalize(fusion_LG, p=2, dim=-1)
                fusion_LC = F.normalize(fusion_LC, p=2, dim=-1)
                logit_scale = model.logit_scale.exp().clamp(max=100)

                similarity_matrix = torch.matmul(fusion_LG, fusion_LC.t()) * logit_scale
                logits_per_LG = similarity_matrix
                logits_per_LC = similarity_matrix.T

                _, predicted_LC = torch.max(pred_LC, 1)
                _, predicted_LG = torch.max(pred_LG, 1)
                preds_clip_LG = logits_per_LG.argmax(dim=1)
                preds_clip_LC = logits_per_LC.argmax(dim=1)

                valid_correct_LC += (predicted_LC == labels).sum().item()
                valid_correct_LG += (predicted_LG == labels).sum().item()
                batch_clip_correct = ((preds_clip_LG == torch.arange(len(logits_per_LC), device=device)) & (preds_clip_LC == torch.arange(len(logits_per_LC), device=device))).sum().item()
                valid_correct_clip += batch_clip_correct
                valid_total += labels.size(0)

                # 收集用于f1计算的数据
                all_labels.extend(labels.cpu().numpy())
                all_preds_LG.extend(predicted_LG.cpu().numpy())

        acc_LC = 100 * correct_LC / total
        acc_LG = 100 * correct_LG / total
        clip_acc = 100 * correct_clip / total

        valid_acc_LC = 100 * valid_correct_LC / valid_total
        valid_acc_LG = 100 * valid_correct_LG / valid_total
        valid_clip_acc = 100 * valid_correct_clip / valid_total

        # 计算LG的f1分数 (weighted average)
        lg_f1 = f1_score(all_labels, all_preds_LG, average='weighted')

        logging.info(f"Clinic, Epoch {epoch+1}, Train - Loss: {total_loss.item():.4f}, ACC_LC: {acc_LC:.4f}%, ACC_LG: {acc_LG:.4f}%, Clip_ACC: {clip_acc:.4f}%")
        logging.info(f"Clinic, Epoch {epoch+1}, Valid - ACC_LC: {valid_acc_LC:.4f}%, ACC_LG: {valid_acc_LG:.4f}%, LG_F1: {lg_f1:.4f}, Clip_ACC: {valid_clip_acc:.4f}%")

        # 放宽训练集clip_acc的要求，从95%降低到85%，更早开始监控F1
        if clip_acc > 85.0:
            # 检查是否是最好的LG f1分数
            if lg_f1 > best_lg_f1:
                best_lg_f1 = lg_f1
                patience_counter = 0
                print(f"Clinic, Epoch {epoch+1}: New best LG F1: {lg_f1:.4f}")

                # 保存当前最好的模型权重
                model_save_path = os.path.join(checkpoint_dir, 'model_para_pretrain.pth')
                torch.save({
                    'epoch': epoch,
                    'fold': 'clinic',
                    'model': model.state_dict(),
                    'vq_layer': vq_layer.state_dict(),
                    'fusion_LC': classify_fusion_LC.state_dict(),
                    'fusion_LG': classify_fusion_LG.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_acc': best_acc,
                    'best_lg_f1': best_lg_f1,
                    'valid_acc': valid_clip_acc,
                    'valid_lg_f1': lg_f1
                }, model_save_path)
                print(f"Saved best model for Clinic with LG F1: {lg_f1:.4f}")
                logging.info(f"Saved best model for Clinic with LG F1: {lg_f1:.4f} at epoch {epoch+1}")
                model_saved = True
            else:
                patience_counter += 1
                print(f"Clinic, Epoch {epoch+1}: LG F1: {lg_f1:.4f}, Patience: {patience_counter}/{patience}")
        else:
            print(f"Clinic, Epoch {epoch+1}: Train Clip_ACC: {clip_acc:.2f}% (< 85%), F1 monitoring disabled")

        if valid_clip_acc > best_acc:
            best_acc = valid_clip_acc

        # 早停检查：只有当训练集clip_acc > 85%时才应用早停
        if clip_acc > 85.0:
            if patience_counter >= patience:
                print(f"Clinic: Early stopping at epoch {epoch+1} due to LG F1 not improving for {patience} epochs")
                logging.info(f"Clinic: Early stopping at epoch {epoch+1} due to LG F1 not improving for {patience} epochs")
                print(f"Clinic: Best LG F1 achieved: {best_lg_f1:.4f}")
                break
        else:
            # 重置patience_counter，因为还没有达到训练集clip_acc的要求
            patience_counter = 0
            print(f"Clinic, Epoch {epoch+1}: Train Clip_ACC: {clip_acc:.2f}% (< 85%), early stopping disabled")

        # 如果训练到最后也没有保存过模型（理论上不会发生，因为f1会在某个时候提高），保存最后的模型
        if epoch == 999:  # epoch从0开始，999是第1000次迭代
            print(f"Clinic: Training completed without early stopping. Best LG F1: {best_lg_f1:.4f}")
            logging.info(f"Clinic: Training completed without early stopping. Best LG F1: {best_lg_f1:.4f}")
            break

    # 确保至少保存了一个模型（最后的fallback）
    if not model_saved:
        print(f"Clinic: No model saved during training, saving final model...")
        model_save_path = os.path.join(checkpoint_dir, 'model_para_fold1.pth')
        torch.save({
            'epoch': epoch if 'epoch' in locals() else 999,
            'fold': 'clinic',
            'model': model.state_dict(),
            'vq_layer': vq_layer.state_dict(),
            'fusion_LC': classify_fusion_LC.state_dict(),
            'fusion_LG': classify_fusion_LG.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_acc': best_acc,
            'best_lg_f1': best_lg_f1,
            'valid_acc': valid_clip_acc if 'valid_clip_acc' in locals() else 0.0,
            'valid_lg_f1': lg_f1 if 'lg_f1' in locals() else 0.0
        }, model_save_path)
        print(f"Saved final model for Clinic")
        logging.info(f"Saved final fallback model for Clinic")

def main():
    # 直接使用Clinic数据集进行训练
    csv_path = "dataset/Clinic.csv"
    if os.path.exists(csv_path):
        train_model(csv_path)
        print("Training completed successfully")
        logging.info("Training completed successfully")
    else:
        print(f"Error: {csv_path} not found")
        logging.error(f"Training CSV file not found: {csv_path}")


if __name__ == "__main__":
    main()
