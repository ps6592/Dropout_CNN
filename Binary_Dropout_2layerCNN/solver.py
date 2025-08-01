import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
import csv
import torch.nn.functional as F
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from tqdm import tqdm
import random
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def load_image_dataset(image_dir, split='train', image_size=224, batch_size=25):

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  
    ])

    dataset_path = os.path.join(image_dir, split)
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    images, labels, paths = [], [], []
    for i in range(len(dataset)):
        img, label = dataset[i]
        path, _ = dataset.samples[i]
        images.append(img)
        labels.append(label)
        paths.append(os.path.basename(path))

    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)

    tensor_dataset = TensorDataset(images_tensor, labels_tensor)
    dataloader = DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        drop_last=True
    )

    class_names = dataset.classes 

    return dataloader, paths, class_names


class Solver(object):
    def __init__(self, model, out_csv, data_dir, batch_size=25, num_epochs=100,
                 log_dir='logs', model_save_path='model',lr=0.001, trained_model='model/model.pth',
                 device='cuda'):
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.log_dir = log_dir
        self.model_save_path = model_save_path
        self.trained_model = trained_model
        self.data_dir = data_dir
        self.device = device
        self.out_csv = out_csv
        self.lr=lr
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_save_path, exist_ok=True)

    def train(self):
        print('[*] Training...')
        writer = SummaryWriter(log_dir=self.log_dir)

        train_loader, _, _ = load_image_dataset(
            self.data_dir, split='train', batch_size=self.batch_size)

        optimizer = self.model.configure_optimizer(lr=self.lr)
        self.model.to(self.device)
        self.model.mode = 'train'
        self.model.lr = self.lr

        global_step = 0

        for epoch in range(self.num_epochs):
            print(f"\nEpoch [{epoch + 1}/{self.num_epochs}]")
            self.model.train()
            running_loss = 0.0

            for batch_idx, (batch_imgs, labels) in enumerate(train_loader):
                batch_imgs = batch_imgs.to(self.device)
                labels = labels.to(self.device)

                logits, log_var, _ = self.model(batch_imgs)
                log_var = torch.clamp(log_var, min=-10, max=10)

                loss, _, _ = self.model.build_train_loss(labels, logits, log_var)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print("logits mean:", logits.mean().item())
                print("log_var mean:", log_var.mean().item())
                print("probs mean:", torch.sigmoid(logits).mean().item())
                print("labels mean:", labels.float().mean().item())


                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("Uncertainty/Aleatoric", log_var.exp().mean().item(), global_step)
                global_step += 1

                running_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)} — Loss: {loss.item():.6f}")

            epoch_loss = running_loss / len(train_loader)
            print(f"  Epoch Loss: {epoch_loss:.6f}")

            model_path = os.path.join(self.model_save_path, f'model_epoch_{epoch + 1}.pth')
            torch.save(self.model.state_dict(), model_path)
            print(f"  Model saved to {model_path}")

        # Save final checkpoint
        final_path = os.path.join(self.model_save_path, 'model.pth')
        torch.save(self.model.state_dict(), final_path)
        print(f"  Final model saved to {final_path}")
        writer.close()

    def test(self, checkpoint, output_csv):
        print('[*] Testing...')
        test_loader, image_paths, class_names = load_image_dataset(
            self.data_dir, split='test', batch_size=self.batch_size)

        self.model.load_state_dict(torch.load(checkpoint))
        self.model.to(self.device)
        self.model.eval()
        self.model.mode = 'test'

        results = []

        for i, (batch_imgs, labels) in enumerate(tqdm(test_loader)):
            batch_imgs = batch_imgs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model.build_test_outputs(batch_imgs)

            epistemic = outputs['epistemic_var'].detach().cpu().numpy()
            aleatoric = outputs['aleatoric_var'].detach().cpu().numpy()
            preds = outputs['preds'].cpu().numpy()
            probs = outputs['probs'].detach().cpu().numpy()

            for j in range(batch_imgs.size(0)):
                index = i * self.batch_size + j
                if index >= len(image_paths): continue

                results.append({
                    'image_idx': index,
                    'image_name': image_paths[index],
                    'epistemic_var': float(epistemic[j]),
                    'aleatoric_var': float(aleatoric[j]),
                    'label': int(labels[j].item()),
                    'predicted_label': int(preds[j]),
                    'confidence': float(probs[j]) 
                })

        keys = results[0].keys()
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

        print(f"[*] Results saved to {output_csv}")
        run_name = os.path.splitext(output_csv)[0]

    
        true_labels = [r['label'] for r in results]
        pred_labels = [r['predicted_label'] for r in results]
        labels_sorted = sorted(set(true_labels + pred_labels))

        cm = confusion_matrix(true_labels, pred_labels, labels=labels_sorted)
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        plt.figure(figsize=(8, 6))
        label_names = [class_names[i] for i in labels_sorted]
        sns.heatmap(cm_normalized * 100, annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix (% normalized per row)")
        plt.tight_layout()
        cm_path = f"{run_name}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"[*] Confusion matrix saved to {cm_path}")

        pd.DataFrame(cm_normalized * 100, index=label_names, columns=label_names).to_csv(
            os.path.join(f"{run_name}_confusion_matrix_percent.csv"))

        print(f"[*] Confusion matrices (percent) saved.")
