import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import click
from ViViT.models.cnn import CNN3D
import matplotlib.pyplot as plt
from ViViT.datasets.organmnist import MedMNISTDatasetLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tqdm 
import csv
import os


@click.option("--output", type=click.Path(file_okay=False), default="output/cnn3d")
@click.option("--num_epochs", type=int, default=60)
@click.option("--lr", type=float, default=1e-4)
@click.option("--weight_decay", type=float, default=1e-5)
@click.option("--num_workers", type=int, default=4)
@click.option("--num_classes", type=int, default=11)
@click.option("--batch_size", type=int, default=16)
@click.option("--device", type=str, default="cuda")
@click.option("--seed", type=int, default=0)

def run(
    output,
    num_epochs,
    lr,
    weight_decay,
    num_classes,
    num_workers,
    batch_size,
    device,
    seed,
    input_shape  = (1, 28, 28, 28),  # (C, D, H, W),
    
):
    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(output, exist_ok=True)
    # Set the device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare the CSV log file
    log_file_path = os.path.join(output, "log.csv")
    with open(os.path.join(log_file_path), mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Training Loss", "Validation Loss", "Validation Accuracy (%)"])

    
    dataset_loader = MedMNISTDatasetLoader(dataset_name="organmnist3d", batch_size=batch_size,num_workers=num_workers)
    train_loader, val_loader, test_loader = dataset_loader.prepare_data_loaders()


    model = CNN3D(input_shape, num_classes).to(device)
    print(model)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    all_preds = []
    all_labels = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        # Training with tqdm
        with tqdm.tqdm(total=len(train_loader)) as pbar:
            for data, labels in train_loader:
                data = data.float().to(device)
                labels = labels[:, 0].long().to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                pbar.set_postfix_str("Epoch:{}, Loss: {:.2f} ".format(epoch, loss.item()))
                pbar.update(1)

                

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase with tqdm
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0

   
        with torch.no_grad():
            with tqdm.tqdm(total=len(val_loader)) as pbar:
                for data, labels in val_loader:
                    data = data.float().to(device)
                    labels = labels[:, 0].long().to(device)
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    pbar.set_postfix_str("Epoch:{}, Loss: {:.2f} ".format(epoch, loss.item()))
                    pbar.update(1)

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = 100 * correct / total

        # Log results to CSV
        log_file_path = os.path.join(output, "log.csv")
        with open(log_file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, val_accuracy])

    # Test phase with tqdm
    model.eval()
    total_test_loss = 0.0
    correct = 0
    total = 0
    correct_top1 = 0  # Top-1 doğruluk için başlangıç değeri
    correct_top5 = 0  # Top-5 doğruluk için başlangıç değeri

    with torch.no_grad():
        with tqdm.tqdm(total=len(test_loader)) as pbar:
            for data, labels in test_loader:
                data = data.float().to(device)
                labels = labels[:, 0].long().to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

                # Top-k Accuracy hesaplama
                _, top_k_preds = outputs.topk(5, dim=1)  # Top 5 tahmin
                total += labels.size(0)

                # Top-1 doğruluk
                correct_top1 += (top_k_preds[:, 0] == labels).sum().item()

                # Top-5 doğruluk
                for i in range(labels.size(0)):
                    if labels[i] in top_k_preds[i]:
                        correct_top5 += 1

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
                # Collect predictions and true labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    top1_accuracy = 100 * correct_top1 / total
    top5_accuracy = 100 * correct_top5 / total

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
    # Log test results
    log_file_path = os.path.join(output, "log.csv")
    with open(log_file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Test Results, top1 ,top5", avg_test_loss, top1_accuracy, top5_accuracy])

    # Test phase with tqdm
    model.eval()
    total_test_loss = 0.0
    correct = 0
    total = 0

    # Confusion Matrix
    class_names = [f"Class {i}" for i in range(num_classes)]
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # Plot and save confusion matrix
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    output_path = os.path.join(output, 'confusion_matrix.png')
    plt.savefig(output_path)
  

    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training, Validation, and Test Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    # Save the plot as a PNG file
    output_path = os.path.join(output,'training_validation_loss_plot.png')
    plt.savefig(output_path)

    return model

if __name__ == "__main__":
    run()

