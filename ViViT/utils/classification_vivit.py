import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import click
from ViViT.models.model import ViViT
import matplotlib.pyplot as plt
from ViViT.datasets.organmnist import MedMNISTDatasetLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tqdm 
import csv
import os

@click.command("classification")
@click.option("--output", type=click.Path(file_okay=False), default="output/vivit")
@click.option("--hyperparameter_dir", type=click.Path(file_okay=False), default="hyperparameter_outputs")
@click.option("--hyperparameter", type=bool, default=False)
@click.option("--num_epochs", type=int, default=60)
@click.option("--lr", type=float, default=1e-4)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=16)
@click.option("--num_classes", type=int, default=11)
@click.option("--num_heads", type=int, default=8)
@click.option("--num_layers", type=int, default=8)
@click.option("--projection_dim", type=int, default=256)
@click.option("--patch_size", type=str, default="4,4,4")
@click.option("--device", type=str, default="cuda")
@click.option("--seed", type=int, default=0)

def run(
    output,
    hyperparameter,
    hyperparameter_dir,
    num_epochs,
    lr,
    num_classes,
    num_workers,
    batch_size,
    device,
    seed,
    projection_dim,
    num_heads,
    num_layers,
    patch_size,
    input_shape  = (1, 28, 28, 28),  # (C, D, H, W),
    
):
    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)
    #Convert patch size to int 
    patch_size = tuple(map(int, patch_size.split(",")))

    os.makedirs(output, exist_ok=True)  # Ensure the base output directory exists
    if hyperparameter:
        output = hyperparameter_dir
    output = os.path.join(output, f"lr_{lr}_ps_{patch_size[0]}_nh_{num_heads}_nl_{num_layers}_pd_{projection_dim}") if hyperparameter else output
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


    model = ViViT(input_shape=input_shape, patch_size=patch_size, embed_dim=projection_dim, num_heads=num_heads, 
                  num_layers=num_layers, num_classes=num_classes).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    all_preds = []
    all_labels = []

    # Initialize a list to store learning rate values
    learning_rates = []
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
        
        # # # Step the scheduler
        # # scheduler.step()
        # # Step the scheduler at the end of each epoch or batch
        # scheduler.step(epoch + len(train_loader) / len(train_loader))  # Use batch-level granularity
        # # Log the current learning rate
        # current_lr = scheduler.get_last_lr()[0]
        # learning_rates.append(current_lr) 
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

    # Save learning rates to a NumPy file
    lr_output_path = os.path.join(output, "learning_rates.npy")
    np.save(lr_output_path, np.array(learning_rates))
    print(f"Learning rates saved to {lr_output_path}")
    return model

if __name__ == "__main__":
    run()

