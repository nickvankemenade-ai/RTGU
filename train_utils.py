import torch
import os
import csv
import time

def train_epoch(model, loader, optimizer, criterion, device, log=False):
    model.train()
    total_loss, total_correct = 0, 0
    start = time.time()

    for batch in loader:
        if len(batch) == 3:  # IMDB
            input_ids, attn_mask, labels = batch
        else:  # ECG
            input_ids, labels = batch
            attn_mask = None

        input_ids, labels = input_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attn_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = total_correct / len(loader.dataset)
    elapsed = time.time() - start

    if log:
        print(f"[Train] Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, Time: {elapsed:.2f}s")

    return avg_loss, avg_acc, elapsed


def eval_epoch(model, loader, criterion, device, log=False):
    model.eval()
    total_loss, total_correct = 0, 0
    start = time.time()

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:  # IMDB
                input_ids, attn_mask, labels = batch
            else:  # ECG
                input_ids, labels = batch
                attn_mask = None

            input_ids, labels = input_ids.to(device), labels.to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * input_ids.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = total_correct / len(loader.dataset)
    elapsed = time.time() - start

    if log:
        print(f"[Eval]  Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, Time: {elapsed:.2f}s")

    return avg_loss, avg_acc, elapsed


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_results_file(path="results", filename="results.csv"):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, filename)
    if not os.path.exists(file_path):
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Dataset", "Model", "Epoch", "Train_Loss", "Train_Acc", "Val_Loss", "Val_Acc", "Train_Time", "Val_Time"])
    return file_path


def log_results(file_path, dataset, model_name, epoch, train_loss, train_acc, val_loss, val_acc, train_time, val_time):
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([dataset, model_name, epoch, train_loss, train_acc, val_loss, val_acc, train_time, val_time])
