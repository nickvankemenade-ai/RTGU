import torch
import os
import csv
import time

def train_epoch(model, loader, optimizer, criterion, device, is_lm=False, vocab_size=None, clip=None, log=False):
    model.train()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    start = time.time()

    for batch in loader:
        optimizer.zero_grad()

        if is_lm:  # WikiText-2 LM
            x = batch.to(device)  # (B, T+1)
            inputs = x[:, :-1]
            targets = x[:, 1:].contiguous()
            logits = model(inputs)  # (B, T, V)
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
        else:  # IMDB classification
            inputs, labels = [b.to(device) for b in batch]
            logits = model(inputs)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_tokens += labels.numel()

        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item()

    elapsed = time.time() - start
    avg_loss = total_loss / len(loader)

    if is_lm:
        ppl = float(torch.exp(torch.tensor(avg_loss)))
        if log:
            print(f"[Train] Loss: {avg_loss:.4f}, PPL: {ppl:.2f}, Time: {elapsed:.2f}s")
        return avg_loss, ppl, elapsed
    else:
        acc = total_correct / max(1, total_tokens)
        if log:
            print(f"[Train] Loss: {avg_loss:.4f}, Acc: {acc:.4f}, Time: {elapsed:.2f}s")
        return avg_loss, acc, elapsed


def eval_epoch(model, loader, criterion, device, is_lm=False, vocab_size=None, log=False):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    start = time.time()

    with torch.no_grad():
        for batch in loader:
            if is_lm:  # WikiText-2 LM
                x = batch.to(device)
                inputs = x[:, :-1]
                targets = x[:, 1:].contiguous()
                logits = model(inputs)
                loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            else:  # IMDB classification
                inputs, labels = [b.to(device) for b in batch]
                logits = model(inputs)
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_tokens += labels.numel()

            total_loss += loss.item()

    elapsed = time.time() - start
    avg_loss = total_loss / len(loader)

    if is_lm:
        ppl = float(torch.exp(torch.tensor(avg_loss)))
        if log:
            print(f"[Eval]  Loss: {avg_loss:.4f}, PPL: {ppl:.2f}, Time: {elapsed:.2f}s")
        return avg_loss, ppl, elapsed
    else:
        acc = total_correct / max(1, total_tokens)
        if log:
            print(f"[Eval]  Loss: {avg_loss:.4f}, Acc: {acc:.4f}, Time: {elapsed:.2f}s")
        return avg_loss, acc, elapsed


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_results_file(path="results", filename="results.csv"):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, filename)
    if not os.path.exists(file_path):
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Dataset", "Model", "Epoch",
                "Train_Loss", "Train_Metric",
                "Val_Loss", "Val_Metric",
                "Train_Time", "Val_Time"
            ])
    return file_path


def log_results(file_path, dataset, model_name, epoch,
                train_loss, train_metric, val_loss, val_metric, train_time, val_time):
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            dataset, model_name, epoch,
            train_loss, train_metric,
            val_loss, val_metric,
            train_time, val_time
        ])
