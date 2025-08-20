import torch
import yaml
from torch.utils.data import DataLoader
from torch import nn, optim

# Datasets
from datasets_dir.imdb_dataset import IMDBDataset
from datasets_dir.wikitext_dataset import WikiTextDataset

# Models
from models.lstm import LSTMClassifier
from models.gru import GRUClassifier
from models.rtgu import RTGU

# Training utils
from train_utils import setup_results_file, log_results, count_params


def _get_logits(maybe_tuple):
    return maybe_tuple[0] if isinstance(maybe_tuple, tuple) else maybe_tuple


def train_epoch(model, loader, optimizer, criterion, device, is_lm=False, vocab_size=None, clip=None):
    model.train()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    total_batches = 0
    import time
    start_time = time.time()

    for batch in loader:
        optimizer.zero_grad()

        if is_lm:  # language modeling
            x = batch.to(device)               # (B, T+1)
            inputs = x[:, :-1]
            targets = x[:, 1:].contiguous()
            logits = _get_logits(model(inputs))  # (B, T, V)
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
        else:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                inputs, _, labels = [b.to(device) for b in batch]
                logits = model(inputs)
            else:
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
        total_batches += 1

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, total_batches)
    if is_lm:
        ppl = float(torch.exp(torch.tensor(avg_loss)))
        return avg_loss, ppl, elapsed
    else:
        acc = total_correct / max(1, total_tokens)
        return avg_loss, acc, elapsed


def eval_epoch(model, loader, criterion, device, is_lm=False, vocab_size=None):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    total_batches = 0
    import time
    start_time = time.time()

    with torch.no_grad():
        for batch in loader:
            if is_lm:
                x = batch.to(device)
                inputs = x[:, :-1]
                targets = x[:, 1:].contiguous()
                logits = _get_logits(model(inputs))
                loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            else:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    inputs, _, labels = [b.to(device) for b in batch]
                    logits = model(inputs)
                else:
                    inputs, labels = [b.to(device) for b in batch]
                    logits = model(inputs)
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_tokens += labels.numel()

            total_loss += loss.item()
            total_batches += 1

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, total_batches)
    if is_lm:
        ppl = float(torch.exp(torch.tensor(avg_loss)))
        return avg_loss, ppl, elapsed
    else:
        acc = total_correct / max(1, total_tokens)
        return avg_loss, acc, elapsed


def run_experiment(dataset_name, model_name, params, device, results_file, log=False):
    # ---- Dataset selection ----
    if dataset_name.upper() == "IMDB":
        train_ds = IMDBDataset(split="train")
        test_ds  = IMDBDataset(split="test")
        train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=params["batch_size"])
        is_lm = False
        vocab_size = 30522  # BERT uncased vocab for our Embedding
        num_classes = params["num_classes"]
        reset_params = True

    elif dataset_name.upper() == "WIKITEXT2":
        seq_len = params.get("seq_len", 128)  # context length
        # Build vocab on train, reuse on validation
        train_ds = WikiTextDataset(split="train", block_size=seq_len)
        test_ds  = WikiTextDataset(split="validation", block_size=seq_len, vocab=train_ds.vocab)
        train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True, drop_last=True)
        test_loader  = DataLoader(test_ds,  batch_size=params["batch_size"], drop_last=True)
        is_lm = True
        vocab_size = train_ds.vocab_size
        num_classes = vocab_size
        reset_params = False

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # ---- Model selection ----
    model_kwargs = dict(
        vocab_size=vocab_size,
        embed_dim=params.get("embed_dim", 256),
        hidden_dim=params["hidden_dim"],
        num_layers=params.get("num_layers", 2),
        num_classes=num_classes,
        is_lm=is_lm
    )

    if model_name == "LSTM":
        model = LSTMClassifier(**model_kwargs).to(device)
    elif model_name == "GRU":
        model = GRUClassifier(**model_kwargs).to(device)
    elif model_name == "RTGU":
        model = RTGU(**model_kwargs, reset_params=reset_params).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"\nRunning {model_name} on {dataset_name} | Params: {count_params(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.CrossEntropyLoss()

    # ---- Training loop ----
    for epoch in range(1, params["epochs"] + 1):
        train_loss, train_metric, train_time = train_epoch(
            model, train_loader, optimizer, criterion, device,
            is_lm=is_lm, vocab_size=vocab_size, clip=params.get("clip", None)
        )
        val_loss, val_metric, val_time = eval_epoch(
            model, test_loader, criterion, device,
            is_lm=is_lm, vocab_size=vocab_size
        )

        if is_lm:
            print(
                f"[{dataset_name}] {model_name} Epoch {epoch} | "
                f"Train Loss: {train_loss:.4f} PPL: {train_metric:.2f} | "
                f"Val Loss: {val_loss:.4f} PPL: {val_metric:.2f} | "
                f"Train Time: {train_time:.2f}s Val Time: {val_time:.2f}s"
            )
        else:
            print(
                f"[{dataset_name}] {model_name} Epoch {epoch} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_metric:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_metric:.4f} | "
                f"Train Time: {train_time:.2f}s Val Time: {val_time:.2f}s"
            )

        log_results(
            results_file, dataset_name, model_name, epoch,
            train_loss, train_metric, val_loss, val_metric, train_time, val_time
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("config/hyperparams.yaml", "r") as f:
        hyperparams = yaml.safe_load(f)

    for dataset_name, models in hyperparams.items():
        if dataset_name.upper() == "IMDB":
            results_file = setup_results_file(path="results", filename="imdb_results.csv")
        elif dataset_name.upper() == "WIKITEXT2":
            results_file = setup_results_file(path="results", filename="wikitext2_results.csv")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        for model_name, params in models.items():
            run_experiment(dataset_name, model_name, params, device, results_file, log=True)
