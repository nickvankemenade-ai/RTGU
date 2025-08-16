import torch
import yaml
from torch.utils.data import DataLoader
from torch import nn, optim

# Datasets
from datasets_dir.imdb_dataset import IMDBDataset
from datasets_dir.ecg_dataset import ECGDataset

# Models
from models.lstm import LSTMClassifier
from models.gru import GRUClassifier
from models.rtgu import RTGU

# Training utils
from train_utils import train_epoch, eval_epoch, setup_results_file, log_results, count_params


def run_experiment(dataset_name, model_name, params, device, results_file, log=False):
    # ---- Dataset selection ----
    if dataset_name == "IMDB":
        train_ds = IMDBDataset(split="train")
        test_ds = IMDBDataset(split="test")
        train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=params["batch_size"])
        is_ecg = False
        vocab_size = 30522  # BERT uncased vocab
    elif dataset_name == "ECG":
        train_ds = ECGDataset(split="train")
        test_ds = ECGDataset(split="test")
        train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=params["batch_size"])
        is_ecg = True
        vocab_size = None
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # ---- Model selection ----
    if model_name == "LSTM":
        model = LSTMClassifier(
            embed_dim=params.get("embed_dim", 1),   # only IMDB specifies this
            hidden_dim=params["hidden_dim"],
            num_classes=params["num_classes"],
            vocab_size=vocab_size if not is_ecg else None,
            is_ecg=is_ecg
        ).to(device)

    elif model_name == "GRU":
        model = GRUClassifier(
            embed_dim=params.get("embed_dim", 1),
            hidden_dim=params["hidden_dim"],
            num_classes=params["num_classes"],
            vocab_size=vocab_size if not is_ecg else None,
            is_ecg=is_ecg
        ).to(device)

    elif model_name == "RTGU":
        model = RTGU(
            embed_dim=params.get("embed_dim", 1),
            hidden_dim=params["hidden_dim"],
            num_classes=params["num_classes"],
            vocab_size=vocab_size if not is_ecg else None,
            is_ecg=is_ecg
        ).to(device)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"\nRunning {model_name} on {dataset_name} | Params: {count_params(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.CrossEntropyLoss()

    # ---- Training loop ----
    for epoch in range(1, params["epochs"] + 1):
        train_loss, train_acc, train_time = train_epoch(model, train_loader, optimizer, criterion, device, log=log)
        val_loss, val_acc, val_time = eval_epoch(model, test_loader, criterion, device, log=log)

        print(
            f"[{dataset_name}] {model_name} Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Train Time: {train_time:.2f}s Val Time: {val_time:.2f}s"
        )

        log_results(results_file, dataset_name, model_name, epoch,
                    train_loss, train_acc, val_loss, val_acc, train_time, val_time)


if __name__ == "__main__":
    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load hyperparams from YAML ----
    with open("config/hyperparams.yaml", "r") as f:
        hyperparams = yaml.safe_load(f)

    # ---- Setup results file ----
    results_file = setup_results_file(path="results", filename="results.csv")

    # ---- Run experiments ----
    for dataset_name, models in hyperparams.items():
        for model_name, params in models.items():
            run_experiment(dataset_name, model_name, params, device, results_file, log=True)
