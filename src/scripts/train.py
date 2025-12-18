import torch
import tomllib
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from ml.DigitRecognizer import DigitRecognizer
from utils.ml import train, validate
from utils.data import load_config


def main():
    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = load_config("config.toml")

    # load data
    digits = load_digits()
    X = digits.data
    y = digits.target

    # stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=23)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),torch.tensor(y_train, dtype=torch.long)), batch_size=config["train"]["batch_size"], shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),torch.tensor(y_test, dtype=torch.long)), batch_size=config["train"]["batch_size"], shuffle=False)

    # load model
    model = DigitRecognizer()

    # optimizer, scheduler, loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"], weight_decay=config["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, min_lr=1e-6)
    loss_fn = nn.CrossEntropyLoss()

    # training loop
    patience_counter = 0
    best_val_loss = float('inf')
    for epoch in tqdm(range(config["train"]["epochs"])):
        current_lr = optimizer.param_groups[0]['lr']

        loss_train = train(model, train_loader, optimizer, loss_fn, config)

        loss_val, val_accuracy = validate(model, test_loader, loss_fn)

        checkpoint = model.state_dict()

        print(f"Epoch {epoch+1}/{config['train']['epochs']}, Train Loss: {loss_train:.4f}, Val Loss: {loss_val:.4f}, Val Accuracy: {val_accuracy:.4f}" + f", LR: {current_lr:.6f}")

        if current_lr <= 1e-6:
            print("Learning rate has reached the minimum threshold. Stopping training.")
            break

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            patience_counter = 0

            torch.save({'config': config, 'model_state_dict': checkpoint}, f"{config['files']['output_dir']}model_{run}.pth")
        else:
            patience_counter += 1
            if patience_counter >= config["train"]["patience"]:
                print("Early stopping triggered.")
                break

        scheduler.step(loss_val)


if __name__ == "__main__":
    main()