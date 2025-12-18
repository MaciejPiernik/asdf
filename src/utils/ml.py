import torch


def train(model, dataloader, optimizer, loss_fn, config):
    model.train()
    total_loss = 0

    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)

    return avg_loss


def validate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy