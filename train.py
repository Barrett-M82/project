import torch
import torch.nn as nn
import config
import os


def train_model(model, train_loader, epoch, save_path="model.pth"):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    running_loss = 0.0
    epoch_losses = []
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(config.device)
        labels = labels.to(config.device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_losses.append(loss.item())
        print(f"Epoch [{epoch + 1}],[i]:{i} Loss: {loss.item():.3f}")

    avg_loss = running_loss / len(train_loader)
    scheduler.step(avg_loss)
    print(f"Avg Loss: {avg_loss:.3f}")
    return epoch_losses


def train_model2(model, train_loader, epochs, save_path="model.pth", best_model_path="best_model.pth"):
    best_loss = float('inf')
    patience = 3
    counter = 0
    train_losses = []

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        print(f"已加载最佳模型参数从 {best_model_path}")

    model.to(config.device)
    for epoch in range(epochs):
        epoch_losses = train_model(model, train_loader, epoch, save_path)
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)

        print(f"Epoch {epoch + 1}, Avg Loss: {avg_epoch_loss:.3f}")
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型到 {best_model_path}")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

        train_losses.append(avg_epoch_loss)
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到 {save_path}")
    return train_losses