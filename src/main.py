import torch
from models import ConvNet
from train import train_model2
from test import test_model
from visualizer import Visualizer
import config
import os

if __name__ == "__main__":
    model = ConvNet()
    default_model_path = "../models/model.pth"
    best_model_path = "../models/best_model.pth"
    start_from_scratch = False

    if not start_from_scratch and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        print(f"从最佳模型 {best_model_path} 加载参数，继续训练")
    elif not start_from_scratch and os.path.exists(default_model_path):
        model.load_state_dict(torch.load(default_model_path, weights_only=True))
        print(f"从默认模型 {default_model_path} 加载参数，继续训练")
    else:
        print("未找到已有模型，从头开始训练")

    model.to(config.device)
    visualizer = Visualizer()
    train = True

    if train:
        train_losses = train_model2(model, config.train_loader, epochs=5,
                                    save_path=default_model_path,
                                    best_model_path=best_model_path)
        test_losses, accuracies = [], []
        for epoch in range(5):
            test_loss, accuracy, preds, labels = test_model(model, config.test_loader, load_path=default_model_path)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
        visualizer.plot_training(train_losses, test_losses, accuracies)
        visualizer.plot_confusion_matrix(preds, labels)
    else:
        test_loss, accuracy, preds, labels = test_model(model, config.test_loader, load_path=default_model_path)
        visualizer.plot_confusion_matrix(preds, labels)