import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


class Visualizer:
    def __init__(self):
        pass

    def plot_training(self, train_losses, test_losses=None, accuracies=None):
        plt.figure(figsize=(12, 4))

        # 训练损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        if test_losses:
            plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()

        # 测试准确率曲线
        if accuracies:
            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(accuracies) + 1), accuracies, label="Test Accuracy", color="green")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.title("Accuracy Curve")
            plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, preds, labels):
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()


# 示例用法
if __name__ == "__main__":
    # 假设的数据
    visualizer = Visualizer()
    visualizer.plot_training([0.1, 0.05, 0.03], [0.08, 0.04, 0.02], [98, 99, 99.5])
    visualizer.plot_confusion_matrix([0, 1, 2], [0, 1, 2])