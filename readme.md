# PythonProject3 - 卷积神经网络训练与评估

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

`PythonProject3` 是一个基于 PyTorch 的卷积神经网络（CNN）项目，用于图像分类任务。项目实现了模型定义、训练、测试和结果可视化功能，支持从头训练或基于已有模型继续训练。

## 功能特点
- **模型定义**：实现了一个简单的卷积神经网络（`ConvNet`），包括卷积层、池化层和全连接层。
- **训练支持**：提供批量训练、学习率调度和早停机制，保存最佳模型和最终模型。
- **测试评估**：计算测试集的损失和准确率，生成预测结果和真实标签。
- **可视化**：绘制训练/测试损失曲线、准确率曲线和混淆矩阵。
- **模块化设计**：代码拆分为多个文件，便于维护和扩展。

## 安装

### 环境要求
- Python 3.8 或更高版本
- PyTorch 2.0 或更高版本（支持 GPU 可选）
- 其他依赖：`numpy`, `matplotlib`（用于可视化）

### 安装步骤
1. 克隆项目到本地：
   ```bash
   git clone https://github.com/yourusername/PythonProject3.git
   cd PythonProject3