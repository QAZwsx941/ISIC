import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import ISICDataset
from metrics import MetricMonitor


def test(model, device, test_loader, criterion):
    """
    测试模型

    Args:
        model: 训练好的模型
        device: torch 设备
        test_loader: 测试集的 DataLoader
        criterion: 损失函数

    Returns:
        测试集的损失和评估指标
    """
    model.eval()
    metric_monitor = MetricMonitor()
    test_loss = 0.0

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs = data["image"].to(device)
            targets = data["target"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            metric_monitor.update(targets, outputs)

    return test_loss / len(test_loader), metric_monitor


if __name__ == '__main__':
    # 参数设置
    batch_size = 16
    num_workers = 4
    test_csv_path = 'data/test.csv'
    model_path = 'models/model.pth'

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.to(device)

    # 加载数据集
    test_dataset = ISICDataset(csv_file=test_csv_path, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    # 定义损失函数
    criterion = torch.nn.BCEWithLogitsLoss()

    # 测试模型
    test_loss, metric_monitor = test(model, device, test_loader, criterion)

    # 打印测试集的损失和评估指标
    print(f"Test Loss: {test_loss:.5f}")
    print(f"AUC: {metric_monitor.auc:.5f}")
    print(f"Sensitivity: {metric_monitor.sensitivity:.5f}")
    print(f"Specificity: {metric_monitor.specificity:.5f}")
