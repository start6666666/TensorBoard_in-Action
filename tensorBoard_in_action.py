import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# 1. 设置：设备、超参数、SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.01
BATCH_SIZE = 64
NUM_EPOCHS = 5

# 创建一个数据写入器writer
writer = SummaryWriter('logs')

# 2. 数据加载与预处理
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. 模型定义
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc1(x)
        return x

model = ConvNet().to(device)

# 4. 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# 5. 初始日志记录 (循环开始前)
# 获取一批图像用于记录
examples = iter(train_loader)
example_data, example_targets = next(examples)

# 记录图像网格
img_grid = torchvision.utils.make_grid(example_data)
writer.add_image('mnist_images', img_grid)

# 记录模型计算图
writer.add_graph(model, example_data.to(device))

# 6. 训练循环
running_loss = 0.0
running_corrects = 0
n_total_steps = len(train_loader)

for epoch in range(NUM_EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算训练过程中的指标
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        running_corrects += (predicted == labels).sum().item()
        
        # 每100步记录一次训练损失和准确率
        if (i+1) % 100 == 0:
            step = epoch * n_total_steps + i + 1
            writer.add_scalar('Loss/train', running_loss / 100, step)
            writer.add_scalar('Accuracy/train', running_corrects / (100 * BATCH_SIZE), step)
            running_loss = 0.0
            running_corrects = 0

    # 7. 每个epoch结束后的验证和日志记录
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        epoch_loss = 0
        for images_val, labels_val in test_loader:
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)
            outputs_val = model(images_val)
            
            # 计算验证损失
            loss_val = criterion(outputs_val, labels_val)
            epoch_loss += loss_val.item()

            # 计算验证准确率
            _, predicted_val = torch.max(outputs_val.data, 1)
            n_samples += labels_val.size(0)
            n_correct += (predicted_val == labels_val).sum().item()

        # 记录验证指标
        val_acc = 100.0 * n_correct / n_samples
        val_loss = epoch_loss / len(test_loader)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)
        print(f'Epoch, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

        # 记录权重直方图
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
            
    model.train() # 将模型设置回训练模式

# 8. 清理
writer.close()