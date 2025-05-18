"""
Модуль с моделью ResNet34.
"""

import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet34(pretrained=False, num_classes=1000):
    """Возвращает модель ResNet34"""
    from google.colab import drive
    drive.mount('/content/drive/')
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.autograd import Variable
    from torch.utils.data import DataLoader
    from torchvision import transforms, datasets, models
    
    import numpy as np
    import random
    from sklearn.metrics import f1_score
    import matplotlib.pyplot as plt
    from termcolor import colored
    
    random_state = 1
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    np.random.seed(random_state)
    
    
    batch_size = 50
    num_workers = 2
    use_gpu = torch.cuda.is_available()
    epochs = 50
    
    PATH = '/content/drive/MyDrive/Colab Notebooks/VKR/resnet34_model.pt'
    
    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(root='/content/drive/MyDrive/Colab Notebooks/VKR/Данные/train/', transform=data_transform)
    test_dataset = datasets.ImageFolder(root='/content/drive/MyDrive/Colab Notebooks/VKR/Данные/test/', transform=data_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f'Обучающих изображений: {len(train_dataset)}, Классы: {train_dataset.classes}')
    print(f'Тестовых изображений: {len(test_dataset)}')
    
    net = models.resnet34(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, 2)  # 2 класса
    
    if use_gpu:
        net = net.cuda()
    
    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.0)
    
    # Метрики
    f1_score_train_list = []
    f1_score_test_list = []
    acc_list = []
    loss_list = []
    
    def train():
        for epoch in range(epochs):
            net.train()
            running_loss = 0.0
            train_total = 0
            train_correct = 0
            all_preds = []
            all_labels = []
    
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
    
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    
                running_loss += loss.item()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
            f1_score_train = f1_score(all_labels, all_preds, average='micro')
            f1_score_train_list.append(round(f1_score_train, 3))
    
            # Оценка на тесте
            net.eval()
            correct = 0
            test_loss = 0.0
            test_total = 0
            test_preds = []
            test_labels = []
    
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    if use_gpu:
                        images, labels = images.cuda(), labels.cuda()
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    test_total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    test_preds.extend(predicted.cpu().numpy())
                    test_labels.extend(labels.cpu().numpy())
    
            f1_score_test = f1_score(test_labels, test_preds, average='micro')
            f1_score_test_list.append(round(f1_score_test, 3))
            acc = 100.0 * correct / test_total
            acc_list.append(round(acc, 3))
            loss_list.append(round(test_loss / test_total, 3))
    
            print(colored(f'Эпоха {epoch+1} | Потери: {test_loss / test_total:.3f} | Точность: {acc:.3f}%', 'red', attrs=['bold']))
            print(colored(f'F1 (train): {f1_score_train:.3f} | F1 (test): {f1_score_test:.3f}', 'blue'))
    
            # Сохраняем модель
            torch.save(net.state_dict(), PATH)
    
    train()
    
    print("F1 train:", f1_score_train_list)
    print("F1 test:", f1_score_test_list)
    print("Accuracy:", acc_list)
    
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    
    ax[0].plot(f1_score_train_list, label='Train F1')
    ax[0].plot(f1_score_test_list, label='Test F1')
    ax[0].set_title('F1 Score по эпохам')
    ax[0].legend()
    
    ax[1].plot(acc_list, label='Test Accuracy')
    ax[1].set_title('Accuracy по эпохам')
    ax[1].legend()
    
    plt.show()
    
    import torch
    from torchvision import models
    
    # Загружаем модель
    model = models.resnet34(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Выход на 2 класса
    model.load_state_dict(torch.load('/content/drive/MyDrive/Colab Notebooks/VKR/resnet34_model.pt'))
    model.eval()
    
    !pip install torchsummary
    from torchsummary import summary
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Выводим сводку
    summary(model, input_size=(3, 224, 224))
    
    
    return model  # убедитесь, что модель присвоена переменной `model`
