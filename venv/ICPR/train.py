from dataloader import AgriculturalDataset
from models import SimpleCNN
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF
from PIL import Image

if __name__ == '__main__':
    transform = Compose([
        Resize((264, 264)),
        ToTensor(),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = AgriculturalDataset(root="D:\\pythonProject\\data", train=True, transform=transform)

    # Lấy các chỉ số và nhãn tương ứng từ dataset
    indices = list(range(len(dataset)))
    labels = [dataset.labels[idx] for idx in indices]

    # Chia dữ liệu theo lớp với tỷ lệ 80% cho huấn luyện và 20% cho kiểm tra
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42)

    # Tạo các Subset (tập hợp con) từ dataset dựa trên các chỉ số đã chia
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=24,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=24,
        num_workers=2,
        drop_last=False,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 100


    for epoch in range(num_epochs):
        model.train()
        process_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(process_bar):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            process_bar.set_description(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        torch.save(model.state_dict(), f"saved_model/epoch_{epoch}.pth")

        model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        conf_matrix = confusion_matrix(all_labels, all_predictions)
        acc = accuracy_score(all_labels, all_predictions)
        print(f'\nEpoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {acc:.4f}')



    #
    # def show_image(tensor_image):
    #     image = TF.to_pil_image(tensor_image)  # Chuyển đổi tensor về đối tượng PIL.Image
    #     image.show()  # Hiển thị hình ảnh
    #
    # # Ví dụ hiển thị một hình ảnh từ dataset
    # images, labels = dataset[10]
    # print(f"Label: {labels}")
    # for image in images:
    #     show_image(image)