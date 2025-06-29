import plotly.express as px
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.Compose([
                                               transforms.Resize((32, 32)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.1307,), std=(0.3081,))]),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.Compose([
                                              transforms.Resize((32, 32)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=(0.1325,), std=(0.3105,))]),
                                          download=True)

batch_size = 64
num_classes = 10
learning_rate = 0.01
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def show_image(img_tensor, mean=0.1307, std=0.3081):
    """
    Display a grayscale image using Plotly Express from a normalized tensor.

    Args:
        img_tensor (Tensor): A tensor of shape (1, H, W) or (H, W) with normalized pixel values.
        mean (float): Mean used for normalization.
        std (float): Std used for normalization.
    """
    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.clone().detach()

    # De-normalize
    img_tensor = img_tensor * std + mean

    # Convert to numpy 2D array
    img_np = img_tensor.squeeze().numpy()

    # Show with Plotly
    fig = px.imshow(img_np, color_continuous_scale='gray')
    fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()


print("torch")
show_image(train_dataset[0][0])


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu1(out)
        out = self.fc2(out)

        return out


model = LeNet5(num_classes).to(device)
# setting the loss function
cost = nn.CrossEntropyLoss()

# setting the optimizer with the model parameters and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Press the green button in the gutter to run the script.

# this is defined to print how many steps are remaining when training
total_step = len(train_loader)
num_epochs = 10

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = cost(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 400 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                     loss.item()))

#
# if __name__ == '__main__':
#     print('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
