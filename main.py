import plotly.express as px
import torch
import torchvision
import torchvision.transforms as transforms

train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.Compose([
                                               transforms.Resize((32, 32)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.1307,), std=(0.3081,))]),
                                           download=True)


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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
