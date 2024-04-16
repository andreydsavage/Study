import torch
from torchvision.models import resnet18
import torchvision.transforms as T
import json


# Данные для нормализации
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# Загружаем классы 
def load_classes():
    """
    Returns IMAGENET classes and indexes
    """
    with open('utils/imagenet-simple-labels.json') as f:
        labels = json.load(f)
    return labels

# Функция для того чтобы достать название класса
def class_id_to_label(i):
    """
    :param i: class index
    :return: class name
    """
    labels = load_classes()
    return labels[i]

# Загружаем модель Resnet18, веса расположены локально
def load_model():
    """
    :return: resnet model with IMAGENET weights
    """
    model = resnet18()
    model.load_state_dict(torch.load('utils/resnet18-weights.pth', map_location='cpu'))
    model.eval()
    return model

# Преобразуем изображение
def transform_image(img):
    """
    :param img: PIL img
    :return: transformed img
    """
    trnsfrms = T.Compose(
        [
            T.Resize((224, 224)),
            T.CenterCrop(100),
            T.ToTensor(),
            T.Normalize(mean, std)
        ]
    )
    return trnsfrms(img)
