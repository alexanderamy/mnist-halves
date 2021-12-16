from torchvision import transforms

def split_img(img):
    x = transforms.functional.crop(img, 0, 0, 14, 28)   # top half
    y = transforms.functional.crop(img, 14, 0, 14, 28)  # bottom half
    return x, y

ToPILImage = transforms.ToPILImage()
ToTensor = transforms.ToTensor()