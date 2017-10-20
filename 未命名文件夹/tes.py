from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

im = Image.open("/home/orashi/PycharmProjects/PaintsPytorch/41067384_p0.jpg")
from torchsample.transforms import RandomRotate
im.show()
rotation = RandomRotate(60)
ToPILImage()(rotation(ToTensor()(im))).show()
