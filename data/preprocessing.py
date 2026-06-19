
"""
preprocessing.py 

It contains the function that perform the initial preprocessing: 
- Shape resize to 256 * 256, dimension necessary for the generation method 
- Normalization between [0., 1.]

"""
from torchvision import transforms

def get_transform(image_size=256):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),        
        transforms.ToTensor() #returns a tensor like [channel, H, W] with values [0.,1.]
        ])

    return transform