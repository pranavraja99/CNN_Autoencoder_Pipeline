#Just an eval script to check how well the input is reconstructed by the decoder
#%%
import torch
import torchvision
from dataset import FaceDataset, get_transform
from narch import auto_encoder
import cv2
import albumentations as A
import torchvision.transforms as tra
import matplotlib.pyplot as plt

model=auto_encoder()
model.load_state_dict(torch.load('./experiment_lr01/best_model.pth'))

img=cv2.imread('./val/35.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



aug=A.Cutout(num_holes=1, max_h_size=200, max_w_size=200, always_apply=True)(image=img)
img=aug['image']


# %%
model.eval()

faces=FaceDataset('./val/', get_transform(True))
im,tar=faces[8]
inp=torch.unsqueeze(im,0)
plt.imshow(tra.ToPILImage()(tar))
#%%
plt.imshow(tra.ToPILImage()(model(inp)[0]))

# %%
