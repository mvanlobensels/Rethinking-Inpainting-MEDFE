import torch
from torch.utils.data import DataLoader
from piq import FID, psnr, ssim
from glob import glob
import torchvision.transforms as transforms
from PIL import Image

class DataProcess(torch.utils.data.Dataset):

    def __init__(self, img_root):
        super(DataProcess, self).__init__()
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.img_paths = sorted(glob('{:s}/*'.format(img_root), recursive=True))
        print(self.img_paths)

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        img = self.img_transform(img.convert('RGB'))
        return img

    def __len__(self):
        return len(self.img_paths)

target_images_root = './result/test_set'
predicted_images_root = './result/Mutual-Encode-Decoder_6000_images_1_epoch_alternating_branches'

target_data = DataLoader(DataProcess(target_images_root), batch_size=1, num_workers=4)
predicted_data = DataLoader(DataProcess(predicted_images_root), batch_size=1, num_workers=4)

fid_metric = FID()
target_features = fid_metric.compute_feats(target_data)
predicted_features = fid_metric.compute_feats(predicted_data)

fid = fid_metric(predicted_data, target_data)
print(fid)