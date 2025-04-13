from PIL import Image
import os
from torch.utils.data import DataLoader, Dataset


class PairedDataset(Dataset):
    def __init__(self, face_dir, comic_dir, transform=None):
        self.face_paths = [os.path.join(face_dir, f) for f in os.listdir(face_dir) if f.endswith(('.jpg', '.png'))]
        self.comic_paths = [os.path.join(comic_dir, f) for f in os.listdir(comic_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return max(len(self.face_paths), len(self.comic_paths))

    def __getitem__(self, idx):
        # Use modulo indexing in case datasets are of different sizes
        face_img = Image.open(self.face_paths[idx % len(self.face_paths)]).convert('RGB')
        comic_img = Image.open(self.comic_paths[idx % len(self.comic_paths)]).convert('RGB')
        if self.transform:
            face_img = self.transform(face_img)
            comic_img = self.transform(comic_img)
        return face_img, comic_img