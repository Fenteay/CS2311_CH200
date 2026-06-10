import os

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_ids,
        img_dir,
        mask_dir,
        img_ext,
        mask_ext,
        num_classes,
        transform=None,
        missing_mask_strategy='error',
    ):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.missing_mask_strategy = missing_mask_strategy

    def _resolve_sample(self, sample):
        if isinstance(sample, dict):
            img_id = sample.get('img_id') or sample.get('id')
            if img_id is None:
                raise KeyError('Sample dict must include img_id.')
            return {
                'img_id': str(img_id),
                'save_id': str(sample.get('save_id', img_id)),
                'img_dir': sample.get('img_dir', self.img_dir),
                'mask_dir': sample.get('mask_dir', self.mask_dir),
                'img_ext': sample.get('img_ext', self.img_ext),
                'mask_ext': sample.get('mask_ext', self.mask_ext),
            }

        img_id = str(sample)
        return {
            'img_id': img_id,
            'save_id': img_id,
            'img_dir': self.img_dir,
            'mask_dir': self.mask_dir,
            'img_ext': self.img_ext,
            'mask_ext': self.mask_ext,
        }

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        sample = self._resolve_sample(self.img_ids[idx])
        img_id = sample['img_id']

        img_path = os.path.join(sample['img_dir'], img_id + sample['img_ext'])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")

        mask = []
        for i in range(self.num_classes):
            mask_path = os.path.join(sample['mask_dir'], str(i), img_id + sample['mask_ext'])
            mask_i = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_i is None:
                if self.missing_mask_strategy == 'zeros':
                    mask_i = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                else:
                    raise FileNotFoundError(f"Mask not found or unreadable: {mask_path}")
            else:
                # Normalize any non-zero foreground label (e.g., 38, 255) to binary mask.
                mask_i = (mask_i > 0).astype(np.uint8) * 255
            mask.append(mask_i[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': sample['save_id']}
