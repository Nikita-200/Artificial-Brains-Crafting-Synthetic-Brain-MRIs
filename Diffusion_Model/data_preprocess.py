import os
import nibabel as nib
import numpy as np
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings
import torch

def get_best_slice_index(seg_path):
    """
    Given a .nii segmentation file, return the slice index (along axial axis)
    that contains the most tumor pixels (i.e., non-zero mask values).
    """
    seg_nii = nib.load(seg_path)
    seg_data = seg_nii.get_fdata()  # shape: (H, W, D)

    # Count number of non-zero pixels in each slice along the 3rd axis
    tumor_pixel_counts = [(i, np.count_nonzero(seg_data[:, :, i])) for i in range(seg_data.shape[2])]
    
    # Find the slice index with the maximum non-zero pixel count
    best_slice_idx, max_count = max(tumor_pixel_counts, key=lambda x: x[1])

    return best_slice_idx

def get_dataloaders_from_jpeg(
    data_root="../BraTS_2023",
    archive_root="../archive",
    batch_size=8,
    image_size=64,
    num_workers=2,
    slice_idx=100,
    max_patients=None ####################################################################
):
    # ------------ TRANSFORM ------------
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])

    # ------------ PAIRED DATASET CLASS ------------
    class BraTSPairedDataset(Dataset):
        def __init__(self, paired_paths, transform=None):
            self.paired_paths = []
            # Filter out invalid paths during initialization
            for t1c_path, seg_path in paired_paths:
                if os.path.exists(t1c_path) and os.path.exists(seg_path):
                    self.paired_paths.append((t1c_path, seg_path))
                else:
                    warnings.warn(f"Skipping missing pair: {t1c_path} | {seg_path}")
            # self.paired_paths = paired_paths
            self.transform = transform

        def __len__(self):
            return len(self.paired_paths)

        def __getitem__(self, index):
            t1c_path, seg_path = self.paired_paths[index]
            try:
                best_slice = get_best_slice_index(seg_path)
                # print(f"📌 Best slice index for {seg_path} with most tumor: {best_slice}")
                t1c_img = nib.load(t1c_path).get_fdata()[:, :, best_slice]
                seg_img = nib.load(seg_path).get_fdata()[:, :, best_slice]

                # if np.all(t1c_img == 0) or np.all(seg_img == 0):
                #     print(f"[Skip] Empty slice for: {t1c_path}")
                #     continue
                
                # Handle zero-division and preserve labels
                if np.all(seg_img == 0):  # Check for completely empty masks
                    warnings.warn(f"Empty segmentation mask: {seg_path}")
                    return None
                elif seg_img.max() == 0:
                    warnings.warn(f"Empty segmentation mask: {seg_path}")
                    return None  # Skip this sample
                # if seg_img.max() == 0:
                #     seg_img = np.zeros_like(seg_img)  # Create blank mask
                else:
                    seg_img = seg_img / seg_img.max() * 255  
                    # seg_img = np.round(seg_img)
                    # seg_img = np.round(seg_img / seg_img.max() * 255)  # Preserve 0/1 labels
                
                seg_img = Image.fromarray(seg_img.astype(np.uint8))  # No scaling to 255


                t1c_img = Image.fromarray(np.uint8(t1c_img / t1c_img.max() * 255))
                # seg_img = Image.fromarray(np.uint8(seg_img / seg_img.max() * 255)) #.convert("L")

                if self.transform:
                    t1c_img = self.transform(t1c_img)
                    seg_img = self.transform(seg_img)

                return t1c_img, seg_img

            except Exception as e:
                print(f"[Error] Failed to process {t1c_path}, {seg_path}: {e}")
                raise e

    # ------------ COLLECT PAIRED PATHS ------------
    paired_paths = []
    # for folder in os.listdir(archive_root): # for testing
    #     folder_path = os.path.join(archive_root, folder)
    #     if not os.path.isdir(folder_path):
    #         continue

    #     files = os.listdir(folder_path)
    #     t1c_files = [f for f in files if "t1n" in f.lower() and f.endswith((".nii", ".nii.gz"))] ############### CHANGE
    #     seg_files = [f for f in files if "seg" in f.lower() and f.endswith((".nii", ".nii.gz"))]

    #     if t1c_files and seg_files:
    #         t1c_path = os.path.join(folder_path, t1c_files[0])
    #         seg_path = os.path.join(folder_path, seg_files[0])

    #         # Check if files are accessible before adding
    #         if (os.path.isfile(t1c_path) and os.path.getsize(t1c_path) > 0 and os.path.isfile(seg_path) and os.path.getsize(seg_path) > 0):
    #         # if os.path.isfile(t1c_path) and os.path.isfile(seg_path):
    #             paired_paths.append((t1c_path, seg_path))
    #         else:
    #             print(f"⚠️ Skipping invalid paths: {t1c_path} | {seg_path}")
    #         # paired_paths.append((t1c_path, seg_path))

    for folder in os.listdir(data_root): # for training
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            continue

        files = os.listdir(folder_path)
        t1c_files = [f for f in files if "t1n" in f.lower() and f.endswith((".nii", ".nii.gz"))] ############### CHANGE
        seg_files = [f for f in files if "seg" in f.lower() and f.endswith((".nii", ".nii.gz"))]

        if t1c_files and seg_files:
            t1c_path = os.path.join(folder_path, t1c_files[0])
            seg_path = os.path.join(folder_path, seg_files[0])

            # Check if files are accessible before adding
            if (os.path.isfile(t1c_path) and os.path.getsize(t1c_path) > 0 and os.path.isfile(seg_path) and os.path.getsize(seg_path) > 0):
            # if os.path.isfile(t1c_path) and os.path.isfile(seg_path):
                paired_paths.append((t1c_path, seg_path))
            else:
                print(f"⚠️ Skipping invalid paths: {t1c_path} | {seg_path}")
            # paired_paths.append((t1c_path, seg_path))
    
    if max_patients:
        paired_paths = paired_paths[:max_patients]

    # ------------ SPLIT DATA ------------
    random.shuffle(paired_paths)
    n = len(paired_paths)
    print("Total pairs", n)
    train_pairs = paired_paths[:int(0.8 * n)]
    val_pairs = paired_paths[int(0.8 * n):]  
    # test_pairs = paired_paths

    # ------------ DATASETS AND LOADERS ------------
    train_dataset = BraTSPairedDataset(train_pairs, transform)
    val_dataset = BraTSPairedDataset(val_pairs, transform)
    # test_dataset = BraTSPairedDataset(test_pairs, transform)

    def collate_fn_filter_none(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None  # Return None for empty batches to handle later
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn_filter_none )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_filter_none )
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_filter_none )

    print(f"✅ Dataset sizes — Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    # print(f"✅ Dataset sizes — Test: {len(test_dataset)}")
    return train_loader, val_loader
    # train_loader, val_loader
    #, test_loader
