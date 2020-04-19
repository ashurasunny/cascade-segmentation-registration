from torch.utils.data import Dataset
import os, cv2
import numpy as np
import torch
# from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import albumentations as A
import scipy.io
from utils import rescale_intensity, data_augmenter, convert_to_one_hot

def crop_img(imgs):
    crop_list = []
    for data in imgs:
        img, mask = data
        img = cv2.imread(img)
        mask = scipy.io.loadmat(mask)['inst_map']
        mask[mask>0] = 1
        w,h = img.shape[:2]
        window_size = 256

        img = img / 255.0
        if w == 500:
            c_startw = (w - window_size) // 2
            c_starth = (h - window_size) // 2
            img = img.transpose([2,0,1])

            # 5 crop,
            crop1 = img[:, :window_size, :window_size]
            crop2 = img[:, -window_size:, :window_size]
            crop3 = img[:, :window_size:, -window_size:]
            crop4 = img[:, -window_size:, -window_size:]
            crop5 = img[:, c_starth:c_starth + window_size, c_startw:c_startw + window_size]

            gt_crop1 = mask[:window_size, :window_size]
            gt_crop2 = mask[-window_size:, :window_size]
            gt_crop3 = mask[:window_size:, -window_size:]
            gt_crop4 = mask[-window_size:, -window_size:]
            gt_crop5 = mask[c_starth:c_starth + window_size, c_startw:c_startw + window_size]

            crop_list.append([crop1, gt_crop1])
            crop_list.append([crop2, gt_crop2])
            crop_list.append([crop3, gt_crop3])
            crop_list.append([crop4, gt_crop4])
            crop_list.append([crop5, gt_crop5])
        else:
            c_startw = (w - window_size) // 2
            c_starth = (h - window_size) // 2
            img = img.transpose([2, 0, 1])

            # 5 crop,
            crop1 = img[:, :window_size, :window_size]
            crop2 = img[:, -window_size:, :window_size]
            crop3 = img[:, :window_size:, -window_size:]
            crop4 = img[:, -window_size:, -window_size:]
            crop5 = img[:, c_starth:c_starth + window_size, c_startw:c_startw + window_size]

            crop6 = img[:, c_starth:c_starth + window_size, :window_size]
            crop7 = img[:, -window_size:, c_startw:c_startw + window_size]
            crop8 = img[:, :window_size:, c_startw:c_startw + window_size]
            crop9 = img[:, c_starth:c_starth + window_size:, -window_size:]

            gt_crop1 = mask[:window_size, :window_size]
            gt_crop2 = mask[-window_size:, :window_size]
            gt_crop3 = mask[:window_size:, -window_size:]
            gt_crop4 = mask[-window_size:, -window_size:]
            gt_crop5 = mask[c_starth:c_starth + window_size, c_startw:c_startw + window_size]

            gt_crop6 = mask[c_starth:c_starth + window_size, :window_size]
            gt_crop7 = mask[-window_size:, c_startw:c_startw + window_size]
            gt_crop8 = mask[:window_size:, c_startw:c_startw + window_size]
            gt_crop9 = mask[c_starth:c_starth + window_size:, -window_size:]



            crop_list.append([crop1, gt_crop1])
            crop_list.append([crop2, gt_crop2])
            crop_list.append([crop3, gt_crop3])
            crop_list.append([crop4, gt_crop4])
            crop_list.append([crop5, gt_crop5])
            crop_list.append([crop6, gt_crop6])
            crop_list.append([crop7, gt_crop7])
            crop_list.append([crop8, gt_crop8])
            crop_list.append([crop9, gt_crop9])

    return crop_list

def make_dataset(root1, root2):
    imgs = []
    n = len(os.listdir(root1))
    for f in os.listdir(root1):
        img = os.path.join(root1, f)
        mask = os.path.join(root2, f.replace('png', 'mat'))
        test = cv2.imread(img)
        print(test.shape)
        imgs.append((img, mask))
    return imgs


class CPM17Dataset(Dataset):
    def __init__(self, root, aug=True):
        root_image = os.path.join(root, 'Images')
        root_label = os.path.join(root, 'Labels')
        imgs = make_dataset(root_image, root_label)
        imgs = crop_img(imgs)
        self.imgs = imgs
        self.augment=aug

    def __getitem__(self, index):
        img_x, img_y = self.imgs[index]
        shift = 10
        rotate = 15
        scale = 0.2
        intensity = 0.1
        flip = True



        if self.augment:
            aug = A.Compose([
                A.ElasticTransform(alpha=200, sigma=200 * 0.05, alpha_affine=200 * 0.03, p=0.4),
                # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50),
                # A.RandomBrightnessContrast(),
                # A.RandomGamma(),
                # A.CLAHE(),
                # A.RGBShift(p=0.2),
                # A.Blur(blur_limit=3),
                # A.GaussNoise(p=0.2),
                # A.Flip(),
                # A.RandomRotate90(),

            ])
            img_x  = img_x.transpose([1,2,0])
            augmented = aug(image=img_x, mask=img_y)
            img_x = augmented['image']
            img_y = augmented['mask']
            img_x = img_x.transpose([2, 0, 1])

        img_x = np.expand_dims(img_x, axis=0)
        img_x = img_x.transpose([0,2,3,1])
        img_y = np.expand_dims(img_y, axis=0)
        if self.augment:
            img_x, img_y = data_augmenter(img_x, img_y, shift=shift, rotate=rotate,
                                            scale=scale, intensity=intensity, flip=flip)

        # img_x = np.expand_dims(img_x, axis=0)
        # img_x = img_x.transpose([0, 2, 3, 1])
        # img_y = np.expand_dims(img_y, axis=0)

        labels_onehot = convert_to_one_hot(img_y).astype(np.float32)
        labels_onehot = labels_onehot.transpose([1, 0, 2, 3])


        M = img_x.copy()
        M[img_y == 0] = 0
        # cv2.imwrite('./test.png', M.squeeze()*255)
        M = M.transpose([0, 3, 1, 2])
        M = torch.from_numpy(M).float()

        img_x = img_x.transpose([0, 3, 1, 2])
        img_x = torch.from_numpy(img_x).float()
        img_y = torch.from_numpy(labels_onehot)
        return img_x.squeeze(), img_y.squeeze(), M.squeeze()

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    # imgs = make_dataset('/data/private/xxw993/data/cpm17/train/Images', '/data/private/xxw993/data/cpm17/train/Labels')
    # crop_img(imgs)
    batch_size = 10
    num_workers = 0
    train_dataset = CPM17Dataset("/data/private/xxw993/data/cpm17/train/")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #
    for iteration in range(10):
        sample = train_loader.__iter__()
        inputs, labels = sample

        print(iteration, inputs.size(), labels.size())






