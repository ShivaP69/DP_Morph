#!/usr/bin/python
#
# Copyright 2022 Azade Farshad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as tf
import torchvision.transforms as transforms
import os
from typing import Callable
from torch.utils.data.distributed import DistributedSampler
from torch.utils import data
from PIL import Image
from sklearn.model_selection import train_test_split


def get_files(path, ext=".npy"):
    return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(ext)]


class split_and_copy_victim_only():
    def __init__(self,source_train_dir, source_val_dir):
        # List files in train/images
        train_files = get_files(os.path.join(source_train_dir, 'images'))
        train_mask_files = get_files(os.path.join(source_train_dir, 'masks'))
        train1 = train_files
        mask1_train = train_mask_files

        # List files in val/images
        val_files = get_files(os.path.join(source_val_dir, 'images'))
        mask_val_files = get_files(os.path.join(source_val_dir, 'masks'))
        val1 = val_files
        mask1_val = mask_val_files

        self.victim_train_paths = {'imgs': train1, 'lbls': mask1_train}
        self.victim_val_paths = {'imgs': val1, 'lbls': mask1_val}

        # the following codes are used for calculating loss distribution of target the model
        self.victim_attack_paths_member = {
            'imgs': self.victim_train_paths['imgs'],
            'lbls': self.victim_train_paths['lbls'],
            'member': np.ones(len(self.victim_train_paths['imgs']))
        }
        self.victim_attack_paths_non_member = {
            'imgs': self.victim_val_paths['imgs'],
            'lbls': self.victim_val_paths['lbls'],
            'member': np.zeros(len(self.victim_val_paths['imgs']))
        }




class split_and_copy():
    def __init__(self,source_train_dir, source_val_dir):
        # List files in train/images
        train_files = get_files(os.path.join(source_train_dir, 'images'))
        train_mask_files = get_files(os.path.join(source_train_dir, 'masks'))

        # Split train into two halves
        split_index = len(train_files) // 2
        #split_index= len(train_files)
        train1 = train_files[:split_index]
        train2 = train_files[split_index:]
        mask1_train= train_mask_files[:split_index]
        mask2_train=train_mask_files[split_index:]


        # List files in val/images
        val_files = get_files(os.path.join(source_val_dir, 'images'))
        mask_val_files=get_files(os.path.join(source_val_dir, 'masks'))

        # Split val into two halves
        split_index = len(val_files) // 2 # first time
        #split_index= len(val_files)
        val1 = val_files[:split_index]
        val2 = val_files[split_index:]
        mask1_val= mask_val_files[:split_index]
        mask2_val=mask_val_files[split_index:]

        k= split_index//2 # second time


        # use dataset without any provided label


        self.victim_train_paths= {'imgs':train1, 'lbls':mask1_train}
        self.victim_val_paths= {'imgs':val1, 'lbls':mask1_val}

        self.shadow_train_paths= {'imgs':train2, 'lbls':mask2_train}
        self.shadow_val_paths= {'imgs':val2, 'lbls':mask2_val}

        self.victim_attack_paths = {
            'imgs': np.concatenate([self.victim_train_paths['imgs'][:k], self.victim_val_paths['imgs'][:k]]),
            'lbls': np.concatenate([self.victim_train_paths['lbls'][:k], self.victim_val_paths['lbls'][:k]]),
            'member': np.concatenate([np.ones((k)), np.zeros((k))])
        }

        self.shadow_attack_paths = {
            'imgs': np.concatenate([self.shadow_train_paths['imgs'][:k], self.shadow_val_paths['imgs'][:k]]),
            'lbls': np.concatenate([self.shadow_train_paths['lbls'][:k], self.shadow_val_paths['lbls'][:k]]),
            'member': np.concatenate([np.ones((k)), np.zeros((k))])
        }
        # the following codes are used for calculating loss distribution of target the model
        self.victim_attack_paths_member = {
            'imgs':self.victim_train_paths['imgs'],
            'lbls':self.victim_train_paths['lbls'],
            'member': np.ones(len(self.victim_train_paths['imgs']))
        }
        self.victim_attack_paths_non_member = {
            'imgs': self.victim_val_paths['imgs'],
            'lbls': self.victim_val_paths['lbls'],
            'member': np.zeros(len(self.victim_val_paths['imgs']))
        }

        self.shadow_attack_paths_non_member = {
            'imgs': np.concatenate([self.shadow_train_paths['imgs'], self.shadow_val_paths['imgs']]),
            'lbls': np.concatenate([self.shadow_train_paths['lbls'], self.shadow_val_paths['lbls']]),
            'member': np.concatenate([
                np.zeros(len(self.shadow_train_paths['imgs'])),
                np.zeros(len(self.shadow_val_paths['imgs']))
            ])
        }

        print('************************')
        print('Victim train paths:', len(self.victim_train_paths['imgs']))
        print('Shadow train paths:', len(self.shadow_train_paths['imgs']))
        print('Attack train paths:', len(self.shadow_attack_paths['imgs']))
        print('Attack val paths:', len(self.victim_attack_paths['imgs']))
        print('************************')




# change k!
class split_and_copy_synthetic():
    def __init__(self,source_train_dir, source_val_dir,DME=True,DME7=False):

        # List files in train/images
        train_files = get_files(os.path.join(source_train_dir, 'images'))
        train_mask_files = get_files(os.path.join(source_train_dir, 'masks'))
        k=len(train_files) // 2

        train1 = train_files
        mask1_train= train_mask_files

        # List files in val/images
        val_files = get_files(os.path.join(source_val_dir, 'images'))
        mask_val_files=get_files(os.path.join(source_val_dir, 'masks'))
        val1 = val_files
        mask1_val= mask_val_files


        # use dataset without any provided label
        # read extra datasette
        if DME: # Kermanni dataset
            dme_images= get_files('DME', ext='.jpeg')
        elif DME7: # Srini dataset
            dme_images = get_files('DME7', ext='.tif')

            print("***Reading TIF dataset***")


        self.victim_train_paths = {'imgs': train1, 'lbls': mask1_train}
        self.victim_val_paths = {'imgs': val1, 'lbls': mask1_val}
        self.external_data={'imgs': dme_images}

        print(f"victim train paths:{len(self.victim_train_paths['imgs'])}")
        print(f"victim val paths:{len(self.victim_val_paths['imgs'])}")
        print(f"external_data paths:{len(self.external_data['imgs'])}")
        self.victim_attack_paths = {
            'imgs': np.concatenate([self.victim_train_paths['imgs'][:k], self.victim_val_paths['imgs'][:k]]),
            'lbls': np.concatenate([self.victim_train_paths['lbls'][:k], self.victim_val_paths['lbls'][:k]]),
            'member': np.concatenate([np.ones((k)), np.zeros((k))])
        }


        # the following codes are used for calculating loss distribution of target the model
        self.victim_attack_paths_member = {
            'imgs': self.victim_train_paths['imgs'],
            'lbls': self.victim_train_paths,
            'member': np.ones(len(self.victim_train_paths['imgs']))
        }
        self.victim_attack_paths_non_member = {
            'imgs': self.victim_val_paths['imgs'],
            'lbls': self.victim_val_paths['lbls'],
            'member': np.zeros(len(self.victim_val_paths['imgs']))
        }




class shadow_split_and_copy_synthetic():
    def __init__(self,source_data):
        images_files=get_files(os.path.join(source_data, 'images'))
        masks_files=get_files(os.path.join(source_data, 'masks'))
        train_images, val_images, train_masks, val_masks = train_test_split(images_files, masks_files, test_size=0.2,
                                                                       random_state=42)
        self.shadow_train_paths = {'imgs': train_images, 'lbls': train_masks}
        self.shadow_val_paths = {'imgs': val_images, 'lbls': val_masks}
        #k=len(train_images) // 2
        k = min(len(train_images), len(val_images))
        self.shadow_attack_paths = {
            'imgs': np.concatenate([self.shadow_train_paths['imgs'][:k], self.shadow_val_paths['imgs'][:k]]),
            'lbls': np.concatenate([self.shadow_train_paths['lbls'][:k], self.shadow_val_paths['lbls'][:k]]),
            'member': np.concatenate([np.ones((k)), np.zeros((k))])
        }
        self.shadow_attack_paths_training_attack_model = {
            'imgs': np.concatenate([self.shadow_train_paths['imgs'][:k]]),
            'lbls': np.concatenate([self.shadow_train_paths['lbls'][:k]]),
            'member': np.concatenate([np.ones((k))])
        }

        print("shadow_split_and_copy_synthetic imgs",len(self.shadow_attack_paths['imgs']))
        print("shadow_split_and_copy_synthetic member", len(self.shadow_attack_paths['member']))

class combine_attack_data():
    def __init__(self,data1,data2):
        self.victim_attack_paths=data1.victim_attack_paths # k number of victim_train_paths and k number of victim_val_paths
        #self.shadow_attack_paths=data2.shadow_attack_paths # it is from external data
        self.shadow_attack_paths=data2.shadow_attack_paths_training_attack_model

        print(
            f"Victim Attack Paths - Ones: {np.sum(self.victim_attack_paths['member'] == 1)}, Zeros: {np.sum(self.victim_attack_paths['member'] == 0)}")
        print(
            f"Shadow Attack Paths - Ones: {np.sum(self.shadow_attack_paths['member'] == 1)}, Zeros: {np.sum(self.shadow_attack_paths['member'] == 0)}")


"""class TransformOCTBilinear(object):
    def __new__(cls, img_size=(128, 128), *args, **kwargs):
        return tf.Compose([
            tf.Resize(img_size)
        ])"""


class TransformOCTBilinear(object):
    def __init__(self, img_size=(128, 128), n_channels=None):
        self.img_size = img_size
        self.n_channels = n_channels
        self.resize_transform = transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR,antialias=True)

    def __call__(self, img):
        # Check and handle the number of channels
        if img.mode == 'RGBA':  # Assuming img is a PIL.Image
            # Convert RGBA to RGB by discarding the alpha channel
            img = img.convert('RGB')

        # Apply resizing
        img = self.resize_transform(img)

        # Optionally adjust the number of channels if specifically required
        if self.n_channels == 1:
            # Convert to grayscale (this would now operate on RGB images)
            img = transforms.Grayscale(num_output_channels=1)(img)
        elif self.n_channels == 3 and img.mode != 'RGB':
            # Ensure it is in RGB format
            img = img.convert('RGB')

        return img


class TransformStandardization(object):
    """
    Standardizaton / z-score: (x-mean)/std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return (image - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__ + f": mean {self.mean}, std {self.std}"

class TransformOCTMaskAdjustment_Separate(object):
    """
    Adjust OCT 2015 Mask
    from: classes [0,1,2,3,4,5,6,7,8,9], where 9 is fluid, 0 is empty space above and 8 empty space below
    to: class 0: not class, classes 1-7: are retinal layers, class 8: fluid
    """

    def __call__(self, mask,i):
        #mask = (mask == i).astype(int)
        #mask[mask == 8] = 0
        #mask[mask == 9] = 8
        mask[mask != i] = 0
        mask[mask == i] = 1
        return mask

class TransformOCTMaskAdjustment(object):
    """
    Adjust OCT 2015 Mask
    from: classes [0,1,2,3,4,5,6,7,8,9], where 9 is fluid, 0 is empty space above and 8 empty space below
    to: class 0: not class, classes 1-7: are retinal layers, class 8: fluid
    """

    def __call__(self, mask):
        mask[mask == 8] = 0
        mask[mask == 9] = 8
        return mask


class DatasetOct(Dataset):

    def __init__(self, data, size_transform=None, normalized=True, attack=False,image_transform=None,external=False,label_index=-1) -> None:


        self.img_paths = data['imgs']

        if not external:
            self.lbl_paths = data['lbls']
        self.size_transform = TransformOCTBilinear(img_size=(224, 224),n_channels=1)
        self.normalized = normalized
        self.attack = attack
        self.image_transform = image_transform
        self.label_index = label_index
        if self.label_index>=0:
            self.mask_adjust_one = TransformOCTMaskAdjustment()
            self.mask_adjust_second= TransformOCTMaskAdjustment_Separate()

        else:
            self.mask_adjust_one = TransformOCTMaskAdjustment()


        self.normalize = TransformStandardization(46.3758, 53.9434)
        if attack:
            self.membership = data['member']
            unique, counts = np.unique(self.membership, return_counts=True)
            print(f"Membership distribution: {dict(zip(unique, counts))}")  # Check distribution of membership labels
        """if not external:  # Only check lbl_paths if they are relevant
            assert len(self.img_paths) == len(self.lbl_paths), "Mismatch between img_paths and lbl_paths lengths"
        if attack:  # Only check membership if it is relevant
            print(len(self.img_paths), len(self.membership))
            assert len(self.img_paths) == len(self.membership), "Mismatch between img_paths and membership lengths"""


            #print("members 1: ",len(data['member']==1))
            #print("member 0: ",len(data['member']==0))
        #print(f"number of classes:{n_classes}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        img_path = (self.img_paths[idx])
        """if self.attack:
            print(f"Index: {idx}, Membership: {self.membership[idx]}")"""
        # in the current version, our jpeg and tif dataset does not have masks then I should handle it by for example if self.lbl.paths[idx]:
        # and the rest should be handled based on the type of masks
        if img_path.lower().endswith('.jpg') or img_path.lower().endswith('.jpeg') or img_path.lower().endswith('.tif'):
            img= Image.open(img_path)
            #img = transforms.ToTensor()(img)  # Convert PIL image to tensor
            img = np.array(img)
            img = torch.Tensor(img).reshape(1, 1, *img.shape)
            img = self.size_transform(img)
            #img.reshape(1,1,*img.shape)
            #img=self.size_transform(img)
            #print("DME image size:", img.shape)


        elif img_path.lower().endswith('.npy'): # we already have dataset
            lbl_path = (self.lbl_paths[idx])
            img = np.load(img_path)
            mask = np.load(lbl_path)


            # img_size 128 works - general transforms require (N,C,H,W) dims
            img = img.squeeze()
            mask = mask.squeeze()

            img = torch.Tensor(img).reshape(1, 1, *img.shape)
            mask = torch.Tensor(mask).reshape(1, 1, *mask.shape).int()

            # adjust mask classes
            mask= self.mask_adjust_one(mask)
            if self.label_index>=0:
                mask = self.mask_adjust_second(mask,self.label_index)

            # note for some reason some masks differ in size from the actual image (dims)
            if self.size_transform:
                img = self.size_transform(img)
                mask = self.size_transform(mask)
                #print("npy image size:", img.shape)

        # normalize after size_transform
        if self.normalized:
            img = self.normalize(img)

        """if self.joint_transform:
            img, mask = self.joint_transform([img, mask])"""

        # img = img.reshape(1, img.shape[2], img.shape[3])
        if self.image_transform:
            img = self.image_transform(img)

        # img = img.reshape(1, *img.shape)

        # set image dim to (C,H,W)
        img = img.squeeze()
        img = img.reshape(1, *img.shape)
        #print("npy image size:", img.shape)

        if img_path.lower().endswith('.npy'):
            # set mask dim to (H,W)  where value at (h,w) maps to class of corresponding pixel
            mask = mask.squeeze(dim=1).long()
            if self.attack:
                member = torch.tensor(self.membership[idx])
                return img, mask, member
            return img, mask
        else:
            return img



# We can use this code to check if everything works correctly
# test the functionality of data handling
"""main_dir = 'DukeData'
data= split_and_copy(os.path.join(main_dir, 'train'), os.path.join(main_dir, 'val'))


victim_train= DatasetOct(data.victim_attack_paths,attack=True)

victim_train_dataloader = DataLoader(victim_train, batch_size=8, shuffle=True)
for batch_idx, (imgs, masks,member) in enumerate(victim_train_dataloader):
    print(f"Batch {batch_idx + 1}:")
    print(f"Images shape: {imgs.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Members shape: {member.shape}")
    print(f"memeber: {member}")
    print(f"Images: {imgs}")
    print(f"Masks: {masks}")
    break  # Only print the first batch
"""

# for split and copy synthetic
"""main_dir = 'DukeData'
data= split_and_copy_synthetic(os.path.join(main_dir, 'train'), os.path.join(main_dir, 'val'))

shadow_model=DatasetOct(data.external_data,external=True)

shadow_dataloader = DataLoader(shadow_model, batch_size=8, shuffle=True)
for batch_idx, (imgs) in enumerate(shadow_dataloader):
    print(f"Batch {batch_idx + 1}:")
    print(f"Images shape: {imgs.shape}")


main_dir = 'DukeData'
data= split_and_copy_synthetic(os.path.join(main_dir, 'train'), os.path.join(main_dir, 'val'))
victim_train= DatasetOct(data.victim_attack_paths,attack=True)
victim_train_dataloader = DataLoader(victim_train, batch_size=8, shuffle=True)
for batch_idx, (imgs, masks,member) in enumerate(victim_train_dataloader):
    print(f"Batch {batch_idx + 1}:")
    print(f"Images shape: {imgs.shape}")
"""