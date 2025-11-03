
from args import *
from data import DatasetOct
from torch.utils.data import DataLoader
from train import seg_train
from attack_train import train_attack_model,train_attack_model_patchify,test_attack_model
from train import train_synthetic
from torch.utils.data import ConcatDataset, Subset
from args import OUTPUT_CHANNELS
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# this is a function to change target for those sample that we reserve for victim to be added for testing the shadow model
class FixedTargetWrapper(Dataset):
    def __init__(self, dataset, fixed_target):
        self.dataset = dataset
        self.fixed_target = fixed_target

    def __getitem__(self, index):
        data, label, _ = self.dataset[index]
        return data, label, torch.tensor(self.fixed_target)

    def __len__(self):
        return len(self.dataset)





def create_overlay(img, mask, layer_colors):
    """
    Create an overlay of the mask on the image using specified layer colors.

    Args:
        img (np.ndarray): The image array (H, W, C).
        mask (np.ndarray): The mask array (H, W).
        layer_colors (dict): Dictionary mapping mask values to RGB colors.

    Returns:
        np.ndarray: The image with mask overlay.
    """
    # Ensure the mask is of integer type for indexing

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)  # Remove the singleton dimension

    # Initialize the overlay with the same height and width as the image, but with 3 color channels
    overlay = np.zeros((img.shape[0], img.shape[1], 3))

    # Apply the colors to the overlay
    for value, color in layer_colors.items():
        # Ensure value is in mask
        print(f"maks shape:{mask.shape}")
        overlay[mask == value] = color

    return overlay


def plot_images_with_masks(val_loader,layer, num_examples=1, layer_colors=None):
    """
    Plot images from a dataloader with corresponding masks overlaid.

    Args:
        val_loader (DataLoader): DataLoader providing batches of images and masks.
        num_examples (int): Number of examples to plot.
        layer_colors (dict): Dictionary mapping mask values to RGB colors.
    """
    batch_processed = 0

    if layer_colors is None:
        # Define default colors for each layer
        layer_colors = {
            0: (0, 0, 0),  # Black
            1: (1, 0, 0),  # Red
            2: (0, 1, 0),  # Green
            3: (0, 0, 1),  # Blue
            4: (1, 1, 0),  # Yellow
            5: (1, 0, 1),  # Magenta
            6: (0, 1, 1),  # Cyan
            7: (1, 0.5, 0),  # Orange
            8: (0.8, 0.7, 0.6)  # Light Brown
        }

    for imgs, masks in val_loader:
        for idx in range(imgs.size(0)):
            if batch_processed >= num_examples:
                break

            img = imgs[idx].numpy().transpose(1, 2, 0)  # Convert from CHW to HWC format
            mask = masks[idx].numpy()

            # Ensure image is 3-channel for consistency
            if img.shape[2] == 1:
                img = np.concatenate([img] * 3, axis=-1)

            # Normalize the original image
            min_val = img.min()
            max_val = img.max()
            img_normalized = (img - min_val) / (max_val - min_val)

            # Create overlay for mask
            mask_overlay = create_overlay(img, mask, layer_colors)

            # Combine original image with mask overlay
            combined_image = np.clip(0.7 * img_normalized + 0.3 * mask_overlay, 0, 1)

            # Plotting
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            axes[0].imshow(img_normalized)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(combined_image)
            axes[1].set_title(f'Image with Mask Overlay for layer {layer} ')
            axes[1].axis('off')

            plt.show()
            batch_processed += 1

        if batch_processed >= num_examples:
            break


def get_victim(data, args):
    victim_train = DatasetOct(data.victim_train_paths,args.n_classes)
    victim_train_dataloader = DataLoader(victim_train, batch_size=int(args.batch_size), shuffle=True)
    victim_val = DatasetOct(data.victim_val_paths,args.n_classes)
    victim_val_dataloader = DataLoader(victim_val, batch_size=int(args.batch_size))

    # no defense or argmax defense
    if (args.defensetype == 1) or (args.defensetype == 2) or (args.defensetype == 0):
        if args.DPSGD:
            print("DPSGD is active")
            args.DPSGD = True
            victim_model = seg_train(args, victim_train_dataloader, victim_val_dataloader,counter=None,victim=True,shadow=False)
            #args.DPSGD = False
            args.DPSGD_for_Saving_and_synthetic=True
            #print("DPSGD set to false but DPSGD_for_Saving_and_synthetic set to True")
            #assert not args.DPSGD, "Error: args.DPSGD was not properly set to False after the training!"
        else:
            print("DPSGD is not active")
            victim_model = seg_train(args, victim_train_dataloader, victim_val_dataloader, counter=None, victim=True,
                                    shadow=False,plotting=False)
    # crop training
    """elif args.defensetype == 3:
        victim_model = train_segmentation_model_crop(args, victim_train_dataloader, victim_val_dataloader, 3*SEG_EPOCHS, SEG_LR)
    # mix-up
    elif args.defensetype == 4:
        victim_model = train_segmentation_model_mix(args, victim_train_dataloader, victim_val_dataloader, SEG_EPOCHS, SEG_LR)"""

    return victim_model   # it is ofcourse two output : model, average_loss


def get_shadow(data, args):
    shadow_train = DatasetOct(data.shadow_train_paths)
    shadow_train_dataloader = DataLoader(shadow_train, batch_size=int(args.batch_size), shuffle=True)
    shadow_val = DatasetOct(data.shadow_val_paths)
    shadow_val_dataloader = DataLoader(shadow_val, batch_size=int(args.batch_size))
    print("shadow model training")
    shadow_model ,average_loss = seg_train(args, shadow_train_dataloader, shadow_val_dataloader,counter=None,victim=False,shadow=True,plotting=True)
    return shadow_model, average_loss


def check_balance_in_dataset(dataset):
    num_zeros = 0
    num_ones = 0

    for i in range(len(dataset)):
        # Access the ith sample in the dataset
        img, mask, member = dataset[i]  # Assuming attack=True, so member is returned

        if member == 1:
            num_ones += 1
        elif member == 0:
            num_zeros += 1

    print(f"Total samples: {len(dataset)}")
    print(f"Number of ones attack train (members = 1): {num_ones}")
    print(f"Number of zeros  attack train(members = 0): {num_zeros}")



def prepare_data_for_get_attack(data):
    # first define what is victim attack paths: combination of member and non-members
    # the test data of victim model should not considered as non-memeber in shadow (cheating)


    print(
        f"Victim Attack Paths inside get attack - Ones: {np.sum(data.victim_attack_paths['member'] == 1)}, Zeros: {np.sum(data.victim_attack_paths['member'] == 0)}")
    print(
        f"Shadow Attack Paths inside get attack - Ones: {np.sum(data.shadow_attack_paths['member'] == 1)}, Zeros: {np.sum(data.shadow_attack_paths['member'] == 0)}")

    print(f"Number of images before sending to DatasetOct: {len(data.shadow_attack_paths['imgs'])}")
    print(f"Number of membership labels before sending to DatasetOct: {len(data.shadow_attack_paths['member'])}")

    attack_train = DatasetOct(data.shadow_attack_paths, attack=True)
    check_balance_in_dataset(attack_train)
    attack_val = DatasetOct(data.victim_attack_paths, attack=True)
    # Allocate more data to shadow model (validation is not enough, because we are already sure that It has high performance over its validation set)
    validation_split = .4
    shuffle_dataset = True
    random_seed = 42
    dataset_size = len(attack_val)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    shadow_dataset_size= len(attack_train)
    indices_2 = list(range(shadow_dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    train_indices_2, val_indices_2 = indices_2[:split], indices_2[split:]

    """attack_val_dataloader =DataLoader(attack_val, batch_size=ATTACK_BATCH_SIZE,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),)""" # restricts the DataLoader to a subset of attack_val and shuffles it.

    print(f"Number of victim subset:{len(val_indices)}")
    # Subset of victim data to add to shadow data
    victim_subset = Subset(attack_val, train_indices)  #  picked for training attack model which is based on shadow model
    shadow_train_subet=Subset(attack_train,train_indices_2)
    shadow_val_subset= Subset(attack_train,val_indices_2)
    victim_attack_val = Subset(attack_val, val_indices)
    #training
    victim_with_fixed_target_used_for_shadow = FixedTargetWrapper(victim_subset, fixed_target=0)  # will be used for shadow
    victim_subset_lis=list(victim_with_fixed_target_used_for_shadow)
    shadow_train_subet_list=list(shadow_train_subet)
    shadow_ones=[x for x in shadow_train_subet_list if x[2].item() == 1]
    shadow_zeros = [x for x in victim_subset_lis if x[2].item() == 0]
    min_samples=min(len(shadow_ones),len(shadow_zeros)) # to have same numbers of 0s and 1s
    selected_shadow_ones=shadow_ones[:min_samples]
    selected_shadow_zeros=shadow_zeros[:min_samples]
    new_attack_train=selected_shadow_ones+selected_shadow_zeros
    #new_attack_train = torch.utils.data.ConcatDataset([victim_with_fixed_target_used_for_shadow, shadow_train_subet])
    attack_train_dataloader = DataLoader(new_attack_train, batch_size=ATTACK_BATCH_SIZE, shuffle=True)

    # validation
    shadow_val_subset_fixed_target = FixedTargetWrapper(shadow_val_subset, fixed_target=0) # will be used for victim
    #combined_val_subset=torch.utils.data.ConcatDataset([victim_attack_val,shadow_val_subset_fixed_target])
    # To have balanced validation set
    victim_attack_val_list = list(victim_attack_val)
    shadow_val_fixed_list = list(shadow_val_subset_fixed_target)
    victim_ones = [x for x in victim_attack_val_list if x[2].item() == 1] # x[2] because based on data.py , x[2] is target
    victim_zeros = [x for x in victim_attack_val_list if x[2].item() == 0]
    shadow_zeros = shadow_val_fixed_list
    min_zeros = min(len(victim_zeros), len(shadow_zeros)) # Find the minimum number of 0s we can include from both victim and shadow
    selected_victim_zeros = victim_zeros[:min_zeros]
    selected_shadow_zeros = shadow_zeros[:min_zeros]
    selected_zeros= selected_victim_zeros + selected_shadow_zeros
    min_zeros_ones=min(len(selected_zeros), len(victim_ones))
    combined_val_subset = victim_ones [:min_zeros_ones] + selected_zeros[:min_zeros_ones]
    attack_val_dataloader=DataLoader(combined_val_subset,batch_size=ATTACK_BATCH_SIZE,shuffle=True)

    # check number of ones and zeros
    num_zeros_train = 0
    num_ones_train = 0
    num_zeros_val = 0
    num_ones_val = 0
    # Count zeros and ones in the training DataLoader
    for batch_data, batch_labels, batch_targets in attack_train_dataloader:
        num_ones_train += torch.sum(batch_targets == 1).item()
        num_zeros_train += torch.sum(batch_targets == 0).item()

    # Count zeros and ones in the validation DataLoader
    for batch_data, batch_labels, batch_targets in attack_val_dataloader:
        num_ones_val += torch.sum(batch_targets == 1).item()
        num_zeros_val += torch.sum(batch_targets == 0).item()

    print(f"Number of ones in training set: {num_ones_train}")
    print(f"Number of zeros in training set: {num_zeros_train}")
    print(f"Number of ones in validation set: {num_ones_val}")
    print(f"Number of zeros in validation set: {num_zeros_val}")

    return attack_train_dataloader,attack_val_dataloader

def get_attack(attack_train_dataloader,attack_val_dataloader, args, victim_model, shadow_model):
    attack_model = train_attack_model(
        args, shadow_model, victim_model, attack_train_dataloader, attack_val_dataloader, ATTACK_EPOCHS, ATTACK_LR)
    return attack_model



def get_attack_different_shadow (args,data, victim_model, shadow_models):
    attack_models=[]
    aggregated_prediction_validation=[]
    print("Data Preparation")
    attack_train_dataloader,attack_val_dataloader=prepare_data_for_get_attack(data)
    print("Data has been prepared")
    for i in range(len(shadow_models)):
        print(f"*****training attack model {i+1}*****")
        attack_model = get_attack(attack_train_dataloader,attack_val_dataloader,args, victim_model, shadow_models[i])
        attack_models.append(attack_model) # trained attack models
    print("""""testing the mechanism""")
    for model in attack_models:
        _,_, pred_labels,total_true_targets=test_attack_model(args,model, attack_val_dataloader,num_classes=args.n_classes,victim_model = victim_model, accuracy_only = False,input_channels = 1, multiple_shadow = False, layer = -1, number = 1, roc = False, plot_test = False,different_shadow=True)
        aggregated_prediction_validation.append(pred_labels)

    # for total_true_targets, the last one is enough
    print("***total evaluation for validation data when we have more than one shadow models***")
    stacked_batch_pred= np.stack(aggregated_prediction_validation, axis=1) # 2D array : rows: samples , columns: attack models
    #print(f"stacked_batch_pred:{stacked_batch_pred}")
    majority_vote_preds= np.apply_along_axis(lambda x:np.argmax(np.bincount(x.astype(int))), axis=1, arr=stacked_batch_pred)
    a_score = round(accuracy_score(total_true_targets, majority_vote_preds), 4)
    f_score = round(f1_score(total_true_targets, majority_vote_preds), 4)
    print(
        'Validation accuracy:', a_score,
        ', F1-score:', f_score,
    )


def get_attack_patchify(data,args,victim_model,shadow_model):
    attack_train = DatasetOct(data.shadow_attack_paths, attack=True)
    check_balance_in_dataset(attack_train)
    attack_train_dataloader = DataLoader(attack_train, batch_size=ATTACK_BATCH_SIZE, shuffle=True)
    attack_val = DatasetOct(data.victim_attack_paths, attack=True)
    attack_val_dataloader = DataLoader(attack_val, batch_size=ATTACK_BATCH_SIZE,shuffle=True)
    attack_model = train_attack_model_patchify(
        args, shadow_model, victim_model, attack_train_dataloader, attack_val_dataloader, ATTACK_EPOCHS, ATTACK_LR)
    print("train attack model with patchify has finished")





def get_attack_multiple(data, args, victim_model, shadow_models):
    attack_models=[]
    dataloaders=[]
    for i in range (len(shadow_models)):
        print(f"Training {i}the attack model")
        attack_train = DatasetOct(data.shadow_attack_paths,label_index=i, attack=True)
        attack_train_dataloader = DataLoader(attack_train, batch_size=ATTACK_BATCH_SIZE, shuffle=False)
        attack_val = DatasetOct(data.victim_attack_paths,label_index=i, attack=True)
        attack_val_dataloader = DataLoader(attack_val, batch_size=ATTACK_BATCH_SIZE,shuffle=False)
        attack_model = train_attack_model(
            args, shadow_models[i], victim_model, attack_train_dataloader, attack_val_dataloader, ATTACK_EPOCHS, ATTACK_LR,multiple_shadow=True,layer=i)
        attack_models.append(attack_model)
        dataloaders.append(attack_val_dataloader)
    # test entire attack models
    return attack_models,dataloaders




def get_shadow_data_generation(data,args, victim_model,file_shadow,dataset_synthetic): # shadow and victim model are the same
    shadow_train = DatasetOct(data.external_data,external=True)
    shadow_train_dataloader = DataLoader(shadow_train, batch_size=int(args.batch_size), shuffle=False)
    print("DPSGD_for_Saving_and_synthetic: ",args.DPSGD_for_Saving_and_synthetic)
    """if args.DPSGD_for_Saving_and_synthetic:
        if args.main_dir=="DukeData":
            save_path = "shadow_dataset_DPSGD"
        else:
            save_path = "shadow_dataset_DPSGD_UMN"
    else:
        if args.main_dir=="DukeData":
            save_path= "shadow_dataset"
        else:
            save_path = "shadow_dataset_UMN"""
    train_synthetic(victim_model, shadow_train_dataloader, args.device,args.model_name,dataset_synthetic,file_shadow,args.DPSGD_for_Saving_and_synthetic)

def get_shadow_multiple(data, args):
    shadow_models=[]
    losses=[]
    for i in range(args.n_classes):
        print(f"training {i}th shadow model")
        shadow_train = DatasetOct(data.shadow_train_paths,label_index=i)
        shadow_train_dataloader = DataLoader(shadow_train, batch_size=int(args.batch_size), shuffle=True)
        shadow_val = DatasetOct(data.shadow_val_paths,label_index=i)
        shadow_val_dataloader = DataLoader(shadow_val, batch_size=int(args.batch_size))
        #plot_images_with_masks(shadow_val_dataloader,i)
        shadow_model, loss = seg_train(args, shadow_train_dataloader, shadow_val_dataloader,counter=int(i),victim=False,shadow=False,plotting=True,multiple_shadow=True)
        print(f"saving shadow model {i} in shadow models list")
        shadow_models.append(shadow_model)
        losses.append(loss)

    return shadow_models, losses



def having_different_shadow(args,data):
    "here we train different shadow model but all with same training data"
    shadow_models=[]
    shadow_losses=[]
    k=args.number_shadow # how many shadow models we want to have
    # training and validation datasets are shared among all shadow models
    shadow_train = DatasetOct(data.shadow_train_paths)
    shadow_train_dataloader = DataLoader(shadow_train, batch_size=int(args.batch_size), shuffle=True)
    shadow_val = DatasetOct(data.shadow_val_paths)
    shadow_val_dataloader = DataLoader(shadow_val, batch_size=int(args.batch_size))

    for i in range(k):
        print(f"training {i}th shadow model")
        print("shadow model training")
        shadow_model, average_loss = seg_train(args, shadow_train_dataloader, shadow_val_dataloader, counter=None,
                                               victim=False, shadow=True, plotting=True) # train each shadow model

        print(f"saving shadow model {i} in shadow models list")
        shadow_models.append(shadow_model) # save shadow models
        shadow_losses.append(average_loss) # save loss of shadow models

    return shadow_models, shadow_losses