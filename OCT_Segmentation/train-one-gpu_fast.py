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

import argparse
import torch
import tqdm
import csv
from data_one_gpu import get_data
from losses import CombinedLoss, FocalFrequencyLoss
from networks import get_model
from utils import per_class_dice
from utils import MAE_New
from utils import compute_dice
from utils import compute_pa
from opacus import PrivacyEngine
import kornia
from kornia.morphology import opening, closing,dilation,erosion
import os
import numpy as np
#from fastDP import PrivacyEngine
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from fast_differential_privacy.fastDP.privacy_engine  import PrivacyEngine


torch.cuda.empty_cache()
print("Current Directory:", os.getcwd())
PYTORCH_NO_CUDA_MEMORY_CACHING=1



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def argument_parser():
    parser = argparse.ArgumentParser()

    # Add all arguments upfront
    parser.add_argument('--dataset', default='UMN', choices=["Duke", "UMN"])

    parser.add_argument('--batch_size', default=16 , type=int)
    parser.add_argument('--num_iterations', default=100,type=int)
    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--n_classes', default=2, type=int)
    parser.add_argument('--ffc_lambda', default=0, type=float)
    parser.add_argument('--weight_decay', default=1e-8, type=float)
    parser.add_argument('--image_size', default='224', type=int)
    parser.add_argument('--model_name', default="NestedUNet",
                        choices=["unet", "y_net_gen", "y_net_gen_ffc", 'ReLayNet', 'UNetOrg', 'LFUNet', 'FCN8s',
                                 'NestedUNet','SimplifiedFCN8s','ConvNet'])
    parser.add_argument('--g_ratio', default=0.5, type=float)
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])
    parser.add_argument('--seed', default=7, type=int)

    parser.add_argument('--in_channels', default=1, type=int)
    # Initially, do not set a default for image_dir
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--DPSGD', type=str2bool, default=True)
    parser.add_argument('--test', type=bool, default=False)
    # if save
    parser.add_argument('--model_should_be_saved', type=bool, default=False)
    parser.add_argument('--model_should_be_load', type=bool, default=False)
    parser.add_argument('--save_dir', default='./saved_models/', type=str, help="Directory to save the models")
    parser.add_argument('--epsilon', default=1.1, type=float, help="Privacy epsilon value for DPSGD")
    parser.add_argument('--morphology', default=False, type=str2bool, help="morphology")
    parser.add_argument('--operation',default='open', type=str, help="close, open or both")
    parser.add_argument('--kernel_size', default=3, type=int, help="kernel size")

    # Parse the arguments
    args = parser.parse_args()

    # Dynamically set the image_dir based on the dataset argument
    if not args.image_dir:  # Only set image_dir if it wasn't explicitly provided
        if args.dataset == "UMN":
            args.image_dir = "./UMNData/"
        else:
            args.image_dir = "../DukeData/"

    return args



def apply_kornia_morphology_multiclass(pred_mask:torch.Tensor, operation:str='both',kernel_size:int=3) ->torch.Tensor:
    """
        Apply morphological operations using Kornia to refine multi-class predicted masks.

        Args:
            pred_mask (torch.Tensor): Multi-class mask of shape [B, C, H, W], where C = num_classes.
            operation (str): 'open', 'close', 'both', or 'none'.
            kernel_size (int): Size of the structuring element.

        Returns:
            torch.Tensor: Refined multi-class mask of shape [B, C, H, W].
        """
    if operation not in ['open', 'close', 'both', 'none','dilation','erosion']:
        raise ValueError("Operation must be one of 'open', 'close', 'both', or 'none'.")
    kernel= torch.ones(kernel_size, kernel_size).to(pred_mask.device)
    if operation == 'dilation':
        refined_mask = dilation(pred_mask, kernel)
    elif operation == 'open':
        refined_mask = opening(pred_mask, kernel)
    elif operation == 'close':
        refined_mask = closing(pred_mask, kernel)
    elif operation == 'erosion':
        refined_mask = erosion(pred_mask, kernel)
    elif operation == 'both':
        refined_mask = opening(pred_mask, kernel)
        refined_mask = closing(refined_mask, kernel)
    else:
        raise ValueError('open', 'close', 'both', 'none','dilation','erosion')
    return refined_mask


def colored_text(st):
    return '\033[91m' + st + '\033[0m'


def plot_examples(data_loader, model, device, num_examples=3):
    model.eval()  # Set the model to evaluation mode
    fig, axs = plt.subplots(num_examples, 3, figsize=(15, 5 * num_examples))

    batch_processed = 0
    for imgs, masks in data_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():  # We do not need to compute gradients here
            preds = model(imgs)

        # Assuming the output is a softmax layer
        _, predicted_masks = torch.max(preds, dim=1)
        # Determine the maximum label for setting up the colormap
        max_label= max (np.max(masks.cpu().numpy()), np.max(predicted_masks.cpu().numpy()))
        cmap = plt.cm.get_cmap('viridis', max_label+1)

        for idx in range(imgs.size(0)):
            if batch_processed>= num_examples:
                break

            img=imgs[idx].cpu().numpy().squeeze()
            mask=masks[idx].cpu().numpy().squeeze()
            predicted_mask=predicted_masks[idx].cpu().numpy()

            axs[batch_processed, 0].imshow(img, cmap='gray')  # Assuming image is in the first channel
            axs[batch_processed, 0].set_title('Input Image')
            axs[batch_processed, 0].axis('off')

            axs[batch_processed, 1].imshow(mask, cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=max_label))
            axs[batch_processed, 1].set_title('Ground Truth Mask')
            axs[batch_processed, 1].axis('off')

            axs[batch_processed, 2].imshow(predicted_mask, cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=max_label))
            axs[batch_processed, 2].set_title('Predicted Mask')
            axs[batch_processed, 2].axis('off')

            batch_processed+=1
        if batch_processed >= num_examples:
            break

    plt.tight_layout()
    plt.show()
import random
def segmentation_plots_test_morphology(val_loader, model, device,model_name,DPSGD, dataset, num_examples=5): # in this code, we already did not apply morphology

    folder_name = f"{'' if not DPSGD else 'DPSGD'}_{dataset}_images"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    batch_processed=0
    model.eval()
    for imgs, masks in val_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():  # We do not need to compute gradients here
            preds = model(imgs)

            if args.morphology:
                refined_mask = apply_kornia_morphology_multiclass(
                    preds,
                    operation='erosion',
                    kernel_size=3,
                )
                #print("Loss value in original case", MAE(masks, preds, args.n_classes))
                #print("Loss value for refined case", MAE(masks, refined_mask, args.n_classes))



                _, refined_predicted_masks = torch.max(refined_mask, dim=1)

            _, predicted_masks = torch.max(preds, dim=1)

        indices = list(range(imgs.size(0)))
        random.shuffle(indices)


        for idx in indices:  # Use shuffled indices to select random images
            if batch_processed>= num_examples:
                break

            img=imgs[idx].cpu().numpy().squeeze()
            msk=masks[idx].cpu().numpy().squeeze()
            predicted_mask=predicted_masks[idx].cpu().numpy()
            if args.morphology:
             refined_predicted_mask =refined_predicted_masks[idx].cpu().numpy()

            # Define colors for each layer (RGB tuples)
            layer_colors = {
                0: (0, 0, 0),  # Black for the modified 0
                1: (1, 0, 0),  # Red
                2: (0, 1, 0),  # Green
                3: (0, 0, 1),  # Blue
                4: (1, 1, 0),  # Yellow
                5: (1, 0, 1),  # Magenta
                6: (0, 1, 1),  # Cyan
                7: (1, 0.5, 0),  # Orange
                8: (0.8, 0.7, 0.6)  # Light Brown
            }

            def create_overlay(img, mask, layer_colors):
                overlay = np.zeros((img.shape[0], mask.shape[1], 3))
                for value, color in layer_colors.items():
                    overlay[mask == value] = color
                return overlay


            true_overlay = create_overlay(img, msk, layer_colors)
            predicted_overlay = create_overlay(img, predicted_mask, layer_colors)
            if args.morphology:
                refined_predicted_overlay = create_overlay(img, refined_predicted_mask, layer_colors)

            original_rgb = np.stack([img] * 3, axis=-1)
            min_val = original_rgb.min()
            max_val = original_rgb.max()
            # Normalize based on observed min and max
            original_rgb = (original_rgb - min_val) / (max_val - min_val)

            true_combined = np.clip(0.7 * original_rgb + 0.3 * true_overlay, 0, 1)
            predicted_combined = np.clip(0.7 * original_rgb + 0.3 * predicted_overlay, 0, 1)
            if args.morphology:
                refined_predicted_combined=np.clip(0.7 * original_rgb + 0.3 * refined_predicted_overlay, 0, 1)
            if DPSGD==True:
                state='DPSGD'
            else:
                state=''


            # Plotting
            if args.morphology:
             fig, axes = plt.subplots(1, 4, figsize=(30, 10))
            else:
                fig, axes = plt.subplots(1, 3, figsize=(30, 10))
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(true_combined)
            axes[1].set_title('True Mask')
            axes[1].axis('off')

            axes[2].imshow(predicted_combined)
            axes[2].set_title('Predicted Mask')
            axes[2].axis('off')
            if args.morphology:
                axes[3].imshow(refined_predicted_combined)
                axes[3].set_title('Refined Predicted Mask')
                axes[3].axis('off')


            plt.axis('off')

            #file_path = os.path.join(folder_name, f'{model_name}_{state}_{dataset}_{idx}.pdf')
            #plt.savefig(file_path, format='pdf', bbox_inches='tight')
            plt.show()
            batch_processed += 1


def segmentation_plots(val_loader, model, device, model_name, DPSGD, dataset, num_examples=5):  # this version can work after applying morphological approach or normal cases without applying any morphological technique  (because after applying morphology then prediction of the model is the result of morphology)
    folder_name = f"{'' if not DPSGD else 'DPSGD'}_{dataset}_images"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    batch_processed = 0
    model.eval()
    for imgs, masks in val_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():  # We do not need to compute gradients here
            preds = model(imgs)
            _, predicted_masks = torch.max(preds, dim=1)

        indices = list(range(imgs.size(0)))
        random.shuffle(indices)

        for idx in indices:  # Use shuffled indices to select random images
            if batch_processed >= num_examples:
                break

            img = imgs[idx].cpu().numpy().squeeze()
            msk = masks[idx].cpu().numpy().squeeze()
            predicted_mask = predicted_masks[idx].cpu().numpy()

            # Define colors for each layer (RGB tuples)
            layer_colors = {
                0: (0, 0, 0),  # Black for the modified 0
                1: (1, 0, 0),  # Red
                2: (0, 1, 0),  # Green
                3: (0, 0, 1),  # Blue
                4: (1, 1, 0),  # Yellow
                5: (1, 0, 1),  # Magenta
                6: (0, 1, 1),  # Cyan
                7: (1, 0.5, 0),  # Orange
                8: (0.8, 0.7, 0.6)  # Light Brown
            }

            def create_overlay(img, mask, layer_colors):
                overlay = np.zeros((img.shape[0], mask.shape[1], 3))
                for value, color in layer_colors.items():
                    overlay[mask == value] = color
                return overlay

            true_overlay = create_overlay(img, msk, layer_colors)
            predicted_overlay = create_overlay(img, predicted_mask, layer_colors)


            original_rgb = np.stack([img] * 3, axis=-1)
            min_val = original_rgb.min()
            max_val = original_rgb.max()
            # Normalize based on observed min and max
            original_rgb = (original_rgb - min_val) / (max_val - min_val)

            true_combined = np.clip(0.7 * original_rgb + 0.3 * true_overlay, 0, 1)
            predicted_combined = np.clip(0.7 * original_rgb + 0.3 * predicted_overlay, 0, 1)

            if DPSGD == True:
                state = 'DPSGD'
            else:
                state = ''

            # Plotting

            fig, axes = plt.subplots(1, 3, figsize=(30, 10))
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(true_combined)
            axes[1].set_title('True Mask')
            axes[1].axis('off')

            axes[2].imshow(predicted_combined)
            axes[2].set_title('Predicted Mask')
            axes[2].axis('off')


            plt.axis('off')

            # file_path = os.path.join(folder_name, f'{model_name}_{state}_{dataset}_{idx}.pdf')
            # plt.savefig(file_path, format='pdf', bbox_inches='tight')
            plt.show()
            batch_processed += 1


def visualize_batch(images, masks):
    batch_size = len(images)
    fig, ax = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))

    for i in range(batch_size):
        img = images[i].permute(1, 2, 0)  # Convert from CxHxW to HxWxC
        msk = masks[i].squeeze()  # Remove channel dim if it's there

        ax[i, 0].imshow(img)
        ax[i, 0].set_title('Input Image')
        ax[i, 0].axis('off')

        ax[i, 1].imshow(img)
        ax[i, 1].imshow(msk, alpha=0.3, cmap='jet')  # Overlay mask with transparency
        ax[i, 1].set_title('Overlay Mask')
        ax[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_files(path, ext):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(ext)]


def save_results_to_csv(args,file_name,max_grad_norm,noise_multiplier, model_name, learning_rate, batch_size, training_losses, validation_losses,dice_all, validation_dice_scores, privacy_epsilons,iterations,dataset,mae,per_layer_all_list):
    # Prepare data to save in CSV
    if args.morphology:
        row_data = [
            model_name,
            learning_rate,
            batch_size,
            training_losses[-1] if training_losses else None,
            validation_losses[-1] if validation_losses else None,
            dice_all[-1] if dice_all else None,
            validation_dice_scores[-1] if validation_dice_scores else None,
            privacy_epsilons if privacy_epsilons else None,
            max_grad_norm,
            noise_multiplier,
            iterations,
            dataset,
            mae[-1],
            per_layer_all_list[-1],
            args.operation,
            args.kernel_size

        ]
    else:
        row_data = [
            model_name,
            learning_rate,
            batch_size,
            training_losses[-1] if training_losses else None,
            validation_losses[-1] if validation_losses else None,
            dice_all[-1] if dice_all else None,
            validation_dice_scores[-1] if validation_dice_scores else None,
            privacy_epsilons if privacy_epsilons else None,
            max_grad_norm,
            noise_multiplier,
            iterations,
            dataset,
            mae[-1],
            per_layer_all_list[-1]

        ]

    # Check if the CSV file exists
    file_exists = os.path.isfile(file_name)

    try:
        # Open the file for appending (or create it if it doesn't exist)
        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                # Write the header only if the file doesn't exist
                if args.morphology:
                    writer.writerow(
                        ["Model_Name", "Learning_Rate", "Batch_Size", "Training_Loss", "Validation_Loss", "dice_all",
                         "Validation_Dice", "Privacy_Epsilons", 'max_grad_norm', 'noise_multiplier', 'iterations',
                         'dataset', 'mae', 'per_layer_all_list','Operation','Kernel'])


                else:
                 writer.writerow(["Model_Name", "Learning_Rate", "Batch_Size", "Training_Loss", "Validation_Loss", "dice_all","Validation_Dice","Privacy_Epsilons", 'max_grad_norm', 'noise_multiplier', 'iterations', 'dataset','mae', 'per_layer_all_list'])
            # Write the row of data
            writer.writerow(row_data)

        # Get the absolute path of the
        file_path = os.path.abspath(file_name)
        print(f"Results saved to {file_path}")
    except Exception as e:
        print(f"Failed to save results to CSV: {e}")

def eval(val_loader, criterion, model, n_classes, dice_s=True, device="cuda", im_save=False):

    model.eval()
    loss = 0
    counter = 0
    dice = 0
    mae=0

    dice_all = np.zeros(n_classes)
    per_layer_all=np.zeros(n_classes)
    with torch.no_grad():
        for img, label in tqdm.tqdm(val_loader):
            img = img.to(device)
            label = label.to(device)
            label_oh = torch.nn.functional.one_hot(label, num_classes=n_classes)


            pred = model(img)
            max_val, idx = torch.max(pred, 1)
            pred_seg = idx.cpu().data.numpy()
            label_seg = label.cpu().data.numpy()
            ret= compute_dice(label_seg, pred_seg)
            print(f"dice score: {ret}")
            pa =compute_pa(label_seg, pred_seg)
            print(f"pa: {pa}")



            pred_oh = torch.nn.functional.one_hot(idx, num_classes=n_classes)
            #print("shapes before : ", label_oh.shape, pred_oh.shape)

            if dice_s:
                d1, d2 = per_class_dice(pred_oh, label_oh, n_classes)
                dice += d1
                dice_all += d2

            loss += criterion(pred, label.squeeze(1), device=device).item()
            #print(f"y_true shape: {label.shape}")
            #print(f"y_pred shape: {pred.shape}")

            label_mae = torch.nn.functional.one_hot(label, num_classes=n_classes).squeeze()
            label_mae=label_mae.permute(0,3,1,2)
            #init_mae,per_layer = MAE(label,pred, n_classes=args.n_classes)
            init_mae, per_layer = MAE_New(label_mae, pred, n_classes=args.n_classes)
            #init_mae_new = mae_new(label_mae, pred)

            mae+= init_mae.item()
            print(f"MAE in this step: {init_mae}")
            #print(f"MAE in new step: {init_mae_new}")
            print(f"MAE per layer: {per_layer}")
            #print(f"MAE per layer new: {per_layer_new}")
            per_layer_all+= per_layer # average of mean absolute error over each layer over all dataset

            counter += 1

        loss = loss / counter
        dice = dice / counter
        dice_all = dice_all / counter
        per_layer_all = per_layer_all / counter
        mae = mae / counter

        print("Validation loss: ", loss, " Mean Dice: ", dice.item(), "Dice All:", dice_all, "MAE: ", mae, "per layer: ", per_layer_all)
        return dice, loss, dice_all ,mae, per_layer_all


def train(args):
    import os

    device = args.device
    n_classes = args.n_classes
    model_name = args.model_name
    learning_rate = args.learning_rate
    ratio = args.g_ratio
    data_path = args.image_dir
    iterations = args.num_iterations
    img_size = args.image_size
    batch_size = args.batch_size
    test=args.test

    training_losses = []
    validation_losses = []
    dice_all_list=[]
    per_layer_all_list=[]
    mae_all=[]
    validation_dice_scores = []
    max_consecutive_epochs_without_improvement = 10
    consecutive_epochs_without_improvement = 0
    best_test_loss = float("inf")

    # calculating len of data for delta
    train_data_path = os.path.join(data_path, 'train')
    train_data_path_images = os.path.join(train_data_path, 'images')
    images_files = os.listdir(train_data_path_images)
    image_files = [file for file in images_files if file.endswith(('.npy'))]
    number_of_images = len(image_files)

    criterion_seg = CombinedLoss()
    criterion_ffc = FocalFrequencyLoss()
    if args.morphology:
        final_save_dir=os.path.join(args.save_dir, 'morphology_models')
    else:
        final_save_dir=os.path.join(args.save_dir, 'models')

    if not os.path.exists(final_save_dir):
        os.makedirs(final_save_dir)

    if args.DPSGD:
        if args.morphology:
            save_name = os.path.join(final_save_dir, f"{args.model_name}_{args.dataset}_DPSGD_{args.num_iterations}_{args.batch_size}_{args.epsilon}_{args.operation}_{args.kernel_size}.pt")
        else:
            save_name = os.path.join(final_save_dir, f"{args.model_name}_{args.dataset}_DPSGD_{args.num_iterations}_{args.batch_size}_{args.epsilon}.pt")
    else:
        if args.morphology:
            save_name = os.path.join(final_save_dir, f"{args.model_name}_{args.dataset}_{args.num_iterations}_{args.batch_size}_{args.operation}_{args.kernel_size}.pt")
        else:
            save_name = os.path.join(final_save_dir, f"{args.model_name}_{args.dataset}_{args.num_iterations}_{args.batch_size}.pt")

    # save_name = model_name  + ".pt"

    max_dice = 0
    best_test_dice = 0
    best_iter = 0

    model = get_model(model_name, ratio=ratio, num_classes=n_classes).to(device)
    print(model_name)


    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate,
                                 weight_decay=args.weight_decay)

    train_loader, val_loader, test_loader, _, _, _ = get_data(data_path, img_size, batch_size)

    #for images, labels in train_loader:
        #visualize_batch(images, labels)
        #break

    if args.model_should_be_load:
        if os.path.exists(save_name):
            checkpoint = torch.load(save_name)
            state_dict = checkpoint['model_state_dict']

            # Strip the `_module.` prefix if it exists
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace('_module.', '')  # Remove the _module prefix
                new_state_dict[new_key] = state_dict[key]

            model.load_state_dict(new_state_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iteration = checkpoint['iteration'] + 1
            training_losses.append(checkpoint['training_losses'])
            validation_losses.append(checkpoint['validation_losses'])
            validation_dice_scores.append(checkpoint['validation_dice_scores'])
            best_test_loss = checkpoint['best_test_loss']
            dice_all_list = checkpoint['per_layer_all_list']
            mae_all = checkpoint['mae_all']
            dice_all = checkpoint['dice_all_list']
            consecutive_epochs_without_improvement = checkpoint['consecutive_epochs_without_improvement']
            print(f"Resuming training from iteration {start_iteration}")

    else:
        start_iteration = 1

    # delta = 1 / (number_of_images ** 1.1)
    if args.DPSGD==True:
        delta = 1e-5
        #privacy_engine = PrivacyEngine()
        #noise_multiplier = 1
        #max_grad_norm = 1
        model_name=f"{args.model_name}_DPSGD"

        """model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm, )"""
        """model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            # noise_multiplier=noise_multiplier,
            target_epsilon=args.epsilon,
            target_delta=delta,
            epochs=iterations,
            max_grad_norm=max_grad_norm, )"""

        privacy_engine = PrivacyEngine(
            model,
            batch_size=args.batch_size,
            sample_size=50000,
            epochs=200,
            target_epsilon=args.epsilon,
            clipping_fn='automatic',
            clipping_mode='MixOpt',
            origin_params=None,
            clipping_style='all-layer',
        )
        # attaching to optimizers is not needed for multi-GPU distributed learning
        privacy_engine.attach(optimizer)
        privacy_epsilons = []
    else:
        noise_multiplier = 'None'
        max_grad_norm = 'None'
        delta='None'
        model_name=args.model_name


    for t in range(start_iteration, iterations):
        model.train()
        total_loss = 0
        total_samples = 0

        for img, label in (train_loader):
            img = img.to(device)

            label = label.to(device)

            label_oh = torch.nn.functional.one_hot(label, num_classes=n_classes).squeeze()

            pred = model(img)
            #print(pred.shape)
            #print("morphological operation")
            #refined_image= apply_kornia_morphology_multiclass(pred, operation='dilation', kernel_size=3)
            #print(refined_image.shape)

            max_val, idx = torch.max(pred, 1)
            pred_oh = torch.nn.functional.one_hot(idx, num_classes=n_classes)
            pred_oh = pred_oh.permute(0, 3, 1, 2)
            label_oh = label_oh.permute(0, 3, 1, 2)
            """loss = criterion_seg(pred, label.squeeze(1), device=device) + args.ffc_lambda * criterion_ffc(pred_oh,
                                                                                                       label_oh)"""
            #print(pred.shape)
            #print(label.squeeze(1).shape)
            if args.morphology:
                pred= apply_kornia_morphology_multiclass(pred,operation=args.operation, kernel_size=args.kernel_size)

            loss = criterion_seg(pred, label.squeeze(1), device=device)
            optimizer.zero_grad() #zero_grad clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * img.size(0)
            total_samples += img.size(0)

        average_loss = total_loss / total_samples
        training_losses.append(average_loss)

        if args.DPSGD==True:
            #epsilon = privacy_engine.get_epsilon(delta)
            #privacy_epsilons.append(epsilon)

            print(
                f"\tTrain Epoch: [{t + 1}/{iterations}] \t"
                f"Train Loss: {np.mean(average_loss):.6f} "
                #f"(ε = {epsilon:.2f}, δ = {delta})"
            )
        else:
            print(
                f"\tTrain Epoch: [{t + 1}/{iterations}] \t"
                f"Train Loss: {np.mean(average_loss):.6f} ")

        if t % 20 == 0:  # Every 20 epochs
            #plot_examples(val_loader, model, device)
            segmentation_plots(val_loader, model, device, model_name, DPSGD=args.DPSGD, dataset=args.dataset, num_examples=1)

        if t % 10== 0 or t > 45:
            print("Validation")
            dice, validation_loss, dice_all,mae, per_layer_all = eval(val_loader, criterion_seg, model, dice_s=True, n_classes=n_classes)
            #segmentation_plots(val_loader, model, device, model_name, args.DPSGD, args.dataset)
            validation_losses.append(validation_loss)
            validation_dice_scores.append(dice.item())
            mae_all.append(mae)
            dice_all_list.append(dice_all)
            dice_all_str = str(per_layer_all.tolist()) # to make it possible to work with ast
            per_layer_all_list.append(dice_all_str)


            # print("Expert 1 - Test")
            # dice_test = eval(test_loader, criterion_seg, model, n_classes=n_classes)

            if dice > max_dice:
                max_dice = dice
                best_iter = t
                # best_test_dice = dice_test
                # torch.save(model, save_name)
            if args.model_should_be_saved:
                if args.DPSGD == True:
                    print(colored_text("Updating model, epoch: "), t)
                    torch.save({
                        'iteration': t,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'training_losses': training_losses,
                        'validation_losses': validation_losses,
                        'validation_dice_scores': validation_dice_scores,
                        'best_test_loss': best_test_loss,
                        'dice_all_list': dice_all_list,
                        'per_layer_all_list':per_layer_all_list,
                        'mae_all':mae_all,
                        'consecutive_epochs_without_improvement': consecutive_epochs_without_improvement,
                        'privacy_epsilons': privacy_epsilons,

                    }, save_name)
                else:
                    print(colored_text("Updating model, epoch: "), t)
                    torch.save({
                        'iteration': t,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'training_losses': training_losses,
                        'validation_losses': validation_losses,
                        'validation_dice_scores': validation_dice_scores,
                        'best_test_loss': best_test_loss,
                        'dice_all_list': dice_all_list,
                        'per_layer_all_list':per_layer_all_list,
                        'mae_all':mae_all,
                        'consecutive_epochs_without_improvement': consecutive_epochs_without_improvement,
                    }, save_name)

            if validation_loss < best_test_loss : # if there is any improvement
                best_test_loss= validation_loss
                consecutive_epochs_without_improvement = 0  # reset
            else:
                consecutive_epochs_without_improvement = +1
            if consecutive_epochs_without_improvement >= max_consecutive_epochs_without_improvement:
                print(
                    f"Stopping training due to lack of improvement for {max_consecutive_epochs_without_improvement} epochs.")
                break  # Exit the training loop
        """import matplotlib.pyplot as plt
        plt.plot(range(len(privacy_epsilons)), privacy_epsilons)
        plt.xlabel("Epochs")
        plt.ylabel("Privacy Budget (ε)")
        plt.title("Privacy Budget over Epochs")
        plt.show()"""
        model.train()

    # print("Best iteration: ", best_iter, "Best val dice: ", max_dice, "Best test dice: ", best_test_dice)
    print(f" training loss:{training_losses}")
    print(f" validation loss:{validation_losses}")
    if args.DPSGD==True:
        print(f" privacy_epsilons:{privacy_epsilons}")
        privacy_epsilons_str = str(privacy_epsilons)
    else:
        privacy_epsilons_str=0
    print(f" validation dice score:{validation_dice_scores}")


    # privacy_epsilons_str = ""
    """if not test:

        print("Best iteration: ", best_iter, "Best val dice: ", max_dice)

        if args.DPSGD== True:
            privacy_epsilon= privacy_epsilons[-1]
        else:
            privacy_epsilon=0


        if args.morphology:
            file_name='model_results_fixed_epsilon_mae_morphology.csv'
        else:file_name= 'model_results_fixed_epsilon_mae.csv'
        save_results_to_csv(args,
            file_name,
            max_grad_norm,
            noise_multiplier,
            model_name,
            learning_rate,
            batch_size,
            training_losses,
            validation_losses,
            dice_all_list,
            validation_dice_scores,
            privacy_epsilon,
            iterations,
            args.dataset,
            mae_all,
            per_layer_all_list



        )
        print("saving has finished")
        #segmentation_plots(val_loader, model, device, model_name, args.DPSGD, args.dataset,num_examples=20 )


    if test:

        dice_test, test_loss, dice_all_test,mae_test,per_layer_all_test =eval(test_loader, criterion_seg, model, dice_s=True, n_classes=n_classes)

        if args.DPSGD == True:
            privacy_epsilon = privacy_epsilons[-1]
        else:
            privacy_epsilon = 0 # None

        if args.morphology:
            file_name='model_results_fixed_epsilon_mae_morphology.csv'
        else:
            file_name= 'model_results_fixed_epsilon_mae.csv'
        save_results_to_csv(args,
            file_name,
            max_grad_norm,
            noise_multiplier,
            model_name,
            learning_rate,
            batch_size,
            training_losses,
            [test_loss],
            dice_all,
            [dice_test],
            privacy_epsilon,
            iterations,
            args.dataset,
            mae_test,
            per_layer_all_test
        )


        # plotting
        #segmentation_plots(test_loader, model, device, model_name,args.DPSGD,args.dataset,num_examples=20)
    print(f" training loss:{training_losses}")
    print(f" validation loss:{validation_losses}")"""


    return model




if __name__ == "__main__":
    args = argument_parser()
    print(args)
    set_seed(args.seed)
    train(args)


# scp /home/parsar0000/pythonProject4/training.py shiva.parsarad@unibas.ch@chinchilla.dmi.unibas.ch:/home/parsar0000/BPR/pythonProject4


