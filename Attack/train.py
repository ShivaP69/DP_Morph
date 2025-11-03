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
import torch.nn as nn
import argparse
import csv
import sys
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import random
import csv
#from PIL import Image
from losses import CombinedLoss, FocalFrequencyLoss
from networks import get_model
from utils import per_class_dice
from utils import compute_dice
from utils import compute_pa
from opacus import PrivacyEngine
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
#from fastDP import PrivacyEngine
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from kornia.morphology import opening, closing,dilation,erosion
print("Current Directory:", os.getcwd())

model_path_template_shadow = "{model_name}_{type_model}_{state}_{counter}_{data}.pth"
model_path_template="{model_name}_{type_model}_{state}_{data}.pth"
def save_model(args,model, model_name, optimizer, epoch, validation_loss,average_loss,data,counter=None,victim=None, shadow=None, state='normal'):
    if victim:
        type_model = 'victim'
        if args.morphology:
            model_path = model_path_template.format(model_name=f'{model_name}_{args.batch_size}_{args.num_iterations}_{args.operation}_{args.kernel_size}',
                                                    type_model=type_model, state=state, data=data)
        else:
            model_path=model_path_template.format(model_name=f'{model_name}_{args.batch_size}_{args.num_iterations}', type_model=type_model, state=state, data=data)
    elif shadow:
        type_model = 'shadow'
        if args.morphology:
            model_path = model_path_template_shadow.format(
                model_name=f'{model_name}_{args.batch_size}_{args.num_iterations}_{args.operation}_{args.kernel_size}', type_model=type_model, state=state,
                counter=counter, data=data)
        else:
            model_path = model_path_template_shadow.format(model_name=f'{model_name}_{args.batch_size}_{args.num_iterations}', type_model=type_model, state=state,
                                                counter=counter,data=data)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_loss': validation_loss,
        'average_loss': average_loss,

    }, model_path)
    print('Model saved in path: {}'.format(model_path))


def load_model_if_exists(args,model, model_name, optimizer, data, counter=None, victim=None, shadow=None, state='normal'):
    if victim:
        type_model = 'victim'
        if args.morphology:
            model_name=f'{model_name}_{args.batch_size}_{args.num_iterations}_{args.operation}_{args.kernel_size}'
        else:
            model_name= f'{model_name}_{args.batch_size}_{args.num_iterations}'
        model_files = [f for f in os.listdir() if f.startswith(f"{model_name}_{type_model}_{state}_{data}")]

    elif shadow:
        type_model = 'shadow'
        if args.morphology:
            model_name= f'{model_name}_{args.batch_size}_{args.num_iterations}_{args.operation}_{args.kernel_size}'
        else:
            model_name=f'{model_name}_{args.batch_size}_{args.num_iterations}'
        model_files = [f for f in os.listdir() if f.startswith(f"{model_name}_{type_model}_{state}_{counter}_{data}")]


    latest_model = max(model_files, key=os.path.getctime, default=None)
    if latest_model:
        checkpoint = torch.load(latest_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        validation_loss=checkpoint['validation_loss']
        average_loss=checkpoint['average_loss']

        print(f"Loaded model from {latest_model} starting at epoch {start_epoch + 1}")
        return model, optimizer, start_epoch + 1, validation_loss,average_loss

    else:
        print("No model found")
        print("Victim model training will be start")
        return model, optimizer, 0, None,None


def colored_text(st):
    return '\033[91m' + st + '\033[0m'


def plot_examples(data_loader, model, device, num_examples=2):
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

def segmentation_plots(args,val_loader, model, device, model_name,multiple_shadow, DPSGD, dataset,morphology=False,number=0,num_examples=1):
    print("segmentation plots start")
    print(f"dataset:{dataset}")
    if morphology:
        folder_name = f"{'' if not DPSGD else 'DPSGD'}_{dataset}_images_morphology"
    else:
        folder_name = f"{'' if not DPSGD else 'DPSGD'}_{dataset}_images"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    batch_processed=0
    model.eval()
    for imgs, masks in val_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():  # We do not need to compute gradients here
            preds = model(imgs)
            if morphology:
                if args.morphology:
                    preds = apply_kornia_morphology_multiclass(preds, operation=args.operation,
                                                              kernel_size=args.kernel_size)
            if multiple_shadow:
                pred = torch.sigmoid(preds)
                predicted_masks = (pred > 0.5).float()
            else:
                _, predicted_masks = torch.max(preds, dim=1)
        for idx in range(imgs.size(0)):
            if batch_processed>= num_examples:
                break

            img=imgs[idx].cpu().numpy().squeeze()
            msk=masks[idx].cpu().numpy().squeeze()
            predicted_mask=predicted_masks[idx].cpu().numpy()


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
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask.squeeze(0)  # Remove the singleton dimension
                #print(f"image shape:{img.shape}")
                #print(f"mask shape:{mask.shape}")
                overlay = np.zeros((img.shape[0], mask.shape[1], 3))
                for value, color in layer_colors.items():
                    overlay[mask == value] = color
                return overlay

            true_overlay = create_overlay(img, msk, layer_colors)
            predicted_overlay = create_overlay(img, predicted_mask, layer_colors)
            #print(f"max prediction{np.max(predicted_mask)}")
            #print(f"max msk: {np.max(msk)}")


            original_rgb = np.stack([img] * 3, axis=-1)
            min_val = original_rgb.min()
            max_val = original_rgb.max()
            # Normalize based on observed min and max
            original_rgb = (original_rgb - min_val) / (max_val - min_val)
            true_combined = np.clip(0.7 * original_rgb + 0.3 * true_overlay, 0, 1)
            predicted_combined = np.clip(0.7 * original_rgb + 0.3 * predicted_overlay, 0, 1)
            if DPSGD==True:
                state='DPSGD'
            else:
                state=''
            # Plotting
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(true_combined)
            axes[1].set_title('True Mask')
            axes[1].axis('off')

            axes[2].imshow(predicted_combined)
            axes[2].set_title('Predicted Mask val')
            axes[2].axis('off')
            # correct path
            #file_path = os.path.join(folder_name, f'{model_name}_{state}_{dataset}_{idx+1*number}.pdf')
            #plt.savefig(file_path, format='pdf', bbox_inches='tight')

            # this is just temporary place to check the real impact of morphology
            file_path=f'/home/parsar0000/oct_git/main_code/Attack/evaluate_morphology/output_results_{args.morphology}.png'
            plt.savefig(file_path)
            plt.close()
            #plt.show()
            batch_processed += 1

def segmentation_plots_label_less(val_loader, model, device,model_name,dataset,DPSGD, num_examples=1):

    #folder_name = f"{'' if not DPSGD else 'DPSGD'}_{dataset}_images_label_less"
    """if not os.path.exists(folder_name):
        os.makedirs(folder_name)"""
    batch_processed=0
    model.eval()
    for imgs in val_loader:
        imgs = imgs.to(device)
        with torch.no_grad():  # We do not need to compute gradients here
            preds = model(imgs)
            _, predicted_masks = torch.max(preds, dim=1)

        for idx in range(imgs.size(0)):
            if batch_processed>= num_examples:
                break

            img=imgs[idx].cpu().numpy().squeeze()
            predicted_mask=predicted_masks[idx].cpu().numpy()

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


            predicted_overlay = create_overlay(img, predicted_mask, layer_colors)

            original_rgb = np.stack([img] * 3, axis=-1)
            min_val = original_rgb.min()
            max_val = original_rgb.max()
            # Normalize based on observed min and max
            original_rgb = (original_rgb - min_val) / (max_val - min_val)


            predicted_combined = np.clip(0.7 * original_rgb + 0.3 * predicted_overlay, 0, 1)
            if DPSGD==True:
                state='DPSGD'
            else:
                state=''
            # Plotting
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title('Original Image/ DME')
            axes[0].axis('off')

            axes[1].imshow(predicted_combined)
            axes[1].set_title('Predicted Mask/ Synthetic')
            axes[1].axis('off')
            #file_path = os.path.join(folder_name, f'{model_name}_{state}_{dataset}_{idx}_{"label_less"}.pdf')
            #plt.savefig(file_path, format='pdf', bbox_inches='tight')
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

def plot_train_test(args,training_losses,validation_losses):
    # plot train and validation
    epochs = len(training_losses)
    epochs_range=range(1,epochs+1)
    plt.plot(epochs_range, training_losses, label="training")
    plt.plot(epochs_range, validation_losses, label="validation")
    plt.title(f"Training and validation loss when morphology is {args.morphology}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    #plt.show()
    #plt.close()
    # this is just temporary place to check the real impact of morphology
    file_path = f'/home/parsar0000/oct_git/main_code/Attack/evaluate_morphology/train_test_split_{args.morphology}.png'
    plt.savefig(file_path)
    plt.close()


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_files(path, ext):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(ext)]





def apply_kornia_morphology_multiclass(pred_mask:torch.Tensor, operation:str='both',kernel_size:int=3, probability=0.5) ->torch.Tensor:
    """
        Apply morphological operations using Kornia to refine multi-class predicted masks.

        Args:
            pred_mask (torch.Tensor): Multi-class mask of shape [B, C, H, W], where C = num_classes.
            operation (str): 'open', 'close', 'both', or 'none'.
            kernel_size (int): Size of the structuring element.

        Returns:
            torch.Tensor: Refined multi-class mask of shape [B, C, H, W].
        """
    choices=['open', 'close', 'both', 'none','dilation','erosion']
    if operation not in ['open', 'close', 'both', 'none','dilation','erosion']:
        raise ValueError("Operation must be one of 'open', 'close', 'both', or 'none'.")
    """x = random.random()
    if x<probability:
        operation=operation
    else:
        operation = random.choice(choices)"""

    """if x <= 0.26:
        operation = 'close'
    elif 0.26 < x <= 0.5:
        operation = 'open'
    elif 0.5 < x <= 0.75:
        operation = 'both'
    else:
        operation = operation"""
    #operation= dp_select_from_choices(choices, operation, privacy_budget=0.1)

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
    elif operation == 'none':
        refined_mask = pred_mask
    else:
        raise ValueError('open', 'close', 'both', 'none','dilation','erosion')
    return refined_mask


def eval(args,val_loader, criterion, model, n_classes, dice_s=True, device="cuda", im_save=False):
    model.eval()
    loss = 0
    counter = 0
    dice = 0

    dice_all = np.zeros(n_classes)

    for img, label in tqdm.tqdm(val_loader):
        img = img.to(device)
        label = label.to(device)
        if n_classes>1:
            label_oh = torch.nn.functional.one_hot(label, num_classes=n_classes)
            pred = model(img)
            #if args.morphology:
            #    pred = apply_kornia_morphology_multiclass(pred, operation=args.operation, kernel_size=args.kernel_size)
            max_val, idx = torch.max(pred, 1) # extracts the predicted class for each pixel.
            pred_seg = idx.cpu().data.numpy()
            label_seg = label.cpu().data.numpy()
        else:
            pred=model(img)
            #if args.morphology:
            #    pred = apply_kornia_morphology_multiclass(pred, operation=args.operation, kernel_size=args.kernel_size)
            label_oh= label
            #Apply sigmoid and threshold to get binary predictions
            pred = torch.sigmoid(pred)
            pred_seg = (pred > 0.5).float().cpu().data.numpy()
            print(f"max prediction in evaluation phase:{np.max(pred_seg)}")
            label_seg = label.cpu().data.numpy()
            pred_oh = torch.tensor(pred_seg) # For binary segmentation, the prediction is already one-hot
            pred_oh=pred_oh.to(device)
        ret= compute_dice(label_seg, pred_seg)
        print(f"dice score: {ret}")
        pa =compute_pa(label_seg, pred_seg)
        print(f"pa: {pa}")


        if dice_s:
            if n_classes>1:
                pred_oh = torch.nn.functional.one_hot(idx, num_classes=n_classes)
                #print("shapes before : ", label_oh.shape, pred_oh.shape)
                d1, d2 = per_class_dice(pred_oh, label_oh, n_classes)
                dice += d1
                dice_all += d2
            else:

                d1, d2 = per_class_dice(pred_oh, label_oh, n_classes)
                dice+=d1
                dice_all+=d2

        if label.shape == pred.shape:
            loss+= criterion(pred, label, device=device).item()
        else:
            loss += criterion(pred, label.squeeze(1), device=device).item()

        counter += 1

    loss = loss / counter
    dice = dice / counter
    dice_all = dice_all / counter
    if n_classes>1:
        print("Validation loss: ", loss, " Mean Dice: ", dice, "Dice All:", dice_all)
    else:
        print("Validation loss: ", loss, " Mean Dice: ", dice, "Dice All:", dice_all)

    return dice, loss, dice_all




def remove_inplace(m):
    for mod in m.modules():
        if hasattr(mod, "inplace"):
            mod.inplace = False

        # Avoid common in-place layers/ops by type

        if isinstance(mod, (nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ELU, nn.SiLU, nn.GELU, nn.Dropout, nn.AlphaDropout)):
            if hasattr(mod, "inplace"):
                mod.inplace = False



def seg_train(args,train_loader,val_loader,counter,victim=True,shadow=False,plotting=False,multiple_shadow=False):

    if args.morphology:
        print("**********Morphology mode*************")
    device = args.device
    if multiple_shadow:
        n_classes=1
    else:
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
    validation_dice_scores = []
    test_losses=[]
    test_dice_scores=[]
    max_consecutive_epochs_without_improvement = 10
    consecutive_epochs_without_improvement = 0
    best_test_loss = float("inf")
    criterion_seg = CombinedLoss()
    criterion_ffc = FocalFrequencyLoss()
    """if args.DPSGD:
        save_name = model_name + "DPSGD" + ".pt"
    else:
        save_name = model_name  + ".pt"""

    # save_name = model_name  + ".pt"

    max_dice = 0
    best_test_dice = 0
    best_iter = 0
    print(f"n_classes:{n_classes}")
    if args.morphology:
        model=get_model(model_name, ratio=ratio, num_classes=n_classes,morphology=args.morphology,operation=args.operation,kernel_size=args.kernel_size)
    else:
        model = get_model(model_name, ratio=ratio, num_classes=n_classes).to(device)
    print(model_name)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate,
                                 weight_decay=args.weight_decay)
    # load trained model just for victim model

    start_epoch=0
    # for shadow model is commented to enforce shadow model training
    if args.model_should_be_load:
        if victim:
            if args.DPSGD==True:
                state='DPSGD'
            else:
                state='normal'
            model,optimizer,start_epoch,validation_loss,average_loss = load_model_if_exists(args,model,model_name,optimizer,args.main_dir,victim=victim,state=state)

        #elif shadow:
        #    model, optimizer, start_epoch, validation_loss, average_loss = load_model_if_exists(args,model, model_name, optimizer,args.main_dir, counter,shadow=shadow,state="")

    """for images, labels in train_loader:
        visualize_batch(images, labels)
        break"""

    # delta = 1 / (number_of_images ** 1.1)
    if args.DPSGD==True:
        delta = 1e-5
        privacy_engine = PrivacyEngine()
        noise_multiplier = 1
        max_grad_norm = 1
        model_name=f"{args.model_name}_DPSGD"

        """model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm, )"""
        remove_inplace(model) # inplaces should be False  # Always avoid in-place operations when using Opacus unless you explicitly clone tensors.
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            # noise_multiplier=noise_multiplier,
            target_epsilon=args.epsilon,
            target_delta=delta,
            epochs=iterations,
            max_grad_norm=max_grad_norm, )
        print("DPSGD is active in the training process")
        privacy_epsilons = []


    else:
        noise_multiplier = 'None'
        max_grad_norm = 'None'
        delta='None'
        model_name=args.model_name

    model.train()

    for t in range(start_epoch,iterations):
        print(""""starting training loop""")
        """if t>0:
            torch.cuda.empty_cache()"""
        total_loss = 0
        total_samples = 0

        for img, label in (train_loader):
            img = img.to(device)
            label = label.to(device)
            label_oh = torch.nn.functional.one_hot(label, num_classes=n_classes).squeeze()
            pred = model(img)
            # print(pred.shape)
            # print("morphological operation")
            # refined_image= apply_kornia_morphology_multiclass(pred, operation='dilation', kernel_size=3)
            # print(refined_image.shape)

            max_val, idx = torch.max(pred, 1)
            pred_oh = torch.nn.functional.one_hot(idx, num_classes=n_classes)
            #pred_oh = pred_oh.permute(0, 3, 1, 2)
            #label_oh = label_oh.permute(0, 3, 1, 2)
            """loss = criterion_seg(pred, label.squeeze(1), device=device) + args.ffc_lambda * criterion_ffc(pred_oh,
                                                                                                       label_oh)"""
            # print(pred.shape)
            # print(label.squeeze(1).shape)
            #if args.morphology: it is handled in the network structure
            #    pred = apply_kornia_morphology_multiclass(pred, operation=args.operation, kernel_size=args.kernel_size)
            """if label.shape == pred.shape:
                loss = criterion_seg(pred, label, device=device)
            else:
                loss = criterion_seg(pred, label.squeeze(1), device=device)"""
            loss = criterion_seg(pred, label.squeeze(1), device=device)

            optimizer.zero_grad() #zero_grad clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * img.size(0)
            total_samples += img.size(0)

        average_loss = total_loss / total_samples  # for each epoch
        #training_losses.append(average_loss) # remove it from here to have a same size validation and train/but if you need to fully track the training loss, here it is the place to update training_losses

        if args.DPSGD==True:
            epsilon = privacy_engine.get_epsilon(delta)
            privacy_epsilons.append(epsilon)

            print(
                f"\tTrain Epoch: [{t + 1}/{iterations}] \t"
                f"Train Loss: {np.mean(average_loss):.6f} "
                f"(ε = {epsilon:.2f}, δ = {delta})"
            )
        else:
            print(
                f"\tTrain Epoch: [{t + 1}/{iterations}] \t"
                f"Train Loss: {np.mean(average_loss):.6f} ")

        """if t % 50 == 0:  # Every 50 epochs
            plot_examples(val_loader, model, device)"""

        if t % 5== 0 or t > 45:
            print("Validation")
            dice, validation_loss, dice_all = eval(args,val_loader, criterion_seg, model, dice_s=True, n_classes=n_classes)
            validation_losses.append(validation_loss)
            training_losses.append(average_loss)
            if plotting:
                 if shadow:
                     dataset="synthetic"
                 else:
                     dataset=args.main_dir
                 segmentation_plots(args,val_loader, model, device, model_name, multiple_shadow, DPSGD=args.DPSGD, dataset=dataset ,morphology=args.morphology,number=t,num_examples=1)

            if n_classes>1:
                validation_dice_scores.append(dice.item())

            else:
                validation_dice_scores.append(dice)
            # print("Expert 1 - Test")
            # dice_test = eval(test_loader, criterion_seg, model, n_classes=n_classes)

            if dice > max_dice:
                max_dice = dice
                best_iter = t
                # best_test_dice = dice_test
                #print(colored_text("Updating model, epoch: "), t)
                # torch.save(model, save_name)
                """if args.model_should_be_saved:
                    torch.save(model.state_dict(), save_name)"""

        #if  plotting and (t + 1) % 20 == 0:
            #segmentation_plots(args,val_loader, model, device, model_name, multiple_shadow, args.DPSGD,args.image_dir)

        model.train()

        if validation_loss < best_test_loss : # if there is any improvement
            best_test_loss= validation_loss
            consecutive_epochs_without_improvement = 0  # reset
        else:
            consecutive_epochs_without_improvement +=1
        if consecutive_epochs_without_improvement >= max_consecutive_epochs_without_improvement:
            print(
                f"Stopping training due to lack of improvement for {max_consecutive_epochs_without_improvement} epochs.")
            break  # Exit the training loop


        if (t + 1) % 20 == 0 and args.model_should_be_saved:
            print("Saving checkpoint...")
            if victim:
                if args.DPSGD:
                    save_model(args,model, model_name, optimizer, t, validation_loss,average_loss,args.main_dir,victim=victim, state='DPSGD')
                else:
                    save_model(args,model, model_name, optimizer, t, validation_loss,average_loss,args.main_dir,victim=victim,state='normal')
            elif shadow:
                save_model(args,model, model_name, optimizer, t, validation_loss,average_loss,args.main_dir,shadow=True,state='')


    #segmentation_plots(args,val_loader, model, device, model_name, args.DPSGD, args.image_dir)
    # print("Best iteration: ", best_iter, "Best val dice: ", max_dice, "Best test dice: ", best_test_dice)
    if len (training_losses) > 0:
      if shadow:
          print("shadow training")
      print(f" training loss:{training_losses}")
      print(f" validation loss:{validation_losses}")
      if not shadow: # this is used for evaluating the real performance of the model
          dataset=args.main_dir
          segmentation_plots(args, val_loader, model, device, model_name, multiple_shadow, DPSGD=args.DPSGD,
                             dataset=dataset, morphology=args.morphology, number=t, num_examples=3)
          plot_train_test(args,training_losses, validation_losses) # morph-vs-non-morph.ipynb:colab
          #sys.exit()

      file_name = 'losses.txt'
      with open(file_name, 'a') as file:  # Use 'a' to append data to the file
          # Write the training losses
          file.write(f"Training loss {counter}: {training_losses}\n")
          # Write the validation losses
          file.write(f"Validation loss {counter}: {validation_losses}\n")
    #if plotting:
        #segmentation_plots(args,val_loader, model, device, model_name, multiple_shadow, args.DPSGD, args.image_dir)

    return model, average_loss



def train_synthetic(model, data_loader, device,model_name,dataset_synthetic, save_path,DPSGD_for_Saving_and_synthetic,visualize=False):
    model.eval() # we are not training, we are labeling the dataset
    os.makedirs(save_path, exist_ok=True)
    image_save_path=os.path.join(save_path,'images')
    mask_save_path= os.path.join(save_path,'masks')
    os.makedirs(image_save_path, exist_ok=True)
    os.makedirs(mask_save_path, exist_ok=True)

    existing_images = len(os.listdir(image_save_path))
    existing_masks = len(os.listdir(mask_save_path))
    print(f"existing images: {existing_images}")
    i=existing_images
    """segmentation_plots_label_less(data_loader, model, device, model_name, dataset_synthetic,
                                  DPSGD=DPSGD_for_Saving_and_synthetic, num_examples=10)"""
    for imgs in data_loader:
        imgs = imgs.to(device)
        with torch.no_grad():  # We do not need to compute gradients here
            preds = model(imgs)
            # how should I save prediction as masks and imgs as input
            # It is already DatasetOct, so just a getim to return img and masks would be enough
            #if visualize:
            #segmentation_plots_label_less(test_loader, model, device,model_name,dataset_synthetic, DPSGD=DPSGD_for_Saving_and_synthetic, num_examples=1)

            # save as masks
            _, predicted_masks = torch.max(preds, dim=1)
            for idx in range(imgs.size(0)):
                img = imgs[idx].cpu().numpy().squeeze()
                predicted_mask = predicted_masks[idx].cpu().numpy()
                # I should save the images and masks
                img_filename = os.path.join(image_save_path,f"{dataset_synthetic}_{i+idx}.npy")
                np.save(img_filename, img)
                # Save the mask
                mask_filename = os.path.join(mask_save_path,f"{dataset_synthetic}_{i+idx}.npy")
                np.save(mask_filename, predicted_mask)
        i+= imgs.size(0)


# scp /home/parsar0000/pythonProject4/training.py shiva.parsarad@unibas.ch@chinchilla.dmi.unibas.ch:/home/parsar0000/BPR/pythonProject4
