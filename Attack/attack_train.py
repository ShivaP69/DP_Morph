from args import ATTACK_BATCH_SIZE
import torch
from torchvision import models
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import torch.nn as nn
from losses import CrossEntropyLoss2d
import os
import csv
from train import apply_kornia_morphology_multiclass
import torch.nn.functional as F
import time
import matplotlib
matplotlib.use('Agg')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Show prediction for the first sample
def plot_shadow_prediction(pred,labels,target,save=True, save_dir='plots',j=0):
    if save and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    batch_size=pred.shape[0]
    for i in range(batch_size):

        pred_class = torch.argmax(pred[i], dim=0).cpu()  # shape: [224, 224]
        label_img = labels[i][0].cpu()  # shape: [224, 224]
        target_img = target[i].cpu()
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title(f"Predicted for target{target_img}")
        plt.imshow(pred_class, cmap='tab10')
        plt.subplot(1, 2, 2)
        plt.title("Ground Truth")
        plt.imshow(label_img, cmap='tab10')
        #plt.show()
        plt.tight_layout()
        if save:
            plt.savefig(f"{save_dir}/prediction_{i}_{j}.png")
            plt.close()
        #plt.show()



# for attack directory just attack performance has been saved not the model's performance>> better to save model's performance
def save_results_to_csv(args, true_labels, pred_labels_roc, tn,fp,fn,tp,filename='new_metrics.csv'):
    if args.morphology:
        filename = 'new_metrics_morphology_dp.csv'
    try:
        # Convert true_labels and pred_labels_roc to strings to store in CSV
        true_labels_str = ','.join(map(str, true_labels))
        pred_labels_roc_str = ','.join(map(str, pred_labels_roc))
        if args.patchify:
            data_representation=args.data_representation
            patch_size= args.patch_size
            num_patches= args.num_patches
        else:
            data_representation=None
            patch_size=None
            num_patches= None
        if args.DPSGD_for_Saving_and_synthetic:
            epsilon=args.epsilon
        else:
            epsilon=None

        # Prepare the row data for CSV

        if args.morphology:
            row_data = [
                f"{args.main_dir}_synthetic_{args.synthetic_data}_{args.attacktype}_multiple_shadow_{args.multiple_shadow}_DSPGD_{args.DPSGD_for_Saving_and_synthetic}_patchify_{args.patchify}",
                [true_labels_str],
                [pred_labels_roc_str],
                ATTACK_BATCH_SIZE,  # Ensure ATTACK_BATCH_SIZE is part of args or defined properly
                args.batch_size,
                tn, fp, fn, tp, num_patches, patch_size, args.num_iterations, epsilon, data_representation, args.operation, args.kernel_size
            ]
        else:
            row_data = [
                f"{args.main_dir}_synthetic_{args.synthetic_data}_{args.attacktype}_multiple_shadow_{args.multiple_shadow}_DSPGD_{args.DPSGD_for_Saving_and_synthetic}_patchify_{args.patchify}",
                [true_labels_str],
                [pred_labels_roc_str],
                ATTACK_BATCH_SIZE,  # Ensure ATTACK_BATCH_SIZE is part of args or defined properly
                args.batch_size,
                tn, fp, fn, tp, num_patches, patch_size, args.num_iterations, epsilon, data_representation
            ]


        # Debugging: Print the data to be written
        print(f"Row data to write: {row_data}")

        # Check if the file exists
        file_exists = os.path.isfile(filename)

        # Debugging: Check if the file already exists
        print(f"Does the file exist? {file_exists}")

        # Open the file for appending (or create it if it doesn't exist)
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                if args.morphology:
                    writer.writerow(
                        ["sample_names", "true_labels", "pred_labels_roc", "ATTACK_BATCH_SIZE",
                         "victim_shadow_batch_size", "TN", "FP", "FN", "TP", 'num_patches', 'patch_size',
                         'num_iterations', 'epsilon', 'data_representation',"operation", "kernel_size"]
                    )
                else:
                    # Write the header only if the file doesn't exist
                    writer.writerow(
                        ["sample_names", "true_labels", "pred_labels_roc", "ATTACK_BATCH_SIZE", "victim_shadow_batch_size","TN","FP","FN","TP",'num_patches','patch_size','num_iterations','epsilon','data_representation']
                    )
            # Write the row of data
            writer.writerow(row_data)

        # Get the absolute path of the file
        file_path = os.path.abspath(filename)
        print(f"Results saved to {file_path}")

    except Exception as e:
        print(f"Failed to save results to CSV: {e}")


class BinaryClassifier(nn.Module):
    def __init__(self, input_channels):
        super(BinaryClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Initialize a placeholder for the fully connected layer, we'll define it later.
        self.fc = None

    def initialize_fc(self, x):
        """ Initializes the fully connected layer based on the input size """
        # Pass a dummy input through conv layers to get the flattened size
        x = self.conv(x)
        flattened_size = x.view(x.size(0), -1).size(1)  # Compute flattened size

        # Now, define the fully connected layers based on this flattened size
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 128),  # Use computed flattened size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),  # Binary classification output
            #nn.Sigmoid()  # Sigmoid to return probability between 0 and 1
        )

        # Move the fully connected layer to the same device as the input `x`
        self.fc = self.fc.to(x.device)

    def forward(self, x):
        # Initialize the fully connected layers on the first forward pass
        if self.fc is None:
            self.initialize_fc(x)
        print(f"Input device: {x.device}")  # Check input tensor device
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the output from conv layers

        print(f"Conv output device: {x.device}")  # Check output tensor device
        x = self.fc(x)
        return x



def VGG(ATTACK_INPUT_CHANNELS):
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    # Modify the first layer to accept ATTACK_INPUT_CHANNELS
    model.features[0] = nn.Conv2d(ATTACK_INPUT_CHANNELS, 64, kernel_size=3, padding=1, bias=False)
    # Modify the classifier to output a single probability
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 1),
        # nn.Sigmoid()
    )
    return model


def resnet(ATTACK_INPUT_CHANNELS):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # model = models.resnet50(weights=None)
    # Modify the first layer to accept ATTACK_INPUT_CHANNELS
    model.conv1 = nn.Conv2d(ATTACK_INPUT_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Modify the classifier to output a single probability
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        # nn.Dropout(0.5),
        nn.Linear(512, 1),
        #nn.Sigmoid()
    )
    return model

class Attacker(nn.Module):
    """
        Attack model without anchor selector, implemented as the 4-layer MLP
    """
    def __init__(self, input_dim, hidden_dim):
        super(Attacker, self).__init__()
        self.attacker = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                       nn.Linear(hidden_dim, 1))
    def forward(self, x):
        out = self.attacker(x)
        #return torch.sigmoid(out)
        return out




def mixup_data(pred, lam):
    """Apply Mixup to the predictions"""
    batch_size = pred.size(0)
    index = torch.randperm(batch_size).to(pred.device)
    mixed_pred = lam * pred + (1 - lam) * pred[index, :]
    return mixed_pred

def SLM(pred, label, ignore=255):
    sz = label.shape
    #print(f"sz:{sz.shape}")
    loss = torch.zeros(sz, device=pred.device)  # Shape: (H, W)
    #print(f"loss:{loss.shape}")

    unique_label = torch.unique(label)

    for i in unique_label:
        if i == ignore:
            continue

        mask = (label == i)  # Shape: (H, W)
        #print(f"mask:{mask.shape}")


        tmp = pred[i, :, :] * mask
        tmp[tmp < 1e-30] = 1e-30
        tmp = -torch.log(tmp)
        #print(f"tmp:{tmp.shape}")
        loss[mask] = tmp[mask]

    return loss

def normalize_labels_for_onehot(labels):
    """
    Ensure labels is integer tensor with shape [B, H, W] suitable for F.one_hot(..., num_classes).
    Handles common incoming shapes like [B, H, W], [B,1,H,W], or [B,H,W,1].
    """
    # move to CPU? keep on device for speed; we operate on same device
    # Remove any singleton channel dimension if present
    if labels.dim() == 4 and labels.size(1) == 1:
        labels = labels.squeeze(1)       # [B,1,H,W] -> [B,H,W]
    elif labels.dim() == 4 and labels.size(-1) == 1:
        # [B,H,W,1] -> [B,H,W]
        labels = labels.squeeze(-1)
    elif labels.dim() == 5:
        # Unexpected 5D: try to squeeze all singleton dims until 3D or raise
        labels = labels.squeeze()
    # final sanity: ensure we have [B,H,W]
    if labels.dim() != 3:
        raise RuntimeError(f"normalize_labels_for_onehot: expected labels with 3 dims [B,H,W], got {labels.shape}")
    return labels.long()



def train_attack_model(args, shadow_model, victim_model, dataloader, val_dataloader, epochs, lr,multiple_shadow=False,layer=-1,model_name='Binary_Classifier'):
    # argmax defense
    from args import OUTPUT_CHANNELS
    if multiple_shadow:
        print("multiple shadow")
        n_classes = 1
        OUTPUT_CHANNELS=1

    else:
        n_classes = args.n_classes
        OUTPUT_CHANNELS= OUTPUT_CHANNELS


    if args.defensetype == 2:
        # Type-I attack
        if args.attacktype == 1:
            ATTACK_INPUT_CHANNELS = 1
        # Type-II attack
        else:
            ATTACK_INPUT_CHANNELS = 2

    # non argmax defense
    else:
        # Type-I attack
        if args.attacktype == 1:
            ATTACK_INPUT_CHANNELS = OUTPUT_CHANNELS
        # Type-II attack
        else:
            ATTACK_INPUT_CHANNELS = OUTPUT_CHANNELS * 2  # label and predco
    print(f"size input channels:{ATTACK_INPUT_CHANNELS}")
    #  change BCEloss to BCEWithLogitsLoss(): to have consistent structure for all models and de-activate sigmoid layers for all models
    if model_name=="Binary_Classifier":
        model = BinaryClassifier(ATTACK_INPUT_CHANNELS)
        criterion = nn.BCEWithLogitsLoss()

    elif model_name== "VGG":
        model=VGG(ATTACK_INPUT_CHANNELS)
        criterion = nn.BCEWithLogitsLoss()
    elif model_name=="resnet":
        model=resnet(ATTACK_INPUT_CHANNELS)
        criterion = nn.BCEWithLogitsLoss()
    elif model_name=='attacker':
        model = Attacker(input_dim=ATTACK_INPUT_CHANNELS, hidden_dim=128)
        criterion = nn.BCEWithLogitsLoss()
    model.to(device)

    opt = torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

    q_accuracy = []
    q_f1 = []

    for epoch in range(epochs):
        if epoch>0:
            torch.cuda.empty_cache()
        model.train()
        pred_labels = np.array([])
        true_labels = np.array([])
        print(' -- Staring training epoch {} --'.format(epoch + 1))
        num_ones=0
        num_zeros=0
        for data, labels, targets in dataloader:
            print(f'Number of images in this batch: {len(data)}')
            model.train()
            data, labels, targets = data.to(device), labels.to(device), targets.to(device)
            num_ones += torch.sum(targets == 1).item()  # Count number of 1s
            num_zeros += torch.sum(targets == 0).item()  # Count number of 0s

            opt.zero_grad()
            with torch.no_grad():
                pred = shadow_model(data)
                # in this implementation, morphology is not part of the model so we should apply it manually: we do no need it here if morphology was not part of the post processing
                """if args.morphology:
                   pred = apply_kornia_morphology_multiclass(pred, operation=args.operation, kernel_size=args.kernel_size)"""
                # what is pred?
                #print(f"shape pred of shadow model is {pred.shape}")
                #print(f"shape labels {labels.shape}")
                ## plot to understand if there is any difference between shadow model output for members and non-members
                #plot_shadow_prediction(pred,labels,targets,j=pred.shape[0]*epoch)

            # argmax defense
            if args.defensetype == 2:
                pred = torch.argmax(pred, 1, keepdim=True)
                labels = labels.float().unsqueeze(1)

                # Type-II attack
            if args.attacktype == 2:
                # non-argmax defense
                if multiple_shadow==False:
                    if args.defensetype != 2:
                        print(f"label shape before normalize:{labels.shape}")
                        pred = F.softmax(pred, dim=1)
                        labels_int=normalize_labels_for_onehot(labels)
                        print(f"label shape after normalize:{labels_int.shape}")
                        labels = nn.functional.one_hot(labels_int, n_classes).permute(0,3,1,2).float() # one_hot: [B, H, W] -> [B, H, W, C]; permute to [B, C, H, W]
                        print(f"label shape after one-hot:{labels.shape}")

                cat = labels.view(data.shape[0], pred.shape[1], data.shape[2], data.shape[3]).float()
                print(f"label shape after view:{cat.shape}")
                #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                #print(f"difference between cat and pred: {cat-pred}")
                s_output = torch.concat([pred, cat], dim=1)
                print(f"s_output shape:{s_output.shape}")
            else:
                s_output = pred
            #print(f"n_classes:{n_classes}")
            #print(f"S_output:{s_output}")
            s_output = F.avg_pool2d(s_output, kernel_size=8)
            output = model(s_output.float())
            loss = criterion(output.float(), targets.float().view(len(targets), 1))
            loss.backward()
            opt.step()
            output_probs = torch.sigmoid(output)
            #print(f"output_probs:{output_probs}")
            pred_l = output_probs.float().round().view(data.shape[0]).detach().cpu().numpy()
            #print(f"pred_l: {pred_l}")
            true_l = targets.float().view(data.shape[0]).detach().cpu().numpy()

            pred_labels = np.concatenate((pred_labels, pred_l))
            true_labels = np.concatenate((true_labels, true_l))
            print(f"pred_labels training:{pred_labels}")
            print(f"true_labels training:{true_labels}")

        print(
            'Training accuracy:', round(accuracy_score(true_labels, pred_labels), 4),
            ',Training  F1-score:', round(f1_score(true_labels, pred_labels), 4),
        )

        if not multiple_shadow:
            print("#######Testing#######")
            res = test_attack_model(args, model, val_dataloader, n_classes, victim_model=victim_model,
                                    input_channels=ATTACK_INPUT_CHANNELS, multiple_shadow=multiple_shadow, layer=layer,
                                    number=epoch, roc=False, plot_test=True)

        else:

            res=test_attack_model_multiple(args, [model], [val_dataloader], num_classes=1, victim_model=victim_model,roc=False)

        if epoch >= epochs - 10:
            q_accuracy.append(res[0])
            q_f1.append(res[1])

        print(f"Epoch {epoch + 1}: num_ones: {num_ones}, num_zeros: {num_zeros}")


    # at the end
    if not multiple_shadow:
          test_attack_model(args, model, val_dataloader, n_classes, victim_model=victim_model,
                          input_channels=ATTACK_INPUT_CHANNELS, multiple_shadow=multiple_shadow, layer=layer,number=epoch, roc=True,plot_test=False) # at the end so I put epoch, because the last results is the one for the last epoch


    print('\n\nLast 10 epochs testing averages: accuracy: {}, F1-score: {}'.format(
        round(np.mean(q_accuracy), 4),
        round(np.mean(q_f1), 4),
    ))
    print(f"num_ones for training:{num_ones}")
    print(f"num_zeros for training:{num_zeros}")

    return model


def test_attack_model(args, model, dataloader, num_classes, shadow_model=None, victim_model=None, accuracy_only=False,
                      input_channels=1,multiple_shadow=False,layer=-1,number=1,roc=False,plot_test=False, different_shadow=False):
    pred_labels = np.array([])
    true_labels = np.array([])
    pred_labels_roc=np.array([])
    #criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()

    # Testing loop
    model.eval()
    for data, labels, targets in dataloader:
        data, labels, targets = data.to(device), labels.to(device), targets.to(device)
        with torch.no_grad():
            if shadow_model == None:
                pred = victim_model(data)
                #if args.morphology:
                #    pred = apply_kornia_morphology_multiclass(pred, operation=args.operation, kernel_size=args.kernel_size)
                """if multiple_shadow:
                    pred=pred[:,layer,:,:]"""
            else:
                pred = shadow_model(data)
                #if args.morphology:
                #    pred = apply_kornia_morphology_multiclass(pred, operation=args.operation, kernel_size=args.kernel_size)
            if plot_test:
                print("%%%%%%%%%%%This is Working%%%%%%%%%")
                if args.morphology:
                    plot_shadow_prediction(pred, labels, targets, save_dir="morph_test_attack", j=pred.shape[0] * number)
                else:
                    plot_shadow_prediction(pred, labels, targets, save_dir="test_attack",
                                           j=pred.shape[0] * number)
            # argmax defense
            if args.defensetype == 2:
                pred = torch.argmax(pred, 1, keepdim=True)


        if args.mixup:
            lam = np.random.beta(0.4, 0.4)
            pred = mixup_data(pred, lam)

            # One-hot encode the labels
            one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()

            # Check the shape of one_hot_labels for debugging
            print(f"one_hot_labels shape: {one_hot_labels.shape}")

            # Adjust the shape of one_hot_labels to match the shape of `pred`
            # From shape [batch_size, height, width, num_classes] to [batch_size, num_classes, height, width]
            one_hot_labels = one_hot_labels.permute(0, 4, 1, 2, 3).squeeze(2)

            print(f"Adjusted one_hot_labels shape: {one_hot_labels.shape}")

            # Concatenate predictions with one-hot encoded labels
            s_output = torch.cat((pred, one_hot_labels), dim=1)
        else:
            # Type-II attack
            if args.attacktype == 2:
                # non-argmax defense
                if args.defensetype != 2:
                    labels_int = normalize_labels_for_onehot(labels)
                    labels = nn.functional.one_hot(labels_int, num_classes=num_classes).permute(0,3,1,2).float()

                cat = labels.view(data.shape[0], pred.shape[1], data.shape[2], data.shape[3]).float()
                s_output = torch.concat((pred, cat), dim=1)
            else:
                s_output = pred
        s_output = F.avg_pool2d(s_output, kernel_size=8)
        output = model(s_output.float())
        loss = criterion(output.float(), targets.float().view(len(targets), 1))

        #print("validation loss:", loss.item())
        output = torch.sigmoid(output)
        pred_l = output.float().round().view(data.shape[0]).detach().cpu().numpy()
        true_l = targets.float().view(data.shape[0]).detach().cpu().numpy()
        pred_roc = output.view(data.shape[0]).detach().cpu().numpy() # Use raw predictions for ROC
        pred_labels = np.concatenate((pred_labels, pred_l))
        true_labels = np.concatenate((true_labels, true_l))
        pred_labels_roc=np.concatenate((pred_labels_roc, pred_roc))
        print(f"pred_labels_test:{pred_l}")
        print(f"true_labels_test:{true_l}")

    print("validation loss:", loss.item())
    a_score = round(accuracy_score(true_labels, pred_labels), 4)
    f_score = round(f1_score(true_labels, pred_labels), 4)

    if accuracy_only:
        print('Validation accuracy:', accuracy_score)
    else:
        print(
            'Validation accuracy:', a_score,
            ', F1-score:', f_score,
        )

        tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()

        print('TN: {}, FP: {}, FN: {}, TP: {}'.format(tn, fp, fn, tp))


        # Save true_labels and pred_labels_roc to CSV
    if roc:

        save_results_to_csv(args,true_labels, pred_labels_roc,tn,fp,fn,tp,)
        # plot
        print(f"final true labels: {true_labels}")
        print(f"final predicted labels: {pred_labels_roc}")
        print(f"final predictions:{pred_labels}")

        fpr, tpr, thresholds = roc_curve(true_labels, pred_labels_roc)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        if args.DPSGD_for_Saving_and_synthetic:
         plt.title(f"{args.attacktype} {ATTACK_BATCH_SIZE} {args.epsilon}")
        else:
            plt.title(f"{args.attacktype} {ATTACK_BATCH_SIZE}")
        plt.legend(loc="lower right")
        #plt.show()
        #plt.close()
        current_time=time.time()
        save_curves="curves"
        if not os.path.exists(save_curves):
            os.makedirs(save_curves)
        plt.savefig(f"{save_curves}/roc_current_time.png")
    if different_shadow:
        return a_score, f_score, pred_labels,true_labels

    return a_score, f_score



class BinaryClassifier(nn.Module):
    def __init__(self, input_channels):
        super(BinaryClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = None  # Placeholder for the Linear layer

    def initialize_fc(self, x):
        # Pass a dummy input through conv layers to get flattened size
        x = self.conv(x)
        flattened_size = x.view(x.size(0), -1).size(1)  # Calculate flattened size
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 128),  # Use calculated size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            #nn.Sigmoid()
        )
        self.fc = self.fc.to(x.device)

    def forward(self, x):
        if self.fc is None:  # Initialize FC layer dynamically on first forward pass
            self.initialize_fc(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x


def modify_model(base_model, input_channels):
    if isinstance(base_model, models.VGG):
        base_model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False)
        base_model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 224),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(224, 112),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(112, 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    elif isinstance(base_model, models.ResNet):
        base_model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=3, bias=False)
        num_ftrs = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    elif base_model == 'custom':
        base_model= BinaryClassifier(input_channels)

    return base_model



def train_attack_model_patchify(args, shadow_model, victim_model, dataloader, val_dataloader, epochs, lr,multiple_shadow=False,layer=-1):

    # argmax defense
    from args import OUTPUT_CHANNELS

    n_classes = args.n_classes
    OUTPUT_CHANNELS= OUTPUT_CHANNELS
    if args.data_representation == 'loss':
        ATTACK_INPUT_CHANNELS = OUTPUT_CHANNELS
    elif args.data_representation == 'concate':
        ATTACK_INPUT_CHANNELS = OUTPUT_CHANNELS * 2  # label and pred

    if args.attacktype == 1:
        ATTACK_INPUT_CHANNELS = OUTPUT_CHANNELS

    #model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    #model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    #model = models.resnet50(weights=None)
    #model = models.vgg16(weights=None)

    # Call the custom model
    #model = modify_model(model, ATTACK_INPUT_CHANNELS)

    model= modify_model('custom', ATTACK_INPUT_CHANNELS)


    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    criterion = nn.BCELoss()
    q_accuracy = []
    q_f1 = []

    for epoch in range(epochs):
        if epoch>0:
            torch.cuda.empty_cache()

        pred_labels = np.array([])
        true_labels = np.array([])

        model.train()
        print(' -- (Patchify) Staring attack training epoch {} --'.format(epoch + 1))
        num_ones=0
        num_zeros=0
        for data, labels, targets in dataloader:
            data, labels, targets = data.to(device), labels.to(device), targets.to(device)
            num_ones += torch.sum(targets == 1).item()  # Count number of 1s
            num_zeros += torch.sum(targets == 0).item()  # Count number of 0s

            opt.zero_grad()
            #print(f"labels: {labels.shape}")
            with torch.no_grad():
                pred = shadow_model(data) # it is in the one-Hot encoding format # Shape: (batch_size, n_classes, H, W)
            # pachify both labels and prediction

            number_patches = 0
            final_targets = torch.empty(0).to(device)
            final_output=torch.empty(0).to(device)
            labels= labels.squeeze()
            label_one_hot = nn.functional.one_hot(labels, n_classes).permute(0, 3, 1, 2)
            #print(f"label_one_hot: {label_one_hot.shape}")
            #acc_loss = 0.0
            #acc_count = 0

            if args.manuall:
                while number_patches < args.num_patches:
                    # Crop patches for the entire batch
                    bias_x = np.random.randint(pred.shape[2] - args.patch_size)
                    bias_y = np.random.randint(pred.shape[3] - args.patch_size)
                    patch_pred = pred[:, :, bias_x:bias_x + args.patch_size, bias_y:bias_y + args.patch_size]  # Shape: (batch_size, C, H, W)
                    patch_label = label_one_hot[:,:, bias_x:bias_x + args.patch_size, bias_y:bias_y + args.patch_size]  # Shape: (batch_size, H, W)

                    #print(f"patch_pred: {patch_pred.shape}")
                    #print(f"patch_label: {patch_label.shape}")
                    #print("patch_pred",patch_pred.shape)
                    #print("patch_label",patch_label.shape)
                    if args.attacktype == 1:
                        input_patch=patch_pred

                    elif args.attacktype == 2:
                        if args.data_representation == 'loss':
                            input_patch = SLM(patch_pred, patch_label) # shape: (batch_size, C, H, W)
                            #print(input_patch.shape)
                        elif args.data_representation=='concate':
                            input_patch = torch.cat((patch_pred, patch_label), dim=1)

                            #print(f"inpute_patch:{input_patch.shape}")# Shape: (batch_size, 2C, H, W)
                            #print(f"patch_pred:{patch_pred.shape}")
                            #print(f"patch_label:{patch_label.shape}")
                            #print(input_patch.shape)
                    output = model(input_patch)
                    # loss=criterion(output, targets)
                    # patch_loss = loss / accumulation_steps
                    # patch_loss.backward(retain_graph=(i < args.num_patches - 1))
                    # acc_loss += patch_loss.item()
                    # acc_count += 1
                    """if (i + 1) % accumulation_steps == 0:
                        # Once we accumulate a certain number, do an update
                        opt.step()
                        opt.zero_grad()
                        acc_loss = 0.0
                        acc_count = 0"""

                    # patch_targets = targets.float().view(len(targets), 1).detach()  # Detach target to avoid reuse

                    final_targets = torch.cat((final_targets, targets), dim=0)
                    final_output = torch.cat((final_output, output), dim=0)

                    number_patches += 1

            else:
                # Use unfold to extract patches for predictions and labels
                batch_size,_,_,_= pred.shape
                label_one_hot = label_one_hot.float()
                patch_pred = torch.nn.functional.unfold(pred, kernel_size=args.patch_size, stride=args.patch_size)
                patch_label = torch.nn.functional.unfold(label_one_hot, kernel_size=args.patch_size,
                                                         stride=args.patch_size)
                num_patches = patch_pred.shape[-1]  # Total patches per image
                #print(f"num_patches per image:{num_patches}")
                #print(f"Unfolded patches shape: {patch_pred.shape}")  # [B, C * patch_size^2, num_patches]
                targets = targets.repeat_interleave(num_patches)
                #print(f"Expanded targets shape: {targets.shape}")

                patch_pred = patch_pred.view(batch_size, n_classes, args.patch_size, args.patch_size,num_patches).permute(0, 4, 1, 2, 3)
                patch_label = patch_label.view(batch_size, n_classes, args.patch_size, args.patch_size,num_patches).permute(0, 4, 1, 2, 3)
                patch_pred = patch_pred.reshape(-1, n_classes, args.patch_size, args.patch_size)
                print(f"Reshaped patches shape: {patch_pred.shape}")  # [B * num_patches, C, patch_size, patch_size]
                patch_label= patch_label.reshape(-1, n_classes, args.patch_size, args.patch_size)
                #print(f"Patch prediction shape: {patch_pred.shape}")
                #print(f"Patch label shape: {patch_label.shape}")

                if args.attacktype == 1:
                    input_patch = patch_pred

                elif args.attacktype == 2:
                    if args.data_representation == 'loss':
                        input_patch = SLM(patch_pred, patch_label)  # shape: (batch_size, C, H, W)
                        # print(input_patch.shape)
                    elif args.data_representation == 'concate':
                        input_patch = torch.cat((patch_pred, patch_label), dim=1)
                        #print(f"inpute_patch:{input_patch.shape}")  # Shape: (batch_size, 2C, H, W)
                        #print(f"patch_pred:{patch_pred.shape}")
                        #print(f"patch_label:{patch_label.shape}")


                output = model(input_patch)
                final_targets = torch.cat((final_targets, targets), dim=0)
                final_output = torch.cat((final_output, output), dim=0)
            # Store outputs and targets for metrics calculation
            final_targets = final_targets.view(-1, 1)
            final_output = final_output.float()  # Convert final_output to float32
            final_targets = final_targets.float()  # Convert final_targets to float32
            #print(f"final_targets: {final_targets.shape}")
            #print(f"final_output: {final_output.shape}")
            loss=criterion(final_output, final_targets)
            loss.backward()
            opt.step()
            opt.zero_grad()
            """if acc_count > 0:
                opt.step()
                opt.zero_grad()"""
            if args.manuall:
                final_targets = (final_targets.float().view(data.shape[0] *args.num_patches).detach().cpu().numpy())
                final_output = (final_output.float().round().view(data.shape[0]*args.num_patches).detach().cpu().numpy())
            else:
                #print(f"final_targets: {final_targets.shape}")
                #print(f"data:{data.shape}")
                final_targets = (final_targets.float().view(data.shape[0]*num_patches).detach().cpu().numpy())
                final_output = (final_output.float().round().view(data.shape[0]*num_patches).detach().cpu().numpy())
            pred_labels = np.concatenate((pred_labels, final_targets))
            true_labels = np.concatenate((true_labels, final_output))
        print(f"pred_labels:{pred_labels}")   # I need pred_labels.argmax(dim=1)?
        print(
            'Training accuracy:', round(accuracy_score(true_labels, pred_labels), 4),
            ', F1-score:', round(f1_score(true_labels, pred_labels), 4),
        )
        print(f"testing the attack model performance in epcoh:{epoch}")
        res = test_attack_model_patchify(args, model, val_dataloader, victim_model=victim_model,
                                roc=False)
        model.train()
        if epoch >= epochs - 10:
            q_accuracy.append(res[0])
            q_f1.append(res[1])

    # at the end

    test_attack_model_patchify(args, model, val_dataloader, victim_model=victim_model, roc=True)

    print('\n\nLast 10 epochs testing averages: accuracy: {}, F1-score: {}'.format(
        round(np.mean(q_accuracy), 4),
        round(np.mean(q_f1), 4)
    ))

    return model



def test_attack_model_patchify(args, model, dataloader, shadow_model=None, victim_model=None, accuracy_only=False,roc=False):
    pred_labels = np.array([])
    true_labels = np.array([])
    pred_labels_roc=np.array([])
    criterion = nn.BCELoss()

    # Testing loop
    model.eval()
    total_loss=[]
    with torch.no_grad():
        for data, labels, targets in dataloader:
            #print("********************")
            #print(f"labels:{labels}")
            #print(f"targets:{targets}")
            #print("********************")
            data, labels, targets = data.to(device), labels.to(device), targets.to(device)
            with torch.no_grad():
                if shadow_model == None:
                    pred = victim_model(data)
                    """if multiple_shadow:
                        pred=pred[:,layer,:,:]"""

                else:
                    pred = shadow_model(data)

            number_patches = 0
            labels = labels.squeeze()
            label_one_hot = nn.functional.one_hot(labels, args.n_classes).permute(0, 3, 1, 2)
            print(f"label_one_hot: {label_one_hot.shape}")
            batch_size = data.size(0)
            if args.manuall:
                path_out = torch.zeros(batch_size, device=device)

            if args.manuall:
                while number_patches < args.num_patches:
                    print(f"iteration of patches: {number_patches}")
                    # Crop patches for the entire batch
                    bias_x = np.random.randint(pred.shape[2] - args.patch_size)
                    bias_y = np.random.randint(pred.shape[3] - args.patch_size)
                    patch_pred = pred[:, :, bias_x:bias_x + args.patch_size, bias_y:bias_y + args.patch_size]  # Shape: (batch_size, C, H, W)
                    patch_label = label_one_hot[:, :, bias_x:bias_x + args.patch_size, bias_y:bias_y + args.patch_size]  # Shape: (batch_size, H, W)

                    cross_entropy_loss = CrossEntropyLoss2d()
                    patch_label_indices = torch.argmax(patch_label, dim=1)
                    #target = target.type(torch.LongTensor).to(device)
                    loss_eval= cross_entropy_loss.forward(patch_pred, patch_label_indices)
                    #print("check if loss value can be used for threshold:", loss_eval)
                    #print("patch_pred", patch_pred.shape)
                    #print("patch_label", patch_label.shape)
                    # this should be before having output or after ?
                    #loss, _ = MAE(patch_pred, patch_label,args.n_classes)
                    #total_loss.append(loss.item())
                    #if loss_eval>0.1:
                    if args.attacktype == 1:
                        input_patch=patch_pred
                    else:
                        if args.data_representation == 'loss':
                            input_patch = SLM(patch_pred, patch_label)
                        elif args.data_representation == 'concate':
                            input_patch = torch.cat((patch_pred, patch_label), dim=1)  # Shape: (batch_size, 2C, H, W)

                    output = model(input_patch)# size = batch_szie
                    #print(f"output: {output.shape}")
                    path_out += output.squeeze(-1)   # aggregate the information of the per-patch attack on an image-level
                    number_patches+=1

            else:
                batch_size, _, _, _ = pred.shape
                label_one_hot = label_one_hot.float()
                patch_pred = torch.nn.functional.unfold(pred, kernel_size=args.patch_size, stride=args.patch_size)
                patch_label = torch.nn.functional.unfold(label_one_hot, kernel_size=args.patch_size,
                                                         stride=args.patch_size)
                num_patches = patch_pred.shape[-1]  # Total patches per image
                #print(f"num_patches per image:{num_patches}")
                #print(f"Unfolded patches shape: {patch_pred.shape}")  # [B, C * patch_size^2, num_patches]


                patch_pred = patch_pred.view(batch_size, args.n_classes, args.patch_size, args.patch_size,
                                             num_patches).permute(0, 4, 1, 2, 3)
                patch_label = patch_label.view(batch_size, args.n_classes, args.patch_size, args.patch_size,
                                               num_patches).permute(0, 4, 1, 2, 3)
                patch_pred = patch_pred.reshape(-1, args.n_classes, args.patch_size, args.patch_size)
                #print(f"Reshaped patches shape: {patch_pred.shape}")  # [B * num_patches, C, patch_size, patch_size]
                patch_label = patch_label.reshape(-1, args.n_classes, args.patch_size, args.patch_size)
                #print(f"Patch prediction shape: {patch_pred.shape}")
                #print(f"Patch label shape: {patch_label.shape}")

                if args.attacktype == 1:
                    input_patch = patch_pred

                elif args.attacktype == 2:
                    if args.data_representation == 'loss':
                        input_patch = SLM(patch_pred, patch_label)  # shape: (batch_size, C, H, W)
                        # print(input_patch.shape)
                    elif args.data_representation == 'concate':
                        input_patch = torch.cat((patch_pred, patch_label), dim=1)
                        #print(f"inpute_patch:{input_patch.shape}")  # Shape: (batch_size, 2C, H, W)
                        #print(f"patch_pred:{patch_pred.shape}")
                        #print(f"patch_label:{patch_label.shape}")

                output = model(input_patch)
                output = output.view(batch_size, num_patches, -1)  # Shape: [batch_size, num_patches, 1]
                #print(f"output: {output.shape}")
                path_out =output.squeeze(-1)
                path_out = path_out.mean(dim=1)  # Shape: [batch_size]

            #print(f"path_out: {path_out.shape}")
            if args.manuall:
                path_out = path_out / args.num_patches  # (Based on Algorithm 2 of "Segmentations-Leak: Membership Inference Attacks and Defenses in Semantic Image Segmentation")
                pred_round = path_out.float().round().view(data.shape[0]).detach().cpu().numpy()
                true_target = targets.float().view(data.shape[0]).detach().cpu().numpy()
                pred_roc = path_out.view(data.shape[0]).detach().cpu().numpy()  # Use raw predictions for ROC (no round)
            else:
                pred_round = path_out.float().round().view(data.shape[0]).detach().cpu().numpy()
                true_target = targets.float().view(data.shape[0]).detach().cpu().numpy()
                pred_roc = path_out.view(data.shape[0]).detach().cpu().numpy()  # Use raw predictions for ROC (no round)

            pred_labels = np.concatenate((pred_labels, pred_round))
            true_labels = np.concatenate((true_labels, true_target))
            #print(f"pred_labels: {pred_labels.shape}")
            #print(f"true_labels: {true_labels.shape}")
            pred_labels_roc = np.concatenate((pred_labels_roc, pred_roc))

        #print(f"total_loss:{total_loss}")
        #print(f"pred_labels outside: {pred_labels.shape}")
        #print(f"true_labels outside : {true_labels.shape}")
        a_score = round(accuracy_score(true_labels, pred_labels), 4)
        f_score = round(f1_score(true_labels, pred_labels), 4)
        if accuracy_only:
            print('Validation accuracy:', accuracy_score)
        else:
            print(
                'Validation accuracy:', a_score,
                ', F1-score:', f_score,
            )

            tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()

            print('TN: {}, FP: {}, FN: {}, TP: {}'.format(tn, fp, fn, tp))

        if roc:
            save_results_to_csv(args, true_labels, pred_labels_roc, tn, fp, fn, tp)
            # plot
            print(f"true labels: {true_labels}")
            print(f"predicted labels: {pred_labels_roc}")
            print(f"final predictions:{pred_labels}")

            fpr, tpr, thresholds = roc_curve(true_labels, pred_labels_roc)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title(f"{args.attacktype} {ATTACK_BATCH_SIZE} {args.main_dir}")
            plt.legend(loc="lower right")
            plt.show()

    return a_score, f_score



def test_attack_model_multiple(args, attack_models, dataloaders, num_classes, shadow_model=None, victim_model=None, accuracy_only=False,
                      input_channels=1,roc=False):

    true_targets = []
    all_pred_labels = []
    pred_label_roc=[]
    print(f"number of attack models:{len(attack_models)}")
    for i in range(len(attack_models)):
        model=attack_models[i]
        model.eval()
        model_pred = []
        for data, labels, targets in dataloaders[i]:
            #print(f"data loader {i} size:{data.shape}")
            data, labels, targets = data.to(device), labels.to(device), targets.to(device)
            with torch.no_grad():
                if shadow_model == None:
                    """pred = victim_model(data)
                    pred_single_channel = pred[:, i, :, :]  # Shape: [batch_size, height, width]
                    zeros_channel = torch.zeros_like(pred_single_channel)  # Shape: [batch_size, height, width]
                    pred = torch.stack([pred_single_channel, zeros_channel], dim=1)"""
                    model_prediction=victim_model(data)
                    #print(f"model prediction shape is {model_prediction.shape}")
                    pred_single_channel= model_prediction[:,i,:,:] # Shape: [batch_size, height, width]
                    if len(pred_single_channel.shape)<4:
                        pred_single_channel= pred_single_channel.unsqueeze(1)
                    pred= pred_single_channel

                else:
                    pred = shadow_model(data)

                #print(f"prediction before process:{pred}")
                # argmax defense
                if args.defensetype == 2:
                    pred = torch.argmax(pred, 1, keepdim=True)

            # Type-II attack
            if args.attacktype == 2:
                # non-argmax defense
                #if args.defensetype != 2:
                    #labels = nn.functional.one_hot(labels, num_classes=num_classes)
                #print(f"Original labels shape: {labels.shape}")
                cat = labels.view(data.shape[0], pred.shape[1], data.shape[2], data.shape[3]).float()
                #print(f"labels shape: {cat.shape}")
                #print(f"Pred shape: {pred.shape}")
                s_output = torch.concat((pred, cat), dim=1)


            else:

                s_output = pred  # just output of the victim model
            """foreground_pred = pred[:, 0, :, :]
            background_pred = pred[:, 1, :, :]

            # For model output
            model_foreground = labels[:, 0, :, :]
            model_background = labels[:, 1, :, :]

            # Check if the channel order is consistent
            assert torch.allclose(foreground_pred, model_foreground, atol=1e-6)
            assert torch.allclose(background_pred, model_background, atol=1e-6)"""

            output = model(s_output.float())
            #print(f"output shape: {output.shape}")
            #print(f"output : {output}")
            pred_l = output.float().round().view(data.shape[0]).detach().cpu().numpy()
            model_pred.extend(pred_l)
            #model_pred.append(pred_l)
            # save target just for the first model (All other model and dataloader has same target)
            if i==0:
             #true_targets.append(targets.float().view(data.shape[0]).detach().cpu().numpy())
             true_targets.extend(targets.float().view(data.shape[0]).detach().cpu().numpy())

        """if i==0:
            total_true_targets= np.concatenate(true_targets) # flatting the list"""

        #model_preds = [item for sublist in model_pred for item in sublist] # flat all predictions for all samples by current model
        all_pred_labels.append(model_pred)
    #print(f"all_pred_labels:{all_pred_labels}")
    total_true_targets = np.array(true_targets)
    stacked_batch_pred=np.stack(all_pred_labels, axis=1) # 2D array : rows: samples , columns: attack models
    #print(f"stacked_batch_pred:{stacked_batch_pred}")
    majority_vote_preds=np.apply_along_axis(lambda x:np.argmax(np.bincount(x.astype(int))), axis=1, arr=stacked_batch_pred)

    """if args.attacktype == 1:
        print(f"majority_vote_preds:{majority_vote_preds}")
        print(f"total_true_targets:{total_true_targets}")
        print(f"total_true_targets: {total_true_targets.shape}")
        print(f"majority_vote_preds: {majority_vote_preds.shape}")"""

    a_score = round(accuracy_score(total_true_targets, majority_vote_preds), 4)
    f_score = round(f1_score(total_true_targets, majority_vote_preds), 4)

    if accuracy_only:
        print('Validation accuracy:', accuracy_score)
    else:
        print(
            'Validation accuracy:', a_score,
            ', F1-score:', f_score,
        )

        tn, fp, fn, tp = confusion_matrix(total_true_targets, majority_vote_preds).ravel()
        print('TN: {}, FP: {}, FN: {}, TP: {}'.format(tn, fp, fn, tp))



    if roc:
        save_results_to_csv(args, total_true_targets, majority_vote_preds,tn,fp,fn,tp)

        fpr,tpr,thresholds = roc_curve(total_true_targets, majority_vote_preds)
        roc_auc = auc(fpr,tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    return a_score, f_score



def having_multi_shadows(args, attack_models, dataloader):
    """
    inputs:
        trained attack models (k numbers, for example)
        test dataloader : same for all models (train was also the same)

    Returns:
        final prediction
    """
    # iterate over attack models
    # each make a fina prediction for the samples (same as normal test) by test_attack_model
    # keep the prediction of each model
    # use majority vote to output the final prediction


