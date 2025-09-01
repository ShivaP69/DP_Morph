import argparse
from data import split_and_copy
from data import DatasetOct
from utilis_train import get_victim, get_shadow,get_attack, get_shadow_data_generation,get_shadow_multiple, get_attack_multiple, get_attack_patchify
import os
from attack_train import test_attack_model_multiple
from global_attack import global_attack
from data import split_and_copy_synthetic,shadow_split_and_copy_synthetic,combine_attack_data
import torch
import numpy as np
import pandas as pd
from global_attack import membership_loss
from torch.utils.data import DataLoader
from train import apply_kornia_morphology_multiclass
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from losses import CombinedLoss, FocalFrequencyLoss

def save_shaodw_thr(args,shadow_threshold):
    losses_df = pd.DataFrame({
        "Morphology": [args.morphology],
        "Operation": [args.operation if args.morphology else None],
        "epsilon": [args.epsilon if args.DPSGD_for_Saving_and_synthetic else None],
        "shadow_threshold": [shadow_threshold],
    })
    output_folder = "csv_results_chinchilla"
    if args.main_dir=="DukeData":
        dataset="Duke"
    else:
        dataset="UMN"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    losses_csv_path = os.path.join(output_folder, f'{dataset}_shadow.csv')
    if os.path.exists(losses_csv_path):
        losses_df.to_csv(losses_csv_path, mode='a', header=False, index=False)
    else:
        losses_df.to_csv(losses_csv_path, index=False)
    print(f"shadow_threshold: {shadow_threshold}")


def main(args):
    print(args.epsilon)

    if args.main_dir == "UMNData":
        main_path = "../UMNData/"
    else:
        main_path = "../DukeData/"

    if args.synthetic_data:
        data= split_and_copy_synthetic(os.path.join(main_path, 'train'), os.path.join(main_path, 'val'),DME7=True)
        print(""""training victim model when synthetic data is used""")
        victim_model,_=get_victim(data,args)
        # we should use victim model to label external dataset
        print("**** Synthetic data generation ****")
        # one time is enough!

        if args.DPSGD_for_Saving_and_synthetic:
            if args.main_dir == "DukeData":
                if args.morphology==True:
                       file_name_shadow='morphology_shadow_dataset_DPSGD'
                else:
                        file_name_shadow='shadow_dataset_DPSGD'

                if os.path.isdir(file_name_shadow):
                    print("**** DPSGD version of Data generation has been done previously for Duke Dataset***")

                else:
                    print("****Starting Synthetic data generation based on DPSGD data")
                    get_shadow_data_generation(data, args, victim_model, file_name_shadow,dataset_synthetic="DME")
                    print("***Data generation has been finished***")
                    print("****training shadow model***")

            elif args.main_dir=="UMNData":
                if args.morphology==True:
                        file_name_shadow='morphology_shadow_dataset_DPSGD_UMN'
                else:
                        file_name_shadow='shadow_dataset_DPSGD_UMN'
                if os.path.isdir(file_name_shadow):
                    print("**** DPSGD version of Data generation has been done previously for UMN dataset***")

                else:
                    print("****Starting Synthetic data generation based on DPSGD data")
                    get_shadow_data_generation(data, args, victim_model,file_name_shadow, dataset_synthetic="DME")
                    print("***Data generation has been finished***")
                    print("****training shadow model***")

        else:
            if args.main_dir == "DukeData":

                if args.morphology == True:
                    file_name_shadow = 'morphology_shadow_dataset'
                else:
                    file_name_shadow = 'shadow_dataset'
                if os.path.isdir(file_name_shadow):
                    print("**** Data generation has been done previously for Duke dataset***")
                else:
                     print("****Starting....")
                     get_shadow_data_generation(data, args, victim_model,file_name_shadow, dataset_synthetic="DME")
                     print("***Data generation has been finished***")
                     print("****training shadow model***")

            elif args.main_dir == "UMNData":
                if args.morphology == True:
                    file_name_shadow = 'morphology_shadow_dataset_UMN'
                else:
                        file_name_shadow = 'shadow_dataset_UMN'
                if os.path.isdir(file_name_shadow):
                    print("**** Data generation has been done previously for UMN dataset***")
                else:
                    print("****Starting....")
                    get_shadow_data_generation(data, args, victim_model,file_name_shadow, dataset_synthetic="DME")
                    print("***Data generation has been finished***")
                    print("****training shadow model***")


        if args.DPSGD_for_Saving_and_synthetic:
            if args.main_dir == "DukeData":
                 shadow_data=shadow_split_and_copy_synthetic(file_name_shadow) # file_name_shadow: synthetic: the main purpose of "shadow_split_and_copy_synthetic" is putting file_name_shadow in a format which is ok for training shadow model
            else:
                shadow_data=shadow_split_and_copy_synthetic(file_name_shadow)
        else:
            if args.main_dir == "DukeData":
                shadow_data = shadow_split_and_copy_synthetic(file_name_shadow)
            else:
                shadow_data = shadow_split_and_copy_synthetic(file_name_shadow)


        if args.multiple_shadow:
            print("****multiple shadow model is starting****")
            shadow_models, shadow_thresholds = get_shadow_multiple(shadow_data, args)
            combined_attack_data = combine_attack_data(data, shadow_data)
            # ATTACK MODEL
            print(" ** MEMBERSHIP INFERENCE ATTACK **")
            #attack_models,dataloaders= get_attack_multiple(combined_attack_data,args,victim_model,shadow_models)
            print(' ** Type-I **')
            args.attacktype = 1
            attack_models, dataloaders = get_attack_multiple(combined_attack_data, args, victim_model, shadow_models)
            test_attack_model_multiple(args, attack_models, dataloaders, num_classes=1, victim_model=victim_model,roc=True)
            print(' ** Type-II **')
            args.attacktype = 2
            attack_models, dataloaders = get_attack_multiple(combined_attack_data, args, victim_model, shadow_models)
            test_attack_model_multiple(args, attack_models, dataloaders, num_classes=1, victim_model=victim_model,roc=True)
            print(' ** Global loss **')
            average_thereshold=np.mean(shadow_thresholds)
            global_attack(args,shadow_data,combined_attack_data, victim_model,average_thereshold )

        else:
            if not args.patchify:
                shadow_model, shadow_threshold =get_shadow(shadow_data,args)
                print("**** Shadow model has been trained***")

                print(" ** MEMBERSHIP INFERENCE ATTACK **")

                combined_attack_data=combine_attack_data(data,shadow_data)# check here !

                print(
                    f"Victim Attack Paths - Ones: {np.sum(combined_attack_data.victim_attack_paths['member'] == 1)}, Zeros: {np.sum(combined_attack_data.victim_attack_paths['member'] == 0)}")
                print(
                    f"Shadow Attack Paths - Ones: {np.sum(combined_attack_data.shadow_attack_paths['member'] == 1)}, Zeros: {np.sum(combined_attack_data.shadow_attack_paths['member'] == 0)}")

                print("****membership_loss****")
                print(membership_loss(args, data, shadow_data, victim_model))

                print(' ** Global loss **')
                global_attack(args, shadow_data, data, victim_model, shadow_threshold)

                print(' ** Type-I **')
                args.attacktype = 1
                _ = get_attack(combined_attack_data, args, victim_model, shadow_model)

                print(' ** Type-II **')
                args.attacktype = 2
                _ = get_attack(combined_attack_data, args, victim_model, shadow_model)



            elif args.patchify:
                shadow_model, shadow_threshold = get_shadow(shadow_data, args)
                print("**** Shadow model has been trained***")

                print(" ** MEMBERSHIP INFERENCE ATTACK **")
                combined_attack_data = combine_attack_data(data, shadow_data)

                print(' ** Type-II **')
                args.attacktype = 2
                get_attack_patchify(combined_attack_data, args, victim_model, shadow_model)

                print(' ** Type-I **')
                args.attacktype = 1
                get_attack_patchify(combined_attack_data, args, victim_model, shadow_model)

    else:
        if args.multiple_shadow:
            print("****multiple shadow model is starting****")

            data = split_and_copy(os.path.join(main_path, 'train'), os.path.join(main_path, 'val'))
            shadow_models, shadow_thresholds = get_shadow_multiple(data, args)

            victim_model, _ = get_victim(data, args)
            # ATTACK MODEL
            print(" ** MEMBERSHIP INFERENCE ATTACK **")
            # attack_models,dataloaders= get_attack_multiple(combined_attack_data,args,victim_model,shadow_models)
            print("****membership_loss****")
            print(membership_loss(args, data, data, victim_model))

            print(' ** Type-I **')
            args.attacktype = 1
            attack_models, dataloaders = get_attack_multiple(data, args, victim_model, shadow_models)
            test_attack_model_multiple(args, attack_models, dataloaders, num_classes=1, victim_model=victim_model,roc=True)

            print(' ** Type-II **')
            args.attacktype = 2
            attack_models, dataloaders = get_attack_multiple(data, args, victim_model, shadow_models)
            test_attack_model_multiple(args, attack_models, dataloaders, num_classes=1, victim_model=victim_model,roc=True)

            print(' ** Global loss **')
            average_thereshold = np.mean(shadow_thresholds)
            global_attack(data, args, victim_model, average_thereshold)

        else:
            if not args.patchify:
                data= split_and_copy(os.path.join(main_path, 'train'), os.path.join(main_path, 'val'))
                # train victim and shadow models
                print(" ** TRAINING VICTIM MODEL **")
                victim_model, _ = get_victim(data, args)
                print(" ** TRAINING SHADOW MODEL **")
                shadow_model, shadow_threshold = get_shadow(data, args)
                save_shaodw_thr(args,shadow_threshold)
                print(" ** MEMBERSHIP INFERENCE ATTACK **")
                #print("****membership_loss****")
                #print(membership_loss(args, data, data, victim_model))
                print(' ** Type-I **')

                #args.attacktype = 1
                #_ = get_attack(data, args, victim_model, shadow_model)

                #print(' ** Type-II **')
                #args.attacktype = 2
                #_ = get_attack(data, args, victim_model, shadow_model)

                print(' ** Global loss **')
                global_attack(args, data, data, victim_model, shadow_threshold)


                #print(' ** Global loss **')
                #global_attack(data, args, victim_model, shadow_threshold)
            elif args.patchify:
                data = split_and_copy(os.path.join(main_path, 'train'), os.path.join(main_path, 'val'))
                # train victim and shadow models
                print(" ** TRAINING VICTIM MODEL **")
                victim_model, _ = get_victim(data, args)
                print(" ** TRAINING SHADOW MODEL **")
                shadow_model, shadow_threshold = get_shadow(data, args)
                print(f"shadow_threshold: {shadow_threshold}")

                print(" ** MEMBERSHIP INFERENCE ATTACK **")


                print(' ** Type-II **')
                args.attacktype = 2
                get_attack_patchify(data, args, victim_model, shadow_model)
                print(' ** Type-I **')
                args.attacktype = 1
                get_attack_patchify(data, args, victim_model, shadow_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # resnets, mobilenet_v2, vgg11
    parser.add_argument("--victim", type=str, default='resnet34')
    # resnets, mobilenet_v2, vgg11
    parser.add_argument("--shadow", type=str, default='resnet34')
    # 1 -- Type-I attack,
    # 2 -- Type-II attack,
    # 3 -- Global loss-based attack
    parser.add_argument("--attacktype", type=int, default=1)
    # 1 -- no defense,
    # 2 -- argmax defense,
    # 3 -- crop training,
    # 4 -- mix-up
    parser.add_argument("--defensetype", type=int, default=0)
    # same for victim and shadow models
    parser.add_argument("--trainsize", type=int, default=600)
    parser.add_argument("--main_dir", type=str, default='DukeData',choices=['UMNData','DukeData'])
    parser.add_argument('--pure', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--batch_size', default=8 , type=int)
    parser.add_argument('--num_iterations', default=200, type=int)
    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--n_classes', default=9, type=int)
    parser.add_argument('--OUTPUT_CHANNELS', default=9, type=int)
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
    parser.add_argument('--DPSGD', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--DPSGD_for_Saving_and_synthetic', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--patchify', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--multiple_shadow', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--test', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--synthetic_data', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--model_should_be_saved', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--model_should_be_load', type=lambda x: x.lower() == 'true', default=False)


    parser.add_argument('--mixup', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('-data_representation', '--data_representation', type=str, default='concate',
                        help="data representation for attacks. choose 'loss' or 'concate'.")
    parser.add_argument('-num_patches', '--num_patches', default=10, type=int,
                        help='number of patches to use.')
    parser.add_argument('--patch_size', default=90, type=int, help='patch size.')
    parser.add_argument('--epsilon', default=0.1, type=float,help='epsilon value.')
    parser.add_argument('--manuall', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--stride', default=3, type=int, help='stride')
    parser.add_argument('--morphology', type=lambda x: x.lower() == 'true', default=False,help="morphology")
    parser.add_argument('--operation',default='open', type=str, help="close, open or both")
    parser.add_argument('--kernel_size', default=3, type=int, help="kernel size")


    args = parser.parse_args()
    torch.cuda.empty_cache()

    print("Victim encoder: {}".format(args.victim))
    print("Shadow encoder: {}".format(args.shadow))
    defenses = ['No defense', 'Argmax', 'Crop-training', 'Mix-up']
    print("Defense type: {}".format(defenses[args.defensetype]))
    print("Train size: {}".format(args.trainsize))
    print(args)
    main(args)

#RuntimeError: permute(sparse_coo): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = 3 is not equal to len(dims) = 4
# When we have this errror it means that we are using Duke data but synthetic data generation is not active