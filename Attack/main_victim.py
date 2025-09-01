import argparse
from data import split_and_copy_victim_only,split_and_copy
from utilis_train import get_victim, get_shadow,get_attack, get_shadow_data_generation,get_shadow_multiple, get_attack_multiple, get_attack_patchify
import os
import torch
import numpy as np
from global_attack import membership_loss_victim_only,membership_loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from losses import CombinedLoss, FocalFrequencyLoss

def main(args):

        main_dir=args.main_dir
        print(args.morphology)
        #data= split_and_copy(os.path.join(main_dir, 'train'), os.path.join(main_dir, 'val'))
        data = split_and_copy_victim_only(os.path.join(main_dir, 'train'), os.path.join(main_dir, 'val'))
        # train victim and shadow models
        print(" ** TRAINING VICTIM MODEL **")
        victim_model, _ = get_victim(data, args)


        print(" ** MEMBERSHIP INFERENCE ATTACK **")
        print("****membership_loss****")
        #membership_loss(args,data,data,victim_model)
        membership_loss_victim_only(args, data, victim_model)


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
    parser.add_argument('--DPSGD', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--DPSGD_for_Saving_and_synthetic', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--patchify', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--multiple_shadow', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--test', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--synthetic_data', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--model_should_be_saved', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--model_should_be_load', type=lambda x: x.lower() == 'true', default=True)


    parser.add_argument('--mixup', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('-data_representation', '--data_representation', type=str, default='concate',
                        help="data representation for attacks. choose 'loss' or 'concate'.")
    parser.add_argument('-num_patches', '--num_patches', default=10, type=int,
                        help='number of patches to use.')
    parser.add_argument('--patch_size', default=90, type=int, help='patch size.')
    parser.add_argument('--epsilon', default=200, type=float,help='epsilon value.')
    parser.add_argument('--manuall', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--stride', default=3, type=int, help='stride')
    parser.add_argument('--morphology', type=lambda x: x.lower() == 'true',  default=True,help="morphology")
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

