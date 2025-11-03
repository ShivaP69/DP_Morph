import torch
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from losses import CombinedLoss
from data import DatasetOct
from torch.utils.data import DataLoader
from utilis_train import FixedTargetWrapper
from train import apply_kornia_morphology_multiclass
from torch.utils.data import  Subset
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, mannwhitneyu
import os
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def global_attack(args,shadow_data,data,victim_model, threshold, output_folder='csv_results_chinchilla'):

    members = data.victim_attack_paths_member
    non_members_victim = data.victim_attack_paths_non_member
    non_members_synthetic = shadow_data.shadow_attack_paths
    members_data = DatasetOct(members, attack=True)
    non_members_victim_data = DatasetOct(non_members_victim, attack=True)
    non_members_shadow_data = DatasetOct(non_members_synthetic, attack=True)

    # Get dataset sizes
    non_member_victim_size = len(non_members_victim_data)
    non_member_shadow_size = len(non_members_shadow_data)
    #print(f"Non-member victim size: {non_member_victim_size}, Non-member synthetic size: {non_member_synthetic_size}")

    # Sample non-member datasets to have equal sizes
    non_member_target_size = min(non_member_victim_size,
                                 non_member_shadow_size)  # to have the same numbers of out of distribution and same distribution
    if non_member_victim_size > non_member_target_size:
        indices = np.random.choice(non_member_victim_size, non_member_target_size, replace=False)
        non_members_victim_data = Subset(non_members_victim_data, indices)
        #print(f"Sampled non-member victim dataset to size: {non_member_target_size}")
    if non_member_shadow_size > non_member_target_size:
        indices = np.random.choice(non_member_shadow_size, non_member_target_size, replace=False)
        non_members_shadow_data = Subset(non_members_shadow_data, indices)
        #print(f"Sampled non-member synthetic dataset to size: {non_member_target_size}")
    non_members_selected_shadow_data = FixedTargetWrapper(non_members_shadow_data,
                                                    fixed_target=0)  # will be used for the victim"""
    combined_non_member = torch.utils.data.ConcatDataset([non_members_victim_data, non_members_selected_shadow_data])
    #combined_non_member=non_members_victim_data
    member_size = len(members_data)
    non_member_size = len(combined_non_member)
    #print(f"Member dataset size: {member_size}, Non-member dataset size: {non_member_size}")

    # Sample to match the smaller dataset size
    target_size = min(member_size, non_member_size)
    if member_size > target_size:
        # Sample from member dataset
        indices = np.random.choice(member_size, target_size, replace=False)
        members_data = Subset(members_data, indices)

    elif non_member_size > target_size:
        # Sample from non-member dataset
        indices = np.random.choice(non_member_size, target_size, replace=False)
        combined_non_member = Subset(combined_non_member, indices)
        #print(f"Sampled non-member dataset to size: {target_size}")

    print(f"Member dataset size: {len(members_data)}, Non-member dataset size: {len(combined_non_member)}")
    combined_data= torch.utils.data.ConcatDataset([members_data, combined_non_member])
    print(f"Combined dataset size: {len(combined_data)}")
    attack_val_dataloader = DataLoader(combined_data, batch_size=8) # we can define any batch size

    criterion=CombinedLoss()
    pred_labels = np.array([])
    true_labels = np.array([])
    losses = np.array([])

    victim_model.eval()
    with torch.no_grad():
        for data, labels, targets in attack_val_dataloader:
            data, labels, targets = data.to(device), labels.to(device), targets.to(device)
            pred = victim_model(data)
            """if args.morphology:
                pred= apply_kornia_morphology_multiclass(pred, operation=args.operation, kernel_size=args.kernel_size)"""
            if args.defensetype == 2:
                pred= torch.round(pred)
            batch_losses = []
            batch_pred_labels = []
            for i in range(data.size(0)): # if batch size was one, this iteration was not necessary
                single_pred = pred[i:i + 1]
                single_label = labels[i:i + 1].squeeze(1)
                instant_loss = criterion(single_pred, single_label, device=device).item()
                batch_losses.append(instant_loss)
                batch_pred_labels.append(1 if instant_loss <= threshold else 0)

            batch_losses = np.array(batch_losses)
            batch_pred_labels = np.array(batch_pred_labels)
            batch_true_labels = targets.float().view(-1).detach().cpu().numpy()

            losses = np.concatenate((losses, batch_losses))
            pred_labels = np.concatenate((pred_labels, batch_pred_labels))
            true_labels = np.concatenate((true_labels, batch_true_labels))

    print(f"true labels: {len(true_labels)}")
    print(f"pred labels: {len(pred_labels)}")
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    print(
        'Validation accuracy:', round(accuracy, 4),
        ', F-score:', round(f1, 4),
    )
    print('TN: {}, FP: {}, FN: {}, TP: {}'.format(tn, fp, fn, tp))

    # Save results to CSV

    results_df = pd.DataFrame({
        "Model": args.model_name,
        "Morphology": args.morphology,
        "Operation": args.operation if args.morphology else None,
        "Kernel Size": args.kernel_size if args.morphology else None,
        "DPSGD":args.DPSGD_for_Saving_and_synthetic,
        "epsilon": args.epsilon if args.DPSGD_for_Saving_and_synthetic else None,
        'true_label': [true_labels],
        'predicted_label': [pred_labels],
        'loss': [losses],
        'accuracy': accuracy,
        'f1': f1,
    })
    csv_output=os.path.join(output_folder,f"{args.main_dir}_global_attack_results.csv")
    if os.path.exists(csv_output):
        results_df.to_csv(csv_output, mode='a', header=False, index=False)
    else:    results_df.to_csv(csv_output, index=False)

    print("Results saved to {}".format(csv_output))





def compare_loss_distributions(args,mem_loss, non_mem_loss,output_folder="csv_results_chinchilla"):
    """
    Compare member and non-member loss distributions using statistics, visualizations, and statistical tests.

    Args:
        mem_loss (list or np.ndarray): Per-sample losses for member data.
        non_mem_loss (list or np.ndarray): Per-sample losses for non-member data.

    Returns:
        dict: Dictionary containing statistical metrics and test results.
    """
    # Convert to numpy arrays
    mem_loss = np.array(mem_loss)
    non_mem_loss = np.array(non_mem_loss)

    # 1. Compute descriptive statistics
    stats = {
        'member_mean': np.mean(mem_loss),
        'member_std': np.std(mem_loss),
        'member_median': np.median(mem_loss),
        'member_min': np.min(mem_loss),
        'member_max': np.max(mem_loss),
        'non_member_mean': np.mean(non_mem_loss),
        'non_member_std': np.std(non_mem_loss),
        'non_member_median': np.median(non_mem_loss),
        'non_member_min': np.min(non_mem_loss),
        'non_member_max': np.max(non_mem_loss)
    }

    print("Member Loss Statistics:")
    print(f"Mean: {stats['member_mean']:.4f}, Std: {stats['member_std']:.4f}, Median: {stats['member_median']:.4f}")
    print(f"Min: {stats['member_min']:.4f}, Max: {stats['member_max']:.4f}")
    print("\nNon-Member Loss Statistics:")
    print(
        f"Mean: {stats['non_member_mean']:.4f}, Std: {stats['non_member_std']:.4f}, Median: {stats['non_member_median']:.4f}")
    print(f"Min: {stats['non_member_min']:.4f}, Max: {stats['non_member_max']:.4f}")

    # 2. Visualize distributions
    # Kernel Density Estimation (KDE) Plot
    if args.morphology:
        morph="morph"
    else:
        morph="non-morph"
    if args.main_dir=="DukeData":
        dataset="Duke"
        pure=args.pure
    else:
        dataset="UMN"
        pure=args.pure
    if pure:
        folder_name=f'loss_distr_results_chinchilla/{morph}/{dataset}' # for plots # when no synthetic data is used for calculating loss distribution
    else:
        folder_name = f'loss_distr_results_chinchilla/{morph}/{dataset}'  # for plots
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(mem_loss, color='blue', label='Members', fill=True, alpha=0.3)
    sns.kdeplot(non_mem_loss, color='red', label='Non-Members', fill=True, alpha=0.3)
    plt.title(f'Loss Distribution Comparison (KDE), dataset={args.main_dir},Morphology={args.morphology},operations={args.operation if args.morphology else False},DPSGD={args.DPSGD_for_Saving_and_synthetic}')
    plt.xlabel('Loss')
    plt.ylabel('Density')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    if args.DPSGD_for_Saving_and_synthetic:
        if args.morphology:
            plt.savefig(f'{folder_name}/{args.main_dir}_DPSGD_epsilon={args.epsilon}_{args.operation}_morphology_loss_distribution_kde.pdf')
        else:    plt.savefig(f'{folder_name}/{args.main_dir}_DPSGD_epsilon={args.epsilon}_loss_distribution_kde.pdf')
    else:
        if args.morphology:
            plt.savefig(f'{folder_name}/{args.main_dir}_{args.operation}_morphology_loss_distribution_kde.pdf')
        else:
            plt.savefig(f'{folder_name}/{args.main_dir}_loss_distribution_kde.pdf')
    plt.close()

    # Box Plot
    plt.figure(figsize=(8, 6))
    plt.boxplot([mem_loss, non_mem_loss], labels=['Members', 'Non-Members'])
    plt.title(f'Loss Distribution Comparison (Box Plot),dataset={args.main_dir}, Morphology={args.morphology},operations={args.operation},DPSGD={args.DPSGD_for_Saving_and_synthetic}')
    plt.ylabel('Loss')
    if args.DPSGD_for_Saving_and_synthetic:
        if args.morphology:
            plt.savefig(f'{folder_name}/{args.main_dir}_DPSGD_epsilon={args.epsilon}_{args.operation}_morphology_loss_distribution_boxplot.pdf')
        else:plt.savefig(f'{folder_name}/{args.main_dir}_DPSGD_epsilon={args.epsilon}_loss_distribution_boxplot.pdf')
    else:
        if args.morphology:
            plt.savefig(f'{folder_name}/{args.main_dir}_{args.operation}_morphology_loss_distribution_boxplot.pdf')
        else:
            plt.savefig(f'{folder_name}/{args.main_dir}_loss_distribution_boxplot.pdf')
    plt.close()

    # 3. Statistical tests (excluding NaN)
    # Kolmogorov-Smirnov Test
    mem_loss_valid = mem_loss[~np.isnan(mem_loss)]
    non_mem_loss_valid = non_mem_loss[~np.isnan(non_mem_loss)]
    ks_stat, ks_p_value = ks_2samp(mem_loss_valid, non_mem_loss_valid)
    stats['ks_stat'] = ks_stat
    stats['ks_p_value'] = ks_p_value
    print(f"\nKolmogorov-Smirnov Test: Statistic={ks_stat:.4f}, p-value={ks_p_value:.4f}")

    # Mann-Whitney U Test (testing if member losses are smaller)
    mw_stat, mw_p_value = mannwhitneyu(mem_loss, non_mem_loss, alternative='less')
    stats['mw_stat'] = mw_stat
    stats['mw_p_value'] = mw_p_value
    print(f"Mann-Whitney U Test (Members < Non-Members): Statistic={mw_stat:.4f}, p-value={mw_p_value:.4f}")

    # 4. Save statistics to CSV
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    stats_dict = stats.copy()
    stats_dict.update({"Model": args.model_name,
        "Morphology": args.morphology,
        "Operation": args.operation if args.morphology else None,
        "Kernel Size": args.kernel_size if args.morphology else None,
        "DPSGD":args.DPSGD_for_Saving_and_synthetic,
        "epsilon": args.epsilon if args.DPSGD_for_Saving_and_synthetic else None,})

    stats_df = pd.DataFrame([stats_dict])
    stats_csv_path = os.path.join(output_folder, f'{args.main_dir}_loss_distribution_stats.csv')
    if os.path.exists(stats_csv_path):
        stats_df.to_csv(stats_csv_path,mode='a',header=False,index=False)
    else: stats_df.to_csv(stats_csv_path, index=False)
    print(f"Statistics saved to '{stats_csv_path}'")

    # 5. Save loss values for ROC curve
    losses_df = pd.DataFrame({
        "Model": args.model_name,
        "Morphology": args.morphology,
        "Operation": args.operation if args.morphology else None,
        "Kernel Size": args.kernel_size if args.morphology else None,
        "DPSGD": args.DPSGD_for_Saving_and_synthetic,
        "epsilon": args.epsilon if args.DPSGD_for_Saving_and_synthetic else None,
        'loss_mem': [mem_loss],
        'loss_non_mem': [non_mem_loss],
        'true_label':[ np.concatenate([np.ones(len(mem_loss_valid)), np.zeros(len(non_mem_loss_valid))])]
    })
    losses_csv_path = os.path.join(output_folder, f'{args.main_dir}_loss_values.csv')
    if os.path.exists(losses_csv_path):
        losses_df.to_csv(losses_csv_path, mode='a',header=False, index=False)
    else: losses_df.to_csv(losses_csv_path, index=False)
    print(f"members:{mem_loss}")
    print(f"non-members:{non_mem_loss}")
    print(f"the loss is saved in: {losses_csv_path} ")

    return stats



def membership_evaluation(args,data_loader,victim_model,criterion_seg): # based on victim model

    loss_membership = []
    victim_model.eval()
    with torch.no_grad():
        for data, labels, targets in data_loader:
            data, labels, targets = data.to(device), labels.to(device), targets.to(device)
            pred = victim_model(data)
            """# in this implementation, morphology is not part of the model so we should apply it manually
            if args.morphology:
                pred = apply_kornia_morphology_multiclass(pred, operation=args.operation, kernel_size=args.kernel_size)""" #this is not necessary, we are not using Morph as a post processing approach
            for i in range(data.size(0)):
                single_pred = pred[i:i + 1]  # Process one sample
                single_label = labels[i:i + 1].squeeze(1)
                loss = criterion_seg(single_pred, single_label, device=device)
                loss_membership.append(loss.item())
    print(f"Membership loss (per-sample): {loss_membership}")

    return loss_membership


def membership_loss(args,data,shadow_data,victim_model): # in non-synthetic case shadow data=data
    # calculate loss distribution

    criterion_seg = CombinedLoss()
    members = data.victim_attack_paths_member
    non_members_victim = data.victim_attack_paths_non_member
    non_members_shadow_based = shadow_data.shadow_attack_paths_non_member # when we are not using synthetic data, both data and shadow data are the same
    members_data = DatasetOct(members, attack=True)
    non_members_victim_data = DatasetOct(non_members_victim, attack=True)
    non_members_shadow_data = DatasetOct(non_members_shadow_based, attack=True)

    # Get dataset sizes
    non_member_victim_size = len(non_members_victim_data)
    non_member_shadow_size = len(non_members_shadow_data)
    print(f"Non-member victim size: {non_member_victim_size}, Non-member shadow size: {non_member_shadow_size}")

    # Sample non-member datasets to have equal sizes
    non_member_target_size = min(non_member_victim_size, non_member_shadow_size) # to have same numbers of out of distribution and same distribution (However,this is important in case of using synthetic data)
    if non_member_victim_size > non_member_target_size:
        indices = np.random.choice(non_member_victim_size, non_member_target_size, replace=False)
        non_members_victim_data = Subset(non_members_victim_data, indices)
        print(f"Sampled non-member victim dataset to size: {non_member_target_size}")
    if non_member_shadow_size > non_member_target_size:
        indices = np.random.choice(non_member_shadow_size, non_member_target_size, replace=False)
        non_members_shadow_data = Subset(non_members_shadow_data, indices)
        print(f"Sampled non-member synthetic dataset to size: {non_member_target_size}")
    non_members_selected_shadow_data = FixedTargetWrapper(non_members_shadow_data, fixed_target=0)  # will be used for the victim so target should be 0 (non-member)
    combined_non_member = torch.utils.data.ConcatDataset([non_members_victim_data, non_members_selected_shadow_data])
    #combined_non_member=non_members_victim_data
    member_size = len(members_data)
    non_member_size = len(combined_non_member)
    print(f"Member dataset size: {member_size}, Non-member dataset size: {non_member_size}")

    # Sample to match the smaller dataset size
    target_size = min(member_size, non_member_size)
    if member_size > target_size:
        # Sample from member dataset
        indices = np.random.choice(member_size, target_size, replace=False)
        members_data = Subset(members_data, indices)

    elif non_member_size > target_size:
        # Sample from non-member dataset
        indices = np.random.choice(non_member_size, target_size, replace=False)
        combined_non_member = Subset(combined_non_member, indices)
        print(f"Sampled non-member dataset to size: {target_size}")

    data_loader_non_member = DataLoader(combined_non_member, batch_size=int(args.batch_size))
    data_loader_member = DataLoader(members_data, batch_size=int(args.batch_size))
    non_mem_loss = membership_evaluation(args, data_loader_non_member, victim_model, criterion_seg)
    mem_loss = membership_evaluation(args, data_loader_member, victim_model, criterion_seg)
    stats = compare_loss_distributions(args,mem_loss, non_mem_loss)




def membership_loss_victim_only(args,data,victim_model): # in case that we do not want to use shadow data
    # calculate loss distribution
    criterion_seg = CombinedLoss()
    members = data.victim_attack_paths_member
    non_members_victim = data.victim_attack_paths_non_member
    members_data = DatasetOct(members, attack=True)
    non_members_victim_data = DatasetOct(non_members_victim, attack=True)

    member_size = len(members_data)
    non_member_size = len(non_members_victim_data)
    print(f"Member dataset size: {member_size}, Non-member dataset size: {non_member_size}")

    # Sample to match the smaller dataset size
    target_size = min(member_size, non_member_size)
    if member_size > target_size:
        # Sample from member dataset
        indices = np.random.choice(member_size, target_size, replace=False)
        members_data = Subset(members_data, indices)
    elif non_member_size > target_size:
        # Sample from non-member dataset
        indices = np.random.choice(non_member_size, target_size, replace=False)
        combined_non_member = Subset(non_members_victim_data, indices)
        print(f"Sampled non-member dataset to size: {target_size}")

    data_loader_non_member = DataLoader(non_members_victim_data, batch_size=int(args.batch_size))
    data_loader_member = DataLoader(members_data, batch_size=int(args.batch_size))
    non_mem_loss = membership_evaluation(args, data_loader_non_member, victim_model, criterion_seg)
    mem_loss = membership_evaluation(args, data_loader_member, victim_model, criterion_seg)
    stats = compare_loss_distributions(args,mem_loss, non_mem_loss)



