"""import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV
try:
    df = pd.read_csv('csv_results_chinchilla/DukeData_shadow_losses.csv')
    print(df.shape)
except FileNotFoundError:
    print("Error: CSV file 'UMNData_shadow_losses.csv' not found.")
    exit(1)

# Parameter grid
morphologies = [True, False]
dpsgd_options = [True, False]
operations = ['both', 'close', 'open']
epsilons = [0.1, 0.5, 8, 200]
main_dir = 'DukeData'

# Collect all shadow thresholds
all_losses = []

for morphology in morphologies:
    for dpsgd in dpsgd_options:
        current_ops = operations if morphology else [None]
        current_epsilons = epsilons if dpsgd else [None]

        for Operation in current_ops:
            for epsilon in current_epsilons:
                morph_label = "morph" if morphology else "non-morph"

                # Filtering
                if morphology:
                    if dpsgd:
                        filtered = df[(df['Morphology'] == morphology) &
                                      (df['epsilon'] == epsilon) &
                                      (df['Operation'] == Operation)]
                    else:
                        filtered = df[(df['Morphology'] == morphology) &
                                      (df['Operation'] == Operation) &
                                      (pd.isna(df['epsilon']))]
                else:
                    if dpsgd:
                        filtered = df[(df['Morphology'] == morphology) &
                                      (df['epsilon'] == epsilon)]
                    else:
                        filtered = df[(df['Morphology'] == morphology) &
                                      (pd.isna(df['epsilon']))]

                if filtered.empty:
                    print(f"Skipping: morph={morphology}, op={Operation}, eps={epsilon}, dpsgd={dpsgd} — no data")
                    continue

                for _, row in filtered.iterrows():
                    config_label = f"{morph_label}, DP={dpsgd}"
                    operation_label = Operation if Operation else "None"

                    all_losses.append({
                        "Loss": row['shadow_threshold'],
                        "Epsilon": epsilon if epsilon else "None",
                        "Operation": operation_label,
                        "Setting": config_label
                    })

# Convert to DataFrame for seaborn
loss_df = pd.DataFrame(all_losses)

# Create faceted boxplot
sns.set_context("notebook", font_scale=1.5)
g = sns.FacetGrid(loss_df, col="Operation", col_wrap=4, height=4, aspect=1.5)
g.map_dataframe(sns.boxplot, x="Epsilon", y="Loss")
g.set_titles(col_template="{col_name}")
g.set_axis_labels("Epsilon", "Shadow Threshold")
g.fig.suptitle("Shadow Threshold Distribution Across Settings", y=1.05)

# Save the plot
output_path  = "/home/parsar0000/oct_git/main_code/Attack/loss_distr_results_chinchilla/Duke_faceted_boxplot_shadow_thresholds.pdf"
g.savefig(output_path)
plt.close()
print(f"Saved faceted boxplot to {output_path}")

"""

"""import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

def parse_loss_list(s):
    return [float(x) for x in s.strip('[]').split()]



df = pd.read_csv('csv_results_chinchilla/UMNData_global_attack_results.csv')




morphologies = [True, False]
operations = ['both', 'close', 'open']
epsilons = [0.1, 0.5, 8, 200, None]  # None for DPSGD=False


roc_data = {}
for morphology in morphologies:
    current_ops = operations if morphology else [None]
    for operation in current_ops:
        roc_data_key = f"Morph={'morph' if morphology else 'non-morph'}, Op={operation if operation else 'None'}"
        roc_data[roc_data_key] = []

        for epsilon in epsilons:
            # Filtering
            if morphology:
                if epsilon is None:
                    filtered = df[(df['Morphology'] == morphology) &
                                  (df['Operation'] == operation) &
                                  (pd.isna(df['epsilon']))]
                else:
                    filtered = df[(df['Morphology'] == morphology) &
                                  (df['Operation'] == operation) &
                                  (df['epsilon'] == epsilon)]
            else:
                if epsilon is None:
                    filtered = df[(df['Morphology'] == morphology) &
                                  (pd.isna(df['epsilon']))]
                else:
                    filtered = df[(df['Morphology'] == morphology) &
                                  (df['epsilon'] == epsilon)]

            # Clear for current ε


            if not filtered.empty:
                true_labels = []
                pred_labels = []
                for _, row in filtered.iterrows():
                    true = parse_loss_list(row['true_label'])
                    pred = parse_loss_list(row['predicted_label'])
                    true_labels.extend(true)
                    pred_labels.extend(pred)

                if len(np.unique(true_labels)) > 1:  # Requires both classes
                    fpr, tpr, _ = roc_curve(true_labels, pred_labels)
                    auc = roc_auc_score(true_labels, pred_labels)
                    roc_data[roc_data_key].append({
                        'epsilon': str(epsilon) if epsilon is not None else 'None',
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'auc': auc
                    })
                else:
                    print(f"Skipping {roc_data_key}, ε={epsilon}: Only one class present")


# Choose relevant keys for 4 plots
plot_keys = [
    "Morph=morph, Op=both",
    "Morph=morph, Op=close",
    "Morph=morph, Op=open",
    "Morph=non-morph, Op=None"
]

# Create subplot grid
fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)

for ax, key in zip(axes, plot_keys):
    if key not in roc_data or not roc_data[key]:
        ax.set_title(f"{key}\n(no data)")
        ax.axis("off")
        continue

    for curve in roc_data[key]:
        fpr = curve['fpr']
        tpr = curve['tpr']
        auc = curve['auc']
        epsilon = curve['epsilon']
        label = f"ε={epsilon} (AUC={auc:.2f})"
        ax.plot(fpr, tpr, label=label)

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_title(key.replace("Morph=", "").replace(", Op=", "\nOp: "))
    ax.set_xlabel("False Positive Rate")
    if ax == axes[0]:
        ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=10)
    ax.grid(True)

plt.tight_layout()
plt.suptitle("ROC Curves for Morph and Non-Morph Settings", fontsize=16, y=1.05)

# Save or show the plot
output_path = "/home/parsar0000/oct_git/main_code/Attack/loss_distr_results_chinchilla/UMNData_roc_faceted_four_settings.pdf"
plt.savefig(output_path, bbox_inches="tight")
plt.show()
print(f"Saved faceted ROC plot to {output_path}")
"""

"""import pandas as pd
import matplotlib.pyplot as plt

dataset="Duke"
df=pd.read_csv(f"csv_results_chinchilla/{dataset}_shadow.csv")

df['epsilon_str'] = df['epsilon'].apply(lambda x: 'None' if pd.isna(x) else str(x))


op_colors = {
    'both': 'tab:blue',
    'open': 'tab:green',
    'close': 'tab:red',
    'None': 'tab:purple'
}


morph_styles = {
    True: 'solid',
    False: 'dashed'
}


plt.figure(figsize=(10, 6))
for (morph, op), data in df.groupby(['Morphology', 'Operation']):
    color = op_colors.get(op, 'gray')
    style = morph_styles[morph]
    label = f"Op: {op}, {'Morph' if morph else 'Non-Morph'}"

    data = data.sort_values('epsilon_str')
    plt.plot(data['epsilon_str'], data['shadow_threshold'],
             marker='o', linestyle=style, color=color, label=label)

#plt.title("Shadow Threshold vs Epsilon\n(Operations in Color, Morphology in Line Style)", fontsize=14)
plt.xlabel("Epsilon")
plt.ylabel("Shadow Threshold")
plt.legend(fontsize=10)
plt.grid(False)
plt.tight_layout()
#plt.show()

output_path = f"/home/parsar0000/oct_git/main_code/Attack/loss_distr_results_chinchilla/{dataset}_shadow.pdf"
plt.savefig(output_path)
plt.close()
print(f"Saved faceted boxplot to {output_path}")


"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


dataset= "UMNData"
dataset_sum="UMN"
main_df = pd.read_csv(f'csv_results_chinchilla/{dataset}_loss_values.csv')
shadow_df = pd.read_csv(f'csv_results_chinchilla/{dataset_sum}_shadow.csv')


def parse_array_string(s):
    if pd.isna(s) or s.strip() == '':
        return np.array([])
    try:
        # Remove brackets and split by whitespace, convert to float
        return np.array([float(x) for x in s.strip('[]').split() if x])
    except Exception as e:
        print(f"Error parsing string: {s[:50]}... Error: {e}")
        return np.array([])


shadow_df['epsilon'] = shadow_df['epsilon'].fillna('None').astype(str)
shadow_df['Operation'] = shadow_df['Operation'].fillna('None')
shadow_df['Morphology'] = shadow_df['Morphology'].astype(str)


main_df['epsilon'] = main_df['epsilon'].fillna('None').astype(str)
main_df['Operation'] = main_df['Operation'].fillna('None')
main_df['Morphology'] = main_df['Morphology'].astype(str)


main_df['loss_mem'] = main_df['loss_mem'].apply(parse_array_string)
main_df['loss_non_mem'] = main_df['loss_non_mem'].apply(parse_array_string)
main_df['true_label'] = main_df['true_label'].apply(parse_array_string)


def compute_predictions(row, shadow_df):

    shadow_row = shadow_df[
        (shadow_df['Morphology'] == row['Morphology']) &
        (shadow_df['Operation'] == row['Operation']) &
        (shadow_df['epsilon'] == row['epsilon'])
    ]

    if shadow_row.empty:
        print(f"No matching shadow row for Morphology={row['Morphology']}, Operation={row['Operation']}, epsilon={row['epsilon']}")
        return np.zeros_like(row['true_label'])

    threshold = shadow_row['shadow_threshold'].iloc[0]

    loss_mem = row['loss_mem']
    loss_non_mem = row['loss_non_mem']
    true_label = row['true_label']


    lengths = {
        'loss_mem': len(loss_mem),
        'loss_non_mem': len(loss_non_mem),
        'true_label': len(true_label)
    }


    predictions = np.zeros_like(true_label)
    j=0
    for i, label in enumerate(loss_mem):
            loss1 = loss_mem[i]
            predictions[i] = 1 if not np.isnan(loss1) and loss1 <= threshold else 0
            j+=1

    for i, label in enumerate(loss_non_mem):

            loss2 = loss_non_mem[i]
            predictions[j+i] = 1 if not np.isnan(loss2) and loss2 <= threshold else 0

    return predictions

# Apply predictions to each row
main_df['prediction'] = main_df.apply(lambda row: compute_predictions(row, shadow_df), axis=1)

# Convert prediction arrays to string format for CSV output
main_df['prediction'] = main_df['prediction'].apply(lambda x: '[' + ' '.join(map(str, x)) + ']')

# Save the modified dataframe to a new CSV
main_df.to_csv(f'csv_results_chinchilla/{dataset}_loss_values_with_predictions.csv', index=False)

print(f"Modified CSV saved as f'{dataset}_loss_values_with_predictions.csv'")



df = pd.read_csv(f'csv_results_chinchilla/{dataset}_loss_values_with_predictions.csv')




df['true_label'] = df['true_label'].apply(parse_array_string)
df['prediction'] = df['prediction'].apply(parse_array_string)



def compute_metrics(row):
    true_labels = row['true_label']
    predictions = row['prediction']


    if len(true_labels) != len(predictions) or len(true_labels) == 0:
        print(
            f"Length mismatch or empty arrays in row: Morphology={row['Morphology']}, Operation={row['Operation']}, epsilon={row['epsilon']}")
        print(f"true_label length: {len(true_labels)}, prediction length: {len(predictions)}")
        return pd.Series({'f1_score': np.nan, 'accuracy': np.nan})


    f1 = f1_score(true_labels, predictions, zero_division=0)
    accuracy = accuracy_score(true_labels, predictions)

    return pd.Series({'f1_score': f1, 'accuracy': accuracy})



metrics = df.apply(compute_metrics, axis=1)
df['f1_score'] = metrics['f1_score']
df['accuracy'] = metrics['accuracy']


df.to_csv(f'csv_results_chinchilla/f1_accuracy_{dataset_sum}.csv', index=False)

print("Modified CSV with f1_score and accuracy columns saved as 'f1_accuracy'")


mean_df = (df[['Operation', 'epsilon', 'f1_score', 'accuracy']])
mean_df.columns = ['morphology', 'epsilon', 'F1', 'accuracy']
mean_df = mean_df.groupby(['epsilon', 'morphology']).mean()
print(mean_df.to_latex(index=True, multirow=True, float_format="%.2f", label='tab:loss-attack', caption='Comparison of global loss-based attack performances'))

