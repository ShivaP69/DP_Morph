import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


try:
    df = pd.read_csv('csv_results_chinchilla/DukeData_loss_values.csv')
except FileNotFoundError:
    print("Error: CSV file 'DukeData_loss_values.csv' not found.")
    exit(1)

# Helper to parse malformed loss list
def parse_loss_list(s):
    return [float(x) for x in s.strip('[]').split()]

# Parameter grid
morphologies = [True, False]
dpsgd_options = [True, False]
operations = ['open', 'close']
epsilons = [0.1, 0.5, 8, 200]

main_dir = 'DukeData'

for morphology in morphologies:
    for dpsgd in dpsgd_options:
        # Only loop over operations if morphology is True, else None
        if morphology:
            current_ops = operations
        else:
            current_ops = [None]

        current_epsilons = epsilons if dpsgd else [None]

        for Operation in current_ops:
            for epsilon in current_epsilons:
                morph = "morph" if morphology else "non-morph"

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

                all_mem_runs = []
                all_non_mem_runs = []

                for _, row in filtered.iterrows():
                    mem_list = parse_loss_list(row['loss_mem'])
                    non_mem_list = parse_loss_list(row['loss_non_mem'])
                    all_mem_runs.append(mem_list)
                    all_non_mem_runs.append(non_mem_list)

                mem_array = np.array(all_mem_runs)
                non_mem_array = np.array(all_non_mem_runs)

                mem_avg = mem_array.mean(axis=0)
                non_mem_avg = non_mem_array.mean(axis=0)

                # Prepare strings for plotting and saving
                op_str = f"_operation:{Operation}" if Operation is not None else ""
                eps_str = f"_epsilon:{epsilon}" if epsilon is not None else ""

                plt.figure(figsize=(10, 6))
                sns.kdeplot(x=mem_avg, color='blue', label='Members', fill=True, alpha=0.3)
                sns.kdeplot(x=non_mem_avg, color='red', label='Non-Members', fill=True, alpha=0.3)
                plt.title(f'Avg Loss KDE: dataset={main_dir}, Morph={morphology}, DP-SGD={dpsgd}{op_str}{eps_str}')
                plt.xlabel('Loss')
                plt.ylabel('Density')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                plt.tight_layout()

                # Save plot
                output_dir = f'/home/parsar0000/oct_git/main_code/Attack/loss_distr_results_chinchilla/{morph}/Duke'
                os.makedirs(output_dir, exist_ok=True)
                file_name = f'morphology:{morphology}{op_str}{eps_str}_DPSGD={dpsgd}.pdf'
                plt.savefig(os.path.join(output_dir, file_name))
                plt.close()
                print(f"Saved: {file_name}")
