import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")


# Helper to parse malformed loss lists
def parse_loss_list(s):
    return [float(x) for x in s.strip('[]').split()]


datasets = ['Duke', 'UMN']

for dataset in datasets:
    data_file = f'csv_results_chinchilla/{dataset}Data_loss_values.csv'
    try:
        df = pd.read_csv(data_file)
        print(df.shape)
    except FileNotFoundError:
        print(f"Error: CSV file {data_file} not found.")
        exit(1)

    morphologies = [True, False]
    dpsgd_options = [True, False]
    operations = ['both', 'close', 'open']
    epsilons = [0.1, 0.5, 8, 200]

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
                        mem_list = parse_loss_list(row['loss_mem'])
                        non_mem_list = parse_loss_list(row['loss_non_mem'])

                        config_label = f"{morph_label}, DP={dpsgd}"
                        operation_label = Operation if Operation else "none"

                        for val in mem_list:
                            all_losses.append({
                                "Loss": val,
                                "Group": "Member",
                                "Epsilon": epsilon if epsilon else "non-private",
                                "Operation": operation_label,
                                "Setting": config_label
                            })

                        for val in non_mem_list:
                            all_losses.append({
                                "Loss": val,
                                "Group": "Non-Member",
                                "Epsilon": epsilon if epsilon else "non-private",
                                "Operation": operation_label,
                                "Setting": config_label
                            })


    loss_df = pd.DataFrame(all_losses)
    plt.figure(figsize=(5, 3))
    g = sns.FacetGrid(loss_df, col="Epsilon", height=2.5, aspect=1.2, sharey=False)
    #g = sns.FacetGrid(loss_df, col="Operation", col_wrap=4, height=5, aspect=1.5)
    g.map_dataframe(sns.violinplot, x="Operation", y="Loss", hue="Group", palette=sns.color_palette("Paired"), linewidth=0.1)
    #g.add_legend()
    g.add_legend(bbox_to_anchor=(0.9, 0.87), loc='upper right')
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("Morphology", "Loss")
    g.tight_layout()

    output_path = f"loss_distr_results_chinchilla/faceted_boxplot_{dataset}.pdf"
    sns.despine()

    g.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved faceted boxplot to {output_path}")

