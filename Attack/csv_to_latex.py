import pandas as pd


datasets = ['UMN', 'Duke']
dfs = []
for dataset in datasets:
    data = pd.read_csv(f'csv_results_chinchilla/f1_accuracy_{dataset}.csv')
    data['epsilon'] = data['epsilon'].fillna('Non-private')
    data['Operation'] = data.apply(lambda x: 'None' if x['Morphology'] == False else x['Operation'], axis=1)
    data['Dataset'] = dataset
    dfs.append(data)

data = pd.concat(dfs)
mean_df = (data[['Dataset', 'Operation', 'epsilon', 'f1_score', 'accuracy']])
mean_df.columns = ['dataset', 'morphology', 'epsilon', 'F1', 'accuracy']
mean_df = mean_df.groupby(['dataset', 'epsilon', 'morphology']).mean()
print(mean_df.to_latex(index=True, multirow=True, float_format="%.2f", label='tab:loss-attack',
                       caption='Comparison of global loss-based attack performances'))




#
#
# grouped['f1_score'] = grouped['f1_score'].round(3)
# grouped['accuracy'] = grouped['accuracy'].round(3)
#
#
# latex_table = r"""\documentclass{article}
# \usepackage{booktabs}
# \usepackage{siunitx}
# \usepackage{geometry}
# \geometry{a4paper, margin=1in}
#
# \begin{document}
#
# \begin{table}[h]
# \centering
# \caption{F1 Score and Accuracy for Private and Non-Private NestedUNet Models with Different Morphological Operations}
# \begin{tabular}{llSS}
# \toprule
# \textbf{DPSGD} & \textbf{Morphological Operation} & \textbf{F1 Score} & \textbf{Accuracy} \\
# \midrule
# """
#
#
# for _, row in grouped.iterrows():
#     dpsgd = 'Yes' if row['DPSGD'] else 'No'
#     epsilon = str(row['epsilon'])
#     operation = row['Operation'].capitalize() if row['Operation'] != 'None' else 'None'
#     f1_score = f"{row['f1_score']:.3f}"
#     accuracy = f"{row['accuracy']:.3f}"
#
#
#     if epsilon == 'None':
#         epsilon_display = 'Non-Private'
#     else:
#         epsilon_display = f"Private ($\\epsilon={epsilon}$)"
#
#     latex_table += f"{epsilon_display} & {operation} & {f1_score} & {accuracy} \\\\ \n"
#
#
# latex_table += r"""\bottomrule
# \end{tabular}
# \end{table}
#
# \end{document}
# """
#
#
# with open(f'csv_results_chinchilla/{dataset}_performance_table.tex', 'w') as f:
#     f.write(latex_table)
#
# print("LaTeX table code has been generated and saved to 'performance_table.tex'.")