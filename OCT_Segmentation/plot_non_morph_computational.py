import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

def comutational_plotting(item, value):

    sns.set_style("whitegrid", {"axes.facecolor": "white"})
    plt.rcParams.update({
        'font.size': 20,          # General font size for all text
        'axes.titlesize': 20,     # Font size of plot titles
        'axes.labelsize': 20,     # Font size of axis labels
        'xtick.labelsize': 20,    # Font size of x-axis tick labels
        'ytick.labelsize': 20,    # Font size of y-axis tick labels
        'legend.fontsize': 16,    # Font size of the legend
        'figure.titlesize': 20,   # Font size of figure titles
    })

    df = pd.read_excel("oct_final_data.xlsx")

    duke_data_new = df[df['dataset'] == 'duke']
    umn_data_new = df[df['dataset'] == 'umn']

    duke_data_new['type'].replace('private', 'Opacus', inplace=True)
    umn_data_new['type'].replace('private', 'Opacus', inplace=True)
    duke_data_new['type'].replace('normal', 'non-private', inplace=True)
    umn_data_new['type'].replace('normal', 'non-private', inplace=True)

    duke_grouped_pcie_rx_mean = duke_data_new.groupby(['model', 'type'])[item].mean().unstack()
    umn_grouped_pcie_rx_mean = umn_data_new.groupby(['model', 'type'])[item].mean().unstack()

    fig, ax = plt.subplots(1, 2, figsize=(6, 4))
    colors = sns.color_palette("deep", n_colors=3)

    # Custom labels for x-axis
    duke_x_labels = ['LFUNet', ' FCN8s', 'NestedUNet', 'U-Net','y-net-gen']
    umn_x_labels =  ['LFUNet', ' FCN8s', 'NestedUNet', 'U-Net','y-net-gen']


    duke_grouped_pcie_rx_mean.plot(kind='bar', ax=ax[0], color=colors,width=0.8)
    ax[0].set_xticklabels(duke_x_labels, rotation=45, ha='right', fontsize=15)
    ax[0].set_ylabel(value, fontsize=20)
    #ax[0].set_yscale('log', base=10)
    ax[0].legend().remove()
    ax[0].set_xlabel('Duke', fontsize=20)


    umn_grouped_pcie_rx_mean.plot(kind='bar', ax=ax[1], color=colors,width=0.8)
    ax[1].set_xticklabels(umn_x_labels, rotation=45, ha='right', fontsize=15)
    #ax[1].set_ylabel('', fontsize=20)
    #ax[1].set_yscale('log', base=10)
    ax[1].legend(loc='upper right', bbox_to_anchor=(1, 1), frameon=False, fontsize=14)
    ax[1].legend().remove()
    ax[1].set_xlabel('UMN', fontsize=20)

    # Ensure the same y-axis scale for both plots
    y_min = min(duke_grouped_pcie_rx_mean.min().min(), umn_grouped_pcie_rx_mean.min().min())
    y_max = max(duke_grouped_pcie_rx_mean.max().max(), umn_grouped_pcie_rx_mean.max().max())
    for a in ax:
        a.set_ylim(0, y_max)

    for a in ax:
        a.grid(False)
        sns.despine(ax=a, left=True, bottom=True)


    plt.subplots_adjust(wspace=0)
    plt.tight_layout()
    output_path = os.path.join('computational_images', f'duke_umn_{item}_side_by_side.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()


column_name={'smoccupancy_dcgm_mean':'SMOC','Execution Time (minutes)':'Time (Min)','gpumemory_max':'GPU Mem','smactive_dcgm_mean':'SMACT','cpu_top_mean':'CPU util','sysmemory_top_mean':'System Mem','energy (MJ)':'GPU Energy','pow_dcgm_mean':'Power (W)'}
for item, value in column_name.items():
    comutational_plotting(item,value)


