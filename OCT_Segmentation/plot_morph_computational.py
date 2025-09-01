import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_model_name(config):
    if 'FCN8s' in config:
        return 'FCN8s'
    elif 'NestedUNet' in config:
        return 'NestedUNet'
    elif 'LFUNet' in config:
        return 'LFUNet'
    elif 'unet' in config:
        return 'UNet'
    elif 'y_net_gen' in config:
        return 'y_net_gen'


def computational_plotting(item, value):

    sns.set_style("whitegrid", {"axes.facecolor": "white"})

    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 16,
        'figure.titlesize': 20,
    })


    df = pd.read_csv("data_computational_morph.csv")


    df['dataset'] = df['config'].apply(lambda x: 'Duke' if 'Duke' in x else 'UMN')
    df['model'] = df['config'].apply(lambda x: get_model_name(x))

    df['type'] = df['config'].apply(lambda x: 'private' if 'dpTrue' in x else 'non-private')

    duke_data = df[df['dataset'] == 'Duke']
    umn_data = df[df['dataset'] == 'UMN']


    duke_grouped_mean = duke_data.groupby(['model', 'type'])[item].mean().unstack()
    umn_grouped_mean = umn_data.groupby(['model', 'type'])[item].mean().unstack()


    fig, ax = plt.subplots(1, 2, figsize=(6, 4))
    colors = sns.color_palette("deep", n_colors=2)


    x_labels = ['FCN8s', 'NestedUNet', 'LFUNet', 'UNet', 'y_net_gen',]


    duke_grouped_mean.plot(kind='bar', ax=ax[0], color=colors, width=0.4)
    ax[0].set_xticklabels(x_labels, rotation=45, ha='right', fontsize=15)
    ax[0].set_ylabel(value, fontsize=20)
    ax[0].set_xlabel('Duke', fontsize=20)
    ax[0].legend().remove()
    if item == 'max_memory_used_MB':
        ax[0].set_yscale('log', base=10)


    umn_grouped_mean.plot(kind='bar', ax=ax[1], color=colors, width=0.4)
    ax[1].set_xticklabels(x_labels, rotation=45, ha='right', fontsize=15)
    ax[1].set_xlabel('UMN', fontsize=20)
    ax[1].legend(loc='upper right', bbox_to_anchor=(1, 1), frameon=False, fontsize=14)
    if item == 'max_memory_used_MB':
        ax[1].set_yscale('log', base=10)


    y_min = min(duke_grouped_mean.min().min(), umn_grouped_mean.min().min())
    y_max = max(duke_grouped_mean.max().max(), umn_grouped_mean.max().max())
    for a in ax:
        if item == 'max_memory_used_MB':
            a.set_ylim(10, y_max * 1.1)
        else:
            a.set_ylim(0, y_max * 1.1)


    for a in ax:
        a.grid(False)
        sns.despine(ax=a, left=True, bottom=True)


    plt.subplots_adjust(wspace=0)
    plt.tight_layout()

    os.makedirs('computational_images/morph', exist_ok=True)
    plt.savefig(os.path.join('computational_images/morph', f'{item}.pdf'), bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


column_name = {
    'max_memory_used_MB': 'GPU Memory (MB)',
    'dcgm_avg_SMACT': 'SM Active (%)',
    'dcgm_avg_SMOCC': 'SM Occupancy (%)',
    'dcgm_avg_POWER': 'Power (W)',
}


for item, value in column_name.items():
    computational_plotting(item, value)
