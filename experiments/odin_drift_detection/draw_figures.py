"""
In this file, we draw the figures for the p_values



"""

import os
import matplotlib.pyplot as plt
import json

def _load_json(path_list_idx):
    print(f"opening: {path_list_idx}")
    with open(path_list_idx, "r") as file:
        data = json.load(file)
    return data

def load_jsons(results_dir):


    jsons_filename = os.listdir(results_dir)
    all_results = {}

    for filename in jsons_filename:
        save_directory = os.path.join(results_dir, filename)
        all_results[filename] = _load_json(save_directory)

    return all_results



def draw_one_graph_clean(methods: dict, dataset_name, ax):
    color = ['#FFA62B', '#82C0CC', '#3f72af', '#82C0CC', '#3f72af']

    ### y axis
    low = 0
    high = 1

    step = (high - low) / 4
    steps = [low + i * step for i in range(6)]
    ax.set_ylim([low, high])
    ax.set_yticks(steps)


    #ax.set_xticklabels(xsteps,rotation=45,ha='right',rotation_mode='anchor')


    markers = ['o', '*', 'X', '8', 's']
    marker_sizes = [10, 20, 10, 10, 10]
    target_value = 0.05


    ax.plot(methods['p_values'], label = dataset_name)


    ax.axhline(y=target_value, color = 'r', linestyle='--', label = 'target'.upper())




def draw_graphs_clean(results_dict, experiment_name):
    fontsize = 12
    plt.rcParams['xtick.labelsize'] = fontsize - 2
    plt.rcParams['ytick.labelsize'] = fontsize - 2

    ### first we need to figure out how many rows / columns we need
    row_count = 0
    col_count = 0
    query_object = 'car'

    #### it will always be in the graphdict
    for dataset_name in results_dict.keys():
        col_count += 1

    row_count = 1 ## one for 95, one for 99

    # print(f'graph row, col: {row_count, col_count}')
    # print(f'len1: {len(results_dict)}, len2: {len(results_dict["seattle_cherry"])}')
    ### fig size should depend on how many graphs there are....
    fig_size = (col_count * 3, row_count * 2)

    fig, axes = plt.subplots(row_count, col_count, sharex=False, figsize=fig_size)

    row_idx = 0
    col_idx = 0
    for i, dataset_name in enumerate(results_dict):
        print(f"dataset: {dataset_name}")


        ax = axes[col_idx]

        ### format the ax before drawing it
        if row_idx == 0:
            ax.set_title(f'{dataset_name.upper()}', fontsize=fontsize)

        if col_idx == 0:
            ax.set_ylabel(f'P values',
                          fontweight='bold',
                          fontsize=fontsize,
                          rotation=90,
                          rotation_mode='anchor')

        draw_one_graph_clean(results_dict[dataset_name], dataset_name, ax)

        col_idx += 1

    handles, labels = ax.get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 0.55))

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4, hspace=0.4)

    directory = os.path.abspath(__file__)
    base = os.path.dirname(os.path.dirname(directory))
    os.makedirs(os.path.join(base, 'results'), exist_ok=True)
    directory = os.path.join(base, 'results', f'{experiment_name}_clean.png')
    print(f'saved to {directory}')
    fig.savefig(directory, bbox_extra_artists = (lgd,), bbox_inches='tight', dpi=300)







if __name__ == "__main__":
    ### first we load all the relevant files
    experiment_name = os.path.basename(os.getcwd())
    BASE_DIRECTORY = '/srv/data/jbang36/amoeba/experiments'

    ### save directory
    results_dir = os.path.join(BASE_DIRECTORY, experiment_name)


    results = load_jsons(results_dir)

    draw_graphs_clean(results, experiment_name)


