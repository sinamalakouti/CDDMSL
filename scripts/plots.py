import matplotlib.pyplot as plt
import numpy as np
def draw_motivation_barchart():
    import matplotlib.pyplot as plt
    import numpy as np

    # Define data
    datasets = ['CLIPART', 'COMIC', 'WATERCOLOR']
    models = ['RegionClip', 'ImageNet']
    performance = np.array([
        [90, 70],
        [80, 60],
        [85, 75]
    ])

    # Define colors for bars
    colors = {'RegionClip': 'blue', 'ImageNet': 'lightblue'}

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set bar positions and heights
    x_pos = np.arange(len(datasets))
    bar_width = (1.0 / len(models)) * 0.

    # Loop through each dataset and create side-by-side bars for each model
    for i, dataset in enumerate(datasets):
        heights = performance[i]
        for j, model in enumerate(models):
            ax.bar(x_pos[i] + j * bar_width, heights[j], width=bar_width, color=colors[model], align='edge')

    # Remove tick lines
    ax.tick_params(axis='both', length=0)

    # Remove x-axis tick labels and label
    ax.set_xticklabels([])
    ax.set_xlabel('')

    # Remove y-axis ticks and label
    ax.set_yticks([])
    ax.set_ylabel('')

    # Set the legend
    legend_entries = []
    for model in models:
        legend_entries.append(ax.bar(x_pos[0], 0, color=colors[model], label=model))
    ax.legend(handles=legend_entries)

    # Remove border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    fig.patch.set_alpha(0)

    # Display the chart
    plt.show()


# Function to check if a value is -1
def is_valid_delta(delta):
    return [val for val in delta if val != -1]

def draw_staiblity_plots():

    import matplotlib.pyplot as plt
    # compute detlas here


    # OURS
    ours_delta_clipart = 40.4 - (38.7 + 39.8)/2
    ours_delta_watercolor = 49.73 - (49.8+49.4) / 2
    ours_delta_comic = 46.3 - (45.9+44.5) / 2

    # Adative-MT RESNET 50
    AD_delta_resnet50_clipart = 30.5 - (29.0 +25.7)/2
    AD_delta_resnet50_watercolor = 43.7 - (40.6+42.3) / 2
    AD_delta_resnet50_comic = 23.4 - (22.2+24.3) / 2

    # Adative-MT RESNET 101

    AD_delta_resnet101_clipart = 49.3 - (44.5)
    AD_delta_resnet101_watercolor = 59.9 - (56.7)



    # MT RESNET 101

    # AD_delta_resnet101_clipart = c - (34.2)
    # AD_delta_resnet101_watercolor = c - (47.6)

    # IRG RESNET

    # IRG_delta_clipart = d - (31.5 )
    IRG_delta_watercolor =  (53.0) - 48.1


    # Define the delta values for each method and dataset
    datasets = ['Clipart', 'Watercolor', 'Comic']
    my_method_deltas = [ours_delta_clipart, ours_delta_watercolor, ours_delta_comic]
    other_AD_RN50_deltas = [AD_delta_resnet50_clipart, AD_delta_resnet50_watercolor, AD_delta_resnet50_comic]
    other_AD_RN101_deltas = [AD_delta_resnet101_clipart, AD_delta_resnet101_watercolor,-1]
    IRG_delta_watercolor = [-1,IRG_delta_watercolor,-1]

    # Set up the plot
    plt.figure(figsize=(10, 6))
    plt.xlabel('Target Dataset', fontsize=12)
    plt.ylabel('$\Delta$ Values', fontsize=12)
    # plt.title('Comparison of Deltas between Methods')


    # Plot the lines for each dataset
    my_method_line, = plt.plot([val for i, val in enumerate(datasets) if my_method_deltas[i] != -1], is_valid_delta(my_method_deltas), marker='o', label='Ours')
    AD50_line, = plt.plot([val for i, val in enumerate(datasets) if other_AD_RN50_deltas[i] != -1], is_valid_delta(other_AD_RN50_deltas), marker='o', label='Adaptive-MT-RN50')
    AD101_line, = plt.plot([val for i, val in enumerate(datasets) if other_AD_RN101_deltas[i] != -1], is_valid_delta(other_AD_RN101_deltas), marker='o', label='Adaptive-MT-RN101')
    IRG_line, = plt.plot([val for i, val in enumerate(datasets) if IRG_delta_watercolor[i] != -1],
                                    is_valid_delta(IRG_delta_watercolor), marker='o', label='IRG')

    # Annotate the delta values
    for x, y in zip([val for i, val in enumerate(datasets) if my_method_deltas[i] != -1], is_valid_delta(my_method_deltas)):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')

    for x, y in zip([val for i, val in enumerate(datasets) if other_AD_RN50_deltas[i] != -1], is_valid_delta(other_AD_RN50_deltas)):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')

    for x, y in zip([val for i, val in enumerate(datasets) if other_AD_RN101_deltas[i] != -1], is_valid_delta(other_AD_RN101_deltas)):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')
    for x, y in zip([val for i, val in enumerate(datasets) if IRG_delta_watercolor[i] != -1], is_valid_delta(IRG_delta_watercolor)):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')

    # Add legend and grid
    plt.legend([my_method_line, AD50_line, AD101_line,IRG_line], ['Ours', 'Adaptive-MT-RN50', 'Adaptive-MT-RN101', 'IRG'])
    # plt.grid(True)

    plt.savefig('delta_comparison_plot.png')

    # Show the plot
    plt.show()


