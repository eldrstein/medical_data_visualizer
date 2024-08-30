import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1: Read the entire CSV file into a DataFrame
file_name = 'medical_examination.csv'
df = pd.read_csv(file_name)

# 2: Add BMI and convert data
data['overweight'] = np.where(data['weight'] / (data['height'] / 100) ** 2 > 25, 1, 0)

# 3: Normalize
data['gluc'] = np.where(data['gluc'] > 1, 1, 0)
data['cholesterol'] = np.where(data['cholesterol'] > 1, 1, 0)


# 4: Define the function to draw the categorical plot
def draw_cat_plot():
    # 5:
    df_cat = pd.melt(data, id_vars=['cardio'],
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'],
                     var_name='variable', value_name='value')

    # 6: Convert the categorical data to long format
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size().rename(columns={'size': 'total'})

    # 7: Plot the value counts of the categorical features split by 'cardio'
    cat_plot = sns.catplot(x='variable', hue='value', col='cardio', data=df_cat, kind='count', height=5, aspect=1.2)

    # 8: Store the figure in the fig variable
    fig = cat_plot.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11: Clean the data
    df_heat = data[
        (data['ap_lo'] <= data['ap_hi']) &
        (data['height'] >= data['height'].quantile(0.025)) &
        (data['height'] <= data['height'].quantile(0.975)) &
        (data['weight'] >= data['weight'].quantile(0.025)) &
        (data['weight'] <= data['weight'].quantile(0.975))
        ]

    # 12: Calculate the correlation matrix
    corr = df_heat.corr()

    # 13: Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14: Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # 15: Draw the heatmap using Seaborn
    sns.heatmap(
        corr,
        mask=mask,  # Apply the mask
        annot=True,  # Annotate each cell with the correlation value
        fmt='.1f',  # Format the annotations with one decimal point
        cmap='coolwarm',  # Color map for heatmap
        vmax=.3,  # Maximum value for the colormap scale
        center=0,  # Center the colormap at 0
        square=True,  # Make cells square-shaped
        linewidths=.5,  # Line width between cells
        cbar_kws={"shrink": .5},  # Color bar size
        ax=ax  # The axis to draw the heatmap on
    )

    # 16: Save the figure
    fig.savefig('heatmap.png')
    return fig

