# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

def make_population_plot(df, X, Y, hue, title, palette = 'rocket'):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x=X, 
        y=Y, 
        hue=hue, 
        palette=palette
    )

    # Add labels and title
    plt.title(title)
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Show the plot
    plt.show()