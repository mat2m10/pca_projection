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
    
    
def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    
    # Extract loss values
    loss = history.history['loss']
    val_loss = history.history.get('val_loss', None)  # Use get() in case val_loss is missing
    
    epochs = range(1, len(loss) + 1)

    # Plot Training Loss
    plt.plot(epochs, loss, 'bo-', label='Training Loss')

    # Plot Validation Loss if available
    if val_loss:
        plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
