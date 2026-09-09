import matplotlib.pyplot as plt
import seaborn

# TODO: first loop evaluated with MSE --> then flow matching/diffusion model integration

# provare prima flow matching


def plot_loss(num_epochs:int, train_loss_values, test_loss_values, output_path:str):
    # Plot the loss curves
    plt.plot(num_epochs, train_loss_values, label="Train loss")
    plt.plot(num_epochs, test_loss_values, label="Test loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(output_path, dpi='300')