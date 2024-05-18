import itertools

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils import evaluate

def explore_gradient_directions(model, config, train_loader, val_loader, steps = 10, search_range = 1):
    """
    Calculate loss function in two random directions in the model parameter space.
    params: 
        steps: number of steps within the search range
        search_range: If k, steps are taken in [-k, k]
    """
    print("Preparing gradient directions")
    model.eval()   # Set the model to evaluation mode to disable dropout, etc.
    original_state_dict = {name: param.clone() for name, param in model.named_parameters() 
                            if param.requires_grad}

    # Generate random directions for each parameter
    random_directions = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Generate two random directions
            random_dir1 = torch.randn_like(param)
            random_dir2 = torch.randn_like(param)

            # Normalize each direction by its norm
            norm1 = torch.norm(random_dir1) + 1e-10
            norm2 = torch.norm(random_dir2) + 1e-10

            random_dir1.div_(norm1)
            random_dir2.div_(norm2)

            # Store the normalized directions in the dictionary
            random_directions[name] = [random_dir1, random_dir2]
    
    step_sizes = torch.linspace(-1 * search_range, 1 * search_range, steps=steps)  # Define step sizes, e.g., -1 to 1 in 10 steps
    print("step_sizes", step_sizes)

    # Results dictionary
    results = {}

    # Iterate over all combinations of step sizes for two directions
    for steps1, steps2 in itertools.product(step_sizes, repeat=2):
        # Apply each random direction with the respective step size
        for name, param in model.named_parameters():
            if param.requires_grad:
                direction1, direction2 = random_directions[name]
                perturbed_param = param + direction1 * steps1 + direction2 * steps2
                param.data.copy_(perturbed_param)

        # Compute losses
        train_acc, train_loss, val_acc, val_loss = evaluate(model, config, train_loader, val_loader)

        # Reset model parameters to original
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(original_state_dict[name])

        # Store results
        results[(steps1.item(), steps2.item())] = (train_acc, train_loss, val_acc, val_loss)

    return results

def plot_loss_acc_contours_2D(results, save_path, title='Acc and Loss Contours'):
    print("Plotting 2D contours")

    # Prepare the grid data
    step_sizes = sorted(set(key[0] for key in results.keys()))  # Unique sorted step sizes, e.g. (-1, -0.8, ..., 0.8, 1)
    num_steps, cont_range = len(step_sizes), int((max(step_sizes) - min(step_sizes)) / 2)  
    grid_train_acc = np.zeros((len(step_sizes), len(step_sizes)))
    grid_train_loss = np.zeros((len(step_sizes), len(step_sizes)))
    grid_val_acc = np.zeros((len(step_sizes), len(step_sizes)))
    grid_val_loss = np.zeros((len(step_sizes), len(step_sizes)))

    for (step1, step2), (train_acc, train_loss, val_acc, val_loss) in results.items():
        i = step_sizes.index(step1)
        j = step_sizes.index(step2)
        grid_train_acc[i, j] = train_acc
        grid_train_loss[i, j] = train_loss
        grid_val_acc[i, j] = val_acc
        grid_val_loss[i, j] = val_loss

    # Create a meshgrid for plotting
    step1, step2 = np.meshgrid(step_sizes, step_sizes)

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    fig.suptitle(title)

    grids = [grid_train_acc, grid_train_loss, grid_val_acc, grid_val_loss]
    titles = ['Training Acc', 'Training Loss', 'Val Acc', 'Val Loss']
    for ax, grid, title in zip(axs, grids, titles):
        sns.heatmap(grid, ax=ax, xticklabels=False, yticklabels=False, annot=False)
        ax.set_title(title)
        ax.set_xlabel('Step Size Direction 1')
        ax.set_ylabel('Step Size Direction 2')

    save_path = f'{save_path}_ticks{num_steps}_range{cont_range}'
    plt.savefig(save_path + ".pdf", format = 'pdf', bbox_inches = 'tight')
    print(f"Plot saved at {save_path}.pdf")
    plt.show()

def plot_loss_acc_contours_3D(results, save_path, title='Acc and Loss 3D Contours'):
    print("Plotting 3D contours")
    # Prepare the grid data
    step_sizes = sorted(set(key[0] for key in results.keys()))  # Unique step sizes
    num_steps, cont_range = len(step_sizes), int((max(step_sizes) - min(step_sizes)) / 2)  
    grid_train_acc = np.zeros((len(step_sizes), len(step_sizes)))
    grid_train_loss = np.zeros((len(step_sizes), len(step_sizes)))
    grid_val_acc = np.zeros((len(step_sizes), len(step_sizes)))
    grid_val_loss = np.zeros((len(step_sizes), len(step_sizes)))

    for (step1, step2), (train_acc, train_loss, val_acc, val_loss) in results.items():
        i = step_sizes.index(step1)
        j = step_sizes.index(step2)
        grid_train_acc[i, j] = train_acc
        grid_train_loss[i, j] = train_loss
        grid_val_acc[i, j] = val_acc
        grid_val_loss[i, j] = val_loss

    # Create a meshgrid for plotting
    step1, step2 = np.meshgrid(step_sizes, step_sizes)

    # Plotting
    fig = plt.figure(figsize=(20, 6))
    fig.suptitle(title)

    # Create 3D axes
    ax1 = fig.add_subplot(141, projection='3d')
    ax2 = fig.add_subplot(142, projection='3d')
    ax3 = fig.add_subplot(143, projection='3d')
    ax4 = fig.add_subplot(144, projection='3d')

    axs = [ax1, ax2, ax3, ax4]
    grids = [grid_train_acc, grid_train_loss, grid_val_acc, grid_val_loss]
    cmaps = ['viridis', 'viridis', 'magma', 'magma']
    titles = ['Training Acc', 'Training Loss', 'Val Acc', 'Val Loss']
    zlabels = ['Accuracy', 'Loss', 'Accuracy', 'Loss']
    for ax, grid, cmap, title, zlabel in zip(axs, grids, cmaps, titles, zlabels): 
        surf = ax.plot_surface(step1, step2, grid, cmap=cmap, edgecolor='none')
        ax.set_title(title)
        ax.set_xlabel('Step Size Direction 1')
        ax.set_ylabel('Step Size Direction 2')
        ax.set_zlabel(zlabel)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Save the plot
    save_path = f'{save_path}_ticks{num_steps}_range{cont_range}'
    plt.savefig(save_path + "_3D.pdf", format = 'pdf', bbox_inches = 'tight')
    print(f"Plot saved at {save_path}_3D.pdf")
    plt.show()