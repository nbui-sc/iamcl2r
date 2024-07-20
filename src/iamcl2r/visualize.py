import matplotlib.pyplot as plt
import matplotlib.colors as colors



def visualize_compatibility_matrix(comp_mat, save_path=None, vmin=0, vmax=80):
    fig, ax = plt.subplots()
    # scale color map to 0-100
    cmap = plt.get_cmap('viridis')
    cax = ax.imshow(comp_mat * 100, cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax))

    fig.colorbar(cax, ax=ax)


    def get_text_color(value, cmap='viridis', threshold=0.5):
        """
        Calculate text color (black or white) based on the luminance of the background color.
        """
        cmap = plt.get_cmap(cmap)
        rgba = cmap(value)
        # Calculate luminance of the background color
        luminance = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
        return 'white' if luminance < threshold else 'black'

    # Loop over data dimensions and create text annotations with contrast color.
    for i in range(comp_mat.shape[0]):
        for j in range(comp_mat.shape[1]):
            value = comp_mat[i, j] * 100
            text_color = get_text_color(value / 100)  # Normalize value for colormap
            text = ax.text(j, i, f"{value:.2f}%", ha="center", va="center", color=text_color)

    ax.set_title('Compatibility Matrix')
    ax.set_xlabel('Query Model ID')
    ax.set_ylabel('Gallery Model ID')
    ax.set_xticks(range(comp_mat.shape[1]))
    ax.set_yticks(range(comp_mat.shape[0]))
    ax.set_xticklabels(range(1, comp_mat.shape[1] + 1))
    ax.set_yticklabels(range(1, comp_mat.shape[0] + 1))
    ax.set_aspect('equal')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)


def visualize_average_accuracy(comp_mat, replace_ids, save_path=None):
    plt.rcParams.update({
        'text.usetex': False,
        'text.latex.preamble': r'\usepackage{amsfonts}'
    })
    plt.rcParams["figure.figsize"] = (10,8)

    values = average_accuracy(matrix=comp_mat, per_task=True)
    ntasks = comp_mat.shape[0]

    fig, ax = plt.subplots()
    replacemenet_ids = replace_ids[1:]
    for idx in replacemenet_ids:
      plt.axvline(x=idx+1, color='#DDDDDD', linestyle='solid', lw=1.5, zorder=-5)
    plt.grid(axis='y', color='#DDDDDD', linestyle='solid', lw=1.5, zorder=-5)
    x = np.arange(1, ntasks+1)

    face_colors = ['white' if not i in replacemenet_ids else 'yellow' for i in range(len(x))]
    cols = ['#D92929' if not i in replacemenet_ids else 'red' for i in range(len(x))]
    sizes = [150 if i in replacemenet_ids else 50 for i in range(len(x))]

    plt.plot(x, values,  color='#D92929', linestyle='solid', lw=2, zorder=2)
    plt.scatter(x, values, s=sizes, marker='s', edgecolors=cols, lw=1.5, facecolors=face_colors, zorder=2)

    plt.xticks(x)
    plt.xlim(0.7, ntasks + 0.2)
    ax.set_xlabel('Task')
    ax.set_ylabel(r'${AA}_{t}$')
    plt.legend(r"$d$-Simplex-HOC")
    plt.ylim(27, 73)
    plt.title('Average Accuracy per Task')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
