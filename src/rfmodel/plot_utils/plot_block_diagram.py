import matplotlib.pyplot as plt

def plot_pipeline(pipe):
    """
    Plots a linear block diagram of a Pipeline with arrows in the correct direction.
    
    Args:
        pipe (Pipeline): your pipeline instance
    """
    blocks = pipe.blocks
    n = len(blocks)
    
    fig, ax = plt.subplots(figsize=(2*n, 2))
    ax.axis("off")
    
    # Draw blocks
    for i, blk in enumerate(blocks):
        x = i * 2
        y = 0
        rect = plt.Rectangle((x, y-0.5), 1.5, 1, facecolor="lightblue", edgecolor="black")
        ax.add_patch(rect)
        ax.text(x + 0.75, y, blk.name, ha="center", va="center", fontsize=10)
    
        # Draw arrow to next block
        if i < n - 1:
            ax.annotate(
                "",
                xy=(x + 2, y),          # arrowhead
                xytext=(x + 1.5, y),    # tail of the arrow
                arrowprops=dict(arrowstyle="->", lw=2)
            )
    
    ax.set_xlim(-0.5, 2*n)
    ax.set_ylim(-1, 1)
    plt.show()

