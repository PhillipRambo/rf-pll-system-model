import matplotlib.pyplot as plt

def plot_pipeline(pipe):
    """
    Plots a pipeline where every mixer block is drawn as a mixer symbol
    in series with the signal path, and its PLL as an LO source below it.
    Dispatch is by block type, so mixers can be named freely in the YAML.
    """
    blocks = pipe.blocks
    n = len(blocks)
    
    fig, ax = plt.subplots(figsize=(2.5 * n, 4))
    ax.axis("off")
    
    # Coordinates and spacing
    block_width = 1.5
    block_height = 1.0
    spacing = 2.5
    
    for i, blk in enumerate(blocks):
        x = i * spacing
        y = 0
        
        if blk.type_name == "mixer":
            # --- Draw the Mixer (Circle with an 'X') ---
            mixer_center = (x + block_width/2, y)
            circle = plt.Circle(mixer_center, 0.4, facecolor="white", edgecolor="black", zorder=3)
            ax.add_patch(circle)
            
            # The 'X' inside the mixer
            ax.text(mixer_center[0], mixer_center[1], "X", ha="center", va="center", fontsize=14, fontweight='bold')
            ax.text(mixer_center[0], mixer_center[1] + 0.5, "Mixer", ha="center", va="bottom")

            # --- Draw the PLL (Rectangle below the mixer) ---
            pll_y = y - 1.5
            pll_rect = plt.Rectangle((x, pll_y - 0.4), block_width, 0.8, facecolor="lightcoral", edgecolor="black")
            ax.add_patch(pll_rect)
            ax.text(x + block_width/2, pll_y, "PLL / LO", ha="center", va="center", fontsize=10)

            # --- Draw the Connection (PLL -> Mixer) ---
            ax.annotate("", xy=(mixer_center[0], y - 0.4), xytext=(mixer_center[0], pll_y + 0.4),
                        arrowprops=dict(arrowstyle="->", lw=1.5, ls='--'))
            
            # Logic for arrows (Adjusting starting/ending points for circle vs rect)
            arrow_start = x + block_width/2 + 0.4 # edge of circle
            arrow_end = x + spacing
            
        else:
            # --- Standard Linear Block ---
            rect = plt.Rectangle((x, y - 0.4), block_width, 0.8, facecolor="lightblue", edgecolor="black")
            ax.add_patch(rect)
            ax.text(x + block_width/2, y, blk.name, ha="center", va="center", fontsize=10)
            
            arrow_start = x + block_width
            arrow_end = x + spacing

        # --- Draw Signal Path Arrow to next block ---
        if i < n - 1:
            ax.annotate("", xy=(arrow_end, y), xytext=(arrow_start, y),
                        arrowprops=dict(arrowstyle="->", lw=2))

    ax.set_xlim(-0.5, spacing * n)
    ax.set_ylim(-2.5, 1.5)
    plt.title("RF System Pipeline Schematic", loc='left', fontsize=12, pad=20)
    plt.tight_layout()
    plt.show()

# Usage:
# plot_pipeline(pipe)