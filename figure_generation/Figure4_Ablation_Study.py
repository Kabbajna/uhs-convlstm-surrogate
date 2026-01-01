"""
===============================================================================
FIGURE 4: Ablation Study - Architecture Comparison (Q1 STYLE)
===============================================================================
Clean grouped bar chart - publication ready
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path('/Users/narjisse/Documents/Effat Courses/deeponet/publication_figures')
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'mathtext.fontset': 'dejavuserif',
})

# ============================================================================
# DATA
# ============================================================================

models = [
    'ConvLSTM-UNet\n(Scheduled Sampling)',
    'ConvLSTM-UNet\n(Teacher Forcing)',
    '3D U-Net',
    'FNO'
]

single_step_r2 = [0.9965, 0.9918, 0.9803, 0.8956]
rollout_r2 = [0.9950, 0.9051, 0.0754, -12.99]
rollout_r2_display = [max(0, r) for r in rollout_r2]

# Colors - softer for Q1 style
color_single = '#2196F3'  # Material Blue
color_rollout = '#FF7043'  # Material Deep Orange

# ============================================================================
# CREATE FIGURE
# ============================================================================

def create_figure4():
    """Create Figure 4: Ablation Study - Q1 Style"""
    
    print("="*70)
    print("CREATING FIGURE 4: Ablation Study (Q1 Style)")
    print("="*70)
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    x = np.arange(len(models))
    width = 0.35
    
    # Bars
    bars1 = ax.bar(x - width/2, single_step_r2, width, 
                   label='Single-step', color=color_single, 
                   edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, rollout_r2_display, width, 
                   label='14-month rollout', color=color_rollout,
                   edgecolor='white', linewidth=1)
    
    # Value labels - INSIDE bars for clean look
    for bar, val in zip(bars1, single_step_r2):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height - 0.08,
               f'{val:.3f}', ha='center', va='top', 
               fontsize=9, fontweight='bold', color='white')
    
    for bar, val, orig in zip(bars2, rollout_r2_display, rollout_r2):
        height = bar.get_height()
        if orig < 0:
            # Failed model - label at bottom
            ax.text(bar.get_x() + bar.get_width()/2, 0.03,
                   f'Failed\n({orig:.1f})', ha='center', va='bottom',
                   fontsize=8, fontweight='bold', color=color_rollout)
        elif height < 0.15:
            # Short bar - label above
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                   f'{val:.3f}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold', color=color_rollout)
        else:
            # Normal - label inside
            ax.text(bar.get_x() + bar.get_width()/2, height - 0.08,
                   f'{val:.3f}', ha='center', va='top',
                   fontsize=9, fontweight='bold', color='white')
    
    # Best model indicator - simple star
    ax.plot(0, 1.02, marker='*', markersize=15, color='#4CAF50', 
            markeredgecolor='white', markeredgewidth=0.5)
    ax.text(0, 1.06, 'Best', ha='center', va='bottom', 
           fontsize=10, fontweight='bold', color='#4CAF50')
    
    # Axes
    ax.set_ylabel('$R^2$ Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_xlim(-0.6, len(models) - 0.4)
    
    # Legend - horizontal at top
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95,
             ncol=2, columnspacing=1)
    
    # Title
    ax.set_title('Ablation Study: Architecture Comparison',
                fontsize=13, fontweight='bold', pad=12)
    
    # Subtle grid
    ax.yaxis.grid(True, linestyle='-', alpha=0.2)
    ax.set_axisbelow(True)
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    plt.tight_layout()
    
    # ==================== SAVE ====================
    output_png = OUTPUT_DIR / 'Figure4_Ablation_Study.png'
    output_pdf = OUTPUT_DIR / 'Figure4_Ablation_Study.pdf'
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"\n✅ Figure saved:")
    print(f"   PNG: {output_png}")
    print(f"   PDF: {output_pdf}")
    
    plt.close()
    return output_png


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    create_figure4()
    print("\n✅ FIGURE 4 GENERATED!")
