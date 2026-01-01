"""
===============================================================================
FIGURE 8: Multi-fidelity Transfer Learning Results (v2 - Clean)
===============================================================================
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = Path('/Users/narjisse/Documents/Effat Courses/deeponet')
OUTPUT_DIR = DATA_PATH / 'publication_figures'
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
# CREATE FIGURE
# ============================================================================

def create_figure8():
    fig = plt.figure(figsize=(13, 5))
    
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3,
                          left=0.06, right=0.94, top=0.85, bottom=0.12)
    
    ax1 = fig.add_subplot(gs[0])  # Workflow
    ax2 = fig.add_subplot(gs[1])  # Bar chart
    
    # ===== (a) Transfer Learning Workflow - SIMPLIFIED =====
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 5)
    ax1.axis('off')
    ax1.set_title('(a) Transfer Learning Workflow', fontsize=12, 
                  fontweight='bold', pad=12)
    
    # MEDIUM box
    medium_box = FancyBboxPatch((0.3, 1.5), 3.2, 2.5, 
                                 facecolor='#E3F2FD', edgecolor='#1976D2',
                                 boxstyle='round,pad=0.05', linewidth=2)
    ax1.add_patch(medium_box)
    ax1.text(1.9, 3.6, 'MEDIUM', fontsize=12, fontweight='bold', 
            ha='center', color='#1565C0')
    ax1.text(1.9, 3.1, '20×20×20', fontsize=10, ha='center', color='#333')
    ax1.text(1.9, 2.6, '500 sims', fontsize=10, ha='center', color='#333')
    ax1.text(1.9, 2.1, '40 epochs', fontsize=10, ha='center', color='#333')
    
    # Arrow with label
    ax1.annotate('', xy=(6.2, 2.75), xytext=(3.7, 2.75),
                arrowprops=dict(arrowstyle='-|>', color='#333', lw=2,
                              mutation_scale=15))
    ax1.text(4.95, 3.2, 'Transfer', fontsize=10, ha='center', fontweight='bold', color='#333')
    
    # HIGH box
    high_box = FancyBboxPatch((6.2, 1.5), 3.2, 2.5,
                               facecolor='#E8F5E9', edgecolor='#4CAF50',
                               boxstyle='round,pad=0.05', linewidth=2)
    ax1.add_patch(high_box)
    ax1.text(7.8, 3.6, 'HIGH', fontsize=12, fontweight='bold',
            ha='center', color='#2E7D32')
    ax1.text(7.8, 3.1, '40×40×20', fontsize=10, ha='center', color='#333')
    ax1.text(7.8, 2.6, '100 sims', fontsize=10, ha='center', color='#333')
    ax1.text(7.8, 2.1, '20 epochs', fontsize=10, ha='center', color='#333')
    
    # Results summary box at bottom
    results_box = FancyBboxPatch((1.5, 0.2), 6.8, 1.0,
                                  facecolor='#FAFAFA', edgecolor='#666',
                                  boxstyle='round,pad=0.05', linewidth=1)
    ax1.add_patch(results_box)
    
    ax1.text(4.9, 0.85, 'Results:', fontsize=10, fontweight='bold', ha='center', color='#333')
    ax1.text(4.9, 0.45, 'Zero-shot R²=0.968  →  Fine-tuned R²=0.996  (+2.9%)', 
            fontsize=10, ha='center', color='#333')
    
    # ===== (b) Performance Comparison Bar Chart =====
    ax2.set_title('(b) Performance Comparison', fontsize=12, 
                  fontweight='bold', pad=12)
    
    configs = ['MEDIUM\nbaseline', 'HIGH\nZero-shot', 'HIGH\nFine-tuned']
    r2_values = [0.9965, 0.9679, 0.9961]
    colors = ['#1976D2', '#FF9800', '#4CAF50']
    
    x_pos = np.arange(len(configs))
    bars = ax2.bar(x_pos, r2_values, color=colors, edgecolor='white', 
                   linewidth=2, width=0.55)
    
    # Value labels inside bars
    for bar, r2 in zip(bars, r2_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.012,
                f'{r2:.4f}', ha='center', va='top', fontsize=11, 
                fontweight='bold', color='white')
    
    # Reference line
    ax2.axhline(y=0.99, color='#2E7D32', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.text(2.4, 0.9905, 'R²=0.99', fontsize=9, color='#2E7D32', ha='right')
    
    # Improvement annotation - cleaner
    ax2.annotate('+2.9%', xy=(2, 0.9961), xytext=(1.5, 0.958),
                fontsize=10, fontweight='bold', color='#4CAF50', ha='center',
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))
    
    ax2.set_ylabel('Rollout $R^2$ Score', fontsize=12)
    ax2.set_ylim(0.95, 1.005)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(configs, fontsize=10)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Clean spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Key insight - smaller, at bottom
    ax2.text(1, 0.952, '8× resolution with 20% training data', 
            fontsize=9, ha='center', style='italic', color='#555')
    
    plt.tight_layout()
    
    # ==================== SAVE ====================
    output_png = OUTPUT_DIR / 'Figure8_Transfer_Learning.png'
    output_pdf = OUTPUT_DIR / 'Figure8_Transfer_Learning.pdf'
    
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
    print("="*70)
    print("GENERATING FIGURE 8: Multi-fidelity Transfer Learning (v2)")
    print("="*70)
    
    create_figure8()
    
    print("\n" + "="*70)
    print("✅ FIGURE 8 GENERATED!")
    print("="*70)
