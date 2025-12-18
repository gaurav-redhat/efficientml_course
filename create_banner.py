#!/usr/bin/env python3
"""Generate a banner/logo image for EfficientML course."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Polygon
import numpy as np

def create_banner():
    """Create a professional banner for the course."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Gradient background
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect='auto', cmap='Blues', 
              extent=[0, 1, 0, 1], alpha=0.15, zorder=0)
    ax.set_facecolor('#0a0f1a')
    
    # Decorative elements - neural network nodes
    np.random.seed(42)
    for _ in range(30):
        x, y = np.random.random(), np.random.random()
        size = np.random.uniform(0.005, 0.015)
        alpha = np.random.uniform(0.1, 0.3)
        circle = Circle((x, y), size, facecolor='#3b82f6', 
                        edgecolor='none', alpha=alpha)
        ax.add_patch(circle)
    
    # Draw connecting lines between some nodes
    for _ in range(15):
        x1, y1 = np.random.random(), np.random.random()
        x2, y2 = np.random.random(), np.random.random()
        ax.plot([x1, x2], [y1, y2], color='#3b82f6', alpha=0.1, linewidth=0.5)
    
    # Main logo area - left side
    # Efficiency icon (lightning bolt style)
    bolt_x = 0.12
    bolt_y = 0.5
    
    # Draw a stylized chip/processor icon
    chip_size = 0.12
    rect = FancyBboxPatch((bolt_x - chip_size/2, bolt_y - chip_size/2), 
                          chip_size, chip_size,
                          boxstyle='round,pad=0.02', 
                          facecolor='#1e40af', 
                          edgecolor='#60a5fa', 
                          alpha=0.9, linewidth=2)
    ax.add_patch(rect)
    
    # Inner chip details
    inner_size = 0.06
    inner_rect = FancyBboxPatch((bolt_x - inner_size/2, bolt_y - inner_size/2), 
                                inner_size, inner_size,
                                boxstyle='round,pad=0.01', 
                                facecolor='#3b82f6', 
                                edgecolor='none', alpha=0.9)
    ax.add_patch(inner_rect)
    
    # Lightning bolt symbol inside
    ax.text(bolt_x, bolt_y, '⚡', fontsize=24, ha='center', va='center', 
            color='#fbbf24', fontweight='bold')
    
    # Chip pins
    for i in range(4):
        # Top pins
        ax.plot([bolt_x - 0.04 + i*0.027, bolt_x - 0.04 + i*0.027], 
                [bolt_y + chip_size/2, bolt_y + chip_size/2 + 0.03], 
                color='#60a5fa', linewidth=2, alpha=0.7)
        # Bottom pins
        ax.plot([bolt_x - 0.04 + i*0.027, bolt_x - 0.04 + i*0.027], 
                [bolt_y - chip_size/2, bolt_y - chip_size/2 - 0.03], 
                color='#60a5fa', linewidth=2, alpha=0.7)
    
    # Main title
    ax.text(0.35, 0.62, 'EfficientML.ai', fontsize=42, fontweight='bold', 
            va='center', color='white',
            fontfamily='sans-serif')
    
    # Subtitle
    ax.text(0.35, 0.42, 'MIT 6.5940 | Fall 2023', fontsize=18, 
            va='center', color='#94a3b8',
            fontfamily='sans-serif')
    
    # Tagline
    ax.text(0.35, 0.28, 'Making ML Fast, Small & Affordable', fontsize=14, 
            va='center', color='#60a5fa', style='italic',
            fontfamily='sans-serif')
    
    # Right side - key topics
    topics = [
        ('Pruning', '#ef4444'),
        ('Quantization', '#f59e0b'),
        ('NAS', '#22c55e'),
        ('Distillation', '#3b82f6'),
        ('TinyML', '#a855f7'),
    ]
    
    start_x = 0.78
    for i, (topic, color) in enumerate(topics):
        y = 0.75 - i * 0.12
        # Small colored dot
        circle = Circle((start_x - 0.02, y), 0.012, facecolor=color, 
                        edgecolor='none', alpha=0.9)
        ax.add_patch(circle)
        ax.text(start_x, y, topic, fontsize=11, va='center', 
                color='#e2e8f0', fontfamily='sans-serif')
    
    # Bottom bar with stats
    bar_y = 0.08
    stats = [
        ('18 Lectures', '#3b82f6'),
        ('Prof. Song Han', '#22c55e'),
        ('Hands-on Demos', '#f59e0b'),
    ]
    
    for i, (stat, color) in enumerate(stats):
        x = 0.2 + i * 0.28
        ax.text(x, bar_y, stat, fontsize=11, ha='center', va='center', 
                color=color, fontweight='bold', fontfamily='sans-serif')
        if i < 2:
            ax.text(x + 0.14, bar_y, '|', fontsize=11, ha='center', 
                    va='center', color='#475569')
    
    # Decorative line
    ax.plot([0.08, 0.92], [0.16, 0.16], color='#1e3a5f', linewidth=1, alpha=0.5)
    
    plt.tight_layout(pad=0.5)
    fig.savefig('banner.png', dpi=200, facecolor='#0a0f1a', 
                edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Created banner.png")

if __name__ == '__main__':
    create_banner()

