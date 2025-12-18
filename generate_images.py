#!/usr/bin/env python3
"""Generate infographic images for EfficientML course lectures."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Arrow, Circle, FancyArrowPatch
import numpy as np
import os

# Set style
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# Color schemes for different topics
COLORS = {
    'primary': '#2563eb',      # Blue
    'secondary': '#7c3aed',    # Purple
    'accent': '#059669',       # Green
    'warning': '#d97706',      # Orange
    'danger': '#dc2626',       # Red
    'dark': '#1f2937',         # Dark gray
    'light': '#f3f4f6',        # Light gray
    'white': '#ffffff',
}

def create_gradient_background(ax, color1='#1a1a2e', color2='#16213e'):
    """Create a gradient background."""
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect='auto', cmap=plt.cm.Blues, 
              extent=[0, 1, 0, 1], alpha=0.3, zorder=0)
    ax.set_facecolor(color1)

def add_title_box(ax, title, subtitle='', y_pos=0.92):
    """Add a styled title box."""
    ax.text(0.5, y_pos, title, fontsize=18, fontweight='bold', 
            ha='center', va='top', color=COLORS['white'],
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['primary'], 
                     edgecolor='none', alpha=0.9))
    if subtitle:
        ax.text(0.5, y_pos - 0.08, subtitle, fontsize=11, 
                ha='center', va='top', color=COLORS['light'], style='italic')

def create_lecture_01():
    """Introduction - Why Efficient ML?"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    
    add_title_box(ax, 'Why Efficient ML Matters?', 'The Growing Scale Challenge')
    
    # Model size growth bars
    models = [('GPT-2', 1.5, '#60a5fa'), ('GPT-3', 175, '#a78bfa'), 
              ('GPT-4', 1800, '#f472b6'), ('Gemini', 1500, '#34d399')]
    
    ax.text(0.15, 0.72, 'Model Size Growth (Billion Params)', fontsize=12, 
            fontweight='bold', color=COLORS['white'])
    
    max_val = 1800
    for i, (name, size, color) in enumerate(models):
        y = 0.65 - i * 0.12
        width = (size / max_val) * 0.6
        rect = FancyBboxPatch((0.15, y - 0.03), width, 0.06,
                              boxstyle='round,pad=0.01', facecolor=color, 
                              edgecolor='none', alpha=0.8)
        ax.add_patch(rect)
        ax.text(0.13, y, name, fontsize=10, ha='right', va='center', 
                color=COLORS['white'], fontweight='bold')
        ax.text(0.15 + width + 0.02, y, f'{size}B', fontsize=9, 
                ha='left', va='center', color=color)
    
    # Efficiency metrics
    metrics = [
        ('Latency', 'Real-time responses'),
        ('Memory', 'GPU VRAM limits'),
        ('Energy', 'Data center costs'),
        ('Cost', '$4.6M to train GPT-3'),
    ]
    
    ax.text(0.65, 0.72, 'Key Challenges', fontsize=12, 
            fontweight='bold', color=COLORS['white'])
    
    for i, (metric, desc) in enumerate(metrics):
        y = 0.62 - i * 0.11
        circle = Circle((0.62, y), 0.025, facecolor=COLORS['accent'], 
                        edgecolor='none', alpha=0.8)
        ax.add_patch(circle)
        ax.text(0.66, y, metric, fontsize=11, fontweight='bold', 
                va='center', color=COLORS['white'])
        ax.text(0.66, y - 0.035, desc, fontsize=9, va='center', 
                color=COLORS['light'])
    
    # Bottom insight
    ax.text(0.5, 0.08, '"Making models smaller, faster, and cheaper while maintaining accuracy"',
            fontsize=11, ha='center', va='center', color=COLORS['accent'],
            style='italic', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig('01_introduction/overview.png', dpi=150, facecolor='#0f172a', 
                edgecolor='none', bbox_inches='tight')
    plt.close()

def create_lecture_02():
    """Basics - Compute Primitives."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    
    add_title_box(ax, 'Neural Network Compute Basics', 'FLOPs, Memory & Roofline Model')
    
    # Roofline model visualization
    ax.text(0.25, 0.72, 'Roofline Model', fontsize=12, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    # Draw roofline
    x_roof = [0.05, 0.2, 0.45]
    y_roof = [0.35, 0.65, 0.65]
    ax.plot(x_roof, y_roof, 'w-', linewidth=3, alpha=0.9)
    ax.fill_between(x_roof, [0.25]*3, y_roof, alpha=0.2, color=COLORS['primary'])
    
    ax.text(0.12, 0.42, 'Memory\nBound', fontsize=9, ha='center', 
            color=COLORS['warning'], fontweight='bold')
    ax.text(0.35, 0.58, 'Compute\nBound', fontsize=9, ha='center', 
            color=COLORS['accent'], fontweight='bold')
    ax.text(0.25, 0.28, 'Arithmetic Intensity (FLOPs/Byte)', fontsize=8, 
            ha='center', color=COLORS['light'])
    
    # Operation costs table
    ax.text(0.72, 0.72, 'Operation Costs', fontsize=12, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    ops = [
        ('Linear Layer', 'O(in x out)', 'Medium'),
        ('Conv2D', 'O(C x K^2 x HW)', 'High'),
        ('Attention', 'O(N^2 x d)', 'Very High'),
        ('Element-wise', 'O(N)', 'Low'),
    ]
    
    for i, (op, flops, mem) in enumerate(ops):
        y = 0.62 - i * 0.1
        rect = FancyBboxPatch((0.52, y - 0.03), 0.4, 0.07,
                              boxstyle='round,pad=0.01', facecolor=COLORS['dark'], 
                              edgecolor=COLORS['primary'], alpha=0.7, linewidth=1)
        ax.add_patch(rect)
        ax.text(0.54, y, op, fontsize=10, va='center', color=COLORS['white'], 
                fontweight='bold')
        ax.text(0.9, y, flops, fontsize=9, va='center', ha='right', 
                color=COLORS['accent'])
    
    # Key equation
    ax.text(0.5, 0.12, 'MACs = Multiply-Accumulate Operations    |    FLOPs = 2 x MACs',
            fontsize=11, ha='center', va='center', color=COLORS['warning'],
            fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                                         facecolor=COLORS['dark'], alpha=0.8))
    
    plt.tight_layout()
    fig.savefig('02_basics/overview.png', dpi=150, facecolor='#0f172a', 
                edgecolor='none', bbox_inches='tight')
    plt.close()

def create_lecture_03():
    """Pruning Part 1 - Magnitude Pruning."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    
    add_title_box(ax, 'Neural Network Pruning', 'Remove Unnecessary Weights')
    
    # Before/After visualization
    ax.text(0.25, 0.72, 'Before Pruning', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    ax.text(0.75, 0.72, 'After 90% Pruning', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    # Draw weight matrices
    np.random.seed(42)
    for col, (x_start, sparse) in enumerate([(0.08, False), (0.58, True)]):
        for i in range(6):
            for j in range(6):
                x = x_start + j * 0.055
                y = 0.4 + i * 0.05
                if sparse and np.random.random() > 0.1:
                    color = '#374151'
                    alpha = 0.3
                else:
                    color = COLORS['primary'] if not sparse else COLORS['accent']
                    alpha = 0.8
                rect = FancyBboxPatch((x, y), 0.045, 0.04,
                                      boxstyle='round,pad=0.01', facecolor=color, 
                                      edgecolor='none', alpha=alpha)
                ax.add_patch(rect)
    
    # Arrow between matrices
    ax.annotate('', xy=(0.52, 0.52), xytext=(0.45, 0.52),
                arrowprops=dict(arrowstyle='->', color=COLORS['warning'], lw=3))
    
    # Pipeline
    ax.text(0.5, 0.22, 'Pruning Pipeline', fontsize=12, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    steps = ['Train\nFull Model', 'Prune\nWeights', 'Fine-tune', 'Deploy']
    for i, step in enumerate(steps):
        x = 0.15 + i * 0.22
        rect = FancyBboxPatch((x - 0.07, 0.08), 0.14, 0.08,
                              boxstyle='round,pad=0.02', facecolor=COLORS['primary'], 
                              edgecolor='none', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, 0.12, step, fontsize=9, ha='center', va='center', 
                color=COLORS['white'], fontweight='bold')
        if i < 3:
            ax.annotate('', xy=(x + 0.1, 0.12), xytext=(x + 0.07, 0.12),
                        arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2))
    
    # Result box
    ax.text(0.5, 0.02, '90%+ weights removed with <1% accuracy drop!',
            fontsize=11, ha='center', va='center', color=COLORS['accent'],
            fontweight='bold')
    
    plt.tight_layout()
    fig.savefig('03_pruning_sparsity_1/overview.png', dpi=150, facecolor='#0f172a', 
                edgecolor='none', bbox_inches='tight')
    plt.close()

def create_lecture_04():
    """Pruning Part 2 - Lottery Ticket."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    
    add_title_box(ax, 'Lottery Ticket Hypothesis', 'Finding Winning Subnetworks')
    
    # Lottery ticket concept
    ax.text(0.5, 0.72, '"Dense networks contain sparse subnetworks that can train\nto match the full network accuracy"',
            fontsize=10, ha='center', va='center', color=COLORS['light'], style='italic')
    
    # Process flow
    steps = [
        ('1. Train', 'Full network'),
        ('2. Prune', 'Remove 80%'),
        ('3. Reset', 'Original init'),
        ('4. Retrain', 'Same accuracy!'),
    ]
    
    for i, (title, desc) in enumerate(steps):
        x = 0.12 + i * 0.22
        # Box
        rect = FancyBboxPatch((x - 0.08, 0.45), 0.16, 0.15,
                              boxstyle='round,pad=0.02', 
                              facecolor=COLORS['secondary'] if i == 3 else COLORS['dark'], 
                              edgecolor=COLORS['primary'], alpha=0.8, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 0.56, title, fontsize=11, ha='center', va='center', 
                color=COLORS['white'], fontweight='bold')
        ax.text(x, 0.49, desc, fontsize=9, ha='center', va='center', 
                color=COLORS['light'])
        if i < 3:
            ax.annotate('', xy=(x + 0.12, 0.52), xytext=(x + 0.08, 0.52),
                        arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2))
    
    # Structured vs Unstructured
    ax.text(0.3, 0.32, 'Unstructured Pruning', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    ax.text(0.3, 0.27, 'Individual weights\nHigh compression (10-100x)\nNeeds sparse hardware', 
            fontsize=9, color=COLORS['light'], ha='center')
    
    ax.text(0.7, 0.32, 'Structured Pruning', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    ax.text(0.7, 0.27, 'Entire channels/filters\nLower compression (2-4x)\nWorks on any hardware', 
            fontsize=9, color=COLORS['light'], ha='center')
    
    # NVIDIA 2:4 sparsity
    ax.text(0.5, 0.1, 'NVIDIA 2:4 Sparsity: 2x speedup with 50% sparsity on Ampere GPUs',
            fontsize=10, ha='center', va='center', color=COLORS['accent'],
            fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                                         facecolor=COLORS['dark'], alpha=0.8))
    
    plt.tight_layout()
    fig.savefig('04_pruning_sparsity_2/overview.png', dpi=150, facecolor='#0f172a', 
                edgecolor='none', bbox_inches='tight')
    plt.close()

def create_lecture_05():
    """Quantization Part 1 - PTQ."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    
    add_title_box(ax, 'Quantization Basics', 'FP32 to INT8 = 4x Memory Savings')
    
    # Data type comparison
    types = [
        ('FP32', '32 bits', '4 bytes', COLORS['danger']),
        ('FP16', '16 bits', '2 bytes', COLORS['warning']),
        ('INT8', '8 bits', '1 byte', COLORS['accent']),
        ('INT4', '4 bits', '0.5 byte', COLORS['primary']),
    ]
    
    ax.text(0.25, 0.72, 'Data Types', fontsize=12, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    for i, (name, bits, size, color) in enumerate(types):
        y = 0.62 - i * 0.1
        width = (32 - i * 8) / 32 * 0.35
        rect = FancyBboxPatch((0.08, y - 0.025), width, 0.05,
                              boxstyle='round,pad=0.01', facecolor=color, 
                              edgecolor='none', alpha=0.8)
        ax.add_patch(rect)
        ax.text(0.06, y, name, fontsize=10, ha='right', va='center', 
                color=COLORS['white'], fontweight='bold')
        ax.text(0.08 + width + 0.02, y, f'{bits} | {size}', fontsize=9, 
                ha='left', va='center', color=color)
    
    # Quantization formula
    ax.text(0.7, 0.72, 'Quantization Formula', fontsize=12, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    formulas = [
        'q = round(x / scale) + zero_point',
        'x_approx = (q - zero_point) * scale',
    ]
    
    for i, formula in enumerate(formulas):
        y = 0.6 - i * 0.1
        ax.text(0.7, y, formula, fontsize=10, ha='center', va='center', 
                color=COLORS['accent'], family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['dark'], alpha=0.8))
    
    # PTQ Pipeline
    ax.text(0.5, 0.32, 'Post-Training Quantization (PTQ)', fontsize=12, 
            fontweight='bold', color=COLORS['white'], ha='center')
    
    ptq_steps = ['Pretrained\nModel', 'Calibrate', 'Quantize', 'Deploy\nINT8']
    for i, step in enumerate(ptq_steps):
        x = 0.15 + i * 0.22
        rect = FancyBboxPatch((x - 0.07, 0.15), 0.14, 0.1,
                              boxstyle='round,pad=0.02', facecolor=COLORS['primary'], 
                              edgecolor='none', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, 0.2, step, fontsize=9, ha='center', va='center', 
                color=COLORS['white'], fontweight='bold')
        if i < 3:
            ax.annotate('', xy=(x + 0.1, 0.2), xytext=(x + 0.07, 0.2),
                        arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2))
    
    ax.text(0.5, 0.05, 'INT8 works great for most CNNs with <1% accuracy drop',
            fontsize=10, ha='center', color=COLORS['accent'], fontweight='bold')
    
    plt.tight_layout()
    fig.savefig('05_quantization_1/overview.png', dpi=150, facecolor='#0f172a', 
                edgecolor='none', bbox_inches='tight')
    plt.close()

def create_lecture_06():
    """Quantization Part 2 - QAT and LLM."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    
    add_title_box(ax, 'Advanced Quantization', 'QAT & LLM Quantization')
    
    # QAT concept
    ax.text(0.25, 0.72, 'Quantization-Aware Training', fontsize=11, 
            fontweight='bold', color=COLORS['white'], ha='center')
    
    ax.text(0.25, 0.65, 'Train with fake quantization\nModel learns to be robust', 
            fontsize=9, color=COLORS['light'], ha='center')
    
    # STE visualization
    ax.text(0.25, 0.52, 'Straight-Through Estimator', fontsize=10, 
            fontweight='bold', color=COLORS['accent'], ha='center')
    ax.text(0.25, 0.45, 'Forward: y = round(x)\nBackward: grad = grad (pretend identity)', 
            fontsize=9, color=COLORS['light'], ha='center', family='monospace')
    
    # LLM Quantization challenges
    ax.text(0.75, 0.72, 'LLM Quantization', fontsize=11, 
            fontweight='bold', color=COLORS['white'], ha='center')
    
    methods = [
        ('GPTQ', '4-bit, uses Hessian'),
        ('AWQ', '4-bit, activation-aware'),
        ('SmoothQuant', 'Balances W and A'),
        ('LLM.int8()', 'Mixed precision'),
    ]
    
    for i, (method, desc) in enumerate(methods):
        y = 0.6 - i * 0.1
        ax.text(0.6, y, method, fontsize=10, fontweight='bold', 
                color=COLORS['primary'], va='center')
        ax.text(0.72, y, desc, fontsize=9, color=COLORS['light'], va='center')
    
    # Results
    ax.text(0.5, 0.18, 'LLaMA-7B Results', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    results = 'FP16: 5.68 PPL  |  GPTQ 4-bit: 5.85 PPL  |  AWQ 4-bit: 5.78 PPL'
    ax.text(0.5, 0.1, results, fontsize=10, ha='center', color=COLORS['accent'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['dark'], alpha=0.8))
    
    ax.text(0.5, 0.03, '4-bit LLMs work surprisingly well!', fontsize=10, 
            ha='center', color=COLORS['warning'], fontweight='bold')
    
    plt.tight_layout()
    fig.savefig('06_quantization_2/overview.png', dpi=150, facecolor='#0f172a', 
                edgecolor='none', bbox_inches='tight')
    plt.close()

def create_lecture_07():
    """NAS Part 1."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    
    add_title_box(ax, 'Neural Architecture Search', 'Automating Network Design')
    
    # NAS components
    ax.text(0.5, 0.72, 'NAS = Search Space + Strategy + Evaluation', 
            fontsize=11, color=COLORS['accent'], ha='center', fontweight='bold')
    
    components = [
        ('Search Space', 'Conv3x3, Conv5x5\nDepthwise, Skip', 0.18),
        ('Strategy', 'RL, Evolution\nGradient (DARTS)', 0.5),
        ('Evaluation', 'Train & validate\nWeight sharing', 0.82),
    ]
    
    for name, desc, x in components:
        rect = FancyBboxPatch((x - 0.12, 0.45), 0.24, 0.18,
                              boxstyle='round,pad=0.02', facecolor=COLORS['dark'], 
                              edgecolor=COLORS['primary'], alpha=0.8, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 0.58, name, fontsize=11, ha='center', va='center', 
                color=COLORS['white'], fontweight='bold')
        ax.text(x, 0.5, desc, fontsize=9, ha='center', va='center', 
                color=COLORS['light'])
    
    # DARTS
    ax.text(0.5, 0.35, 'DARTS: Differentiable NAS', fontsize=12, 
            fontweight='bold', color=COLORS['white'], ha='center')
    
    ax.text(0.5, 0.28, 'output = a1*op1(x) + a2*op2(x) + a3*op3(x)', 
            fontsize=10, ha='center', color=COLORS['accent'], family='monospace')
    ax.text(0.5, 0.22, 'Learn architecture weights a jointly with network weights', 
            fontsize=9, ha='center', color=COLORS['light'])
    
    # Cost comparison
    ax.text(0.5, 0.12, 'Cost Comparison', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    costs = 'NASNet: 2000 GPU-days  |  DARTS: 1 GPU-day  |  Same accuracy!'
    ax.text(0.5, 0.05, costs, fontsize=10, ha='center', color=COLORS['warning'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['dark'], alpha=0.8))
    
    plt.tight_layout()
    fig.savefig('07_neural_architecture_search_1/overview.png', dpi=150, 
                facecolor='#0f172a', edgecolor='none', bbox_inches='tight')
    plt.close()

def create_lecture_08():
    """NAS Part 2 - Hardware-Aware."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    
    add_title_box(ax, 'Hardware-Aware NAS', 'Optimizing for Real Devices')
    
    # ProxylessNAS concept
    ax.text(0.5, 0.72, 'Loss = Accuracy Loss + lambda * Latency Loss', 
            fontsize=11, color=COLORS['accent'], ha='center', fontweight='bold',
            family='monospace')
    
    # Latency table
    ax.text(0.25, 0.62, 'Latency Lookup Table', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    ops_latency = [
        ('Conv 3x3', '1.2ms'),
        ('Conv 5x5', '2.8ms'),
        ('DWConv 3x3', '0.3ms'),
        ('Skip', '0.0ms'),
    ]
    
    for i, (op, lat) in enumerate(ops_latency):
        y = 0.52 - i * 0.08
        ax.text(0.12, y, op, fontsize=9, color=COLORS['light'], va='center')
        ax.text(0.38, y, lat, fontsize=9, color=COLORS['accent'], 
                va='center', ha='right')
    
    # Once-for-All
    ax.text(0.75, 0.62, 'Once-for-All (OFA)', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    ax.text(0.75, 0.52, 'Train ONE supernet\nContains MANY subnetworks\nNo retraining needed!', 
            fontsize=9, color=COLORS['light'], ha='center')
    
    # Elastic dimensions
    dims = ['Depth', 'Width', 'Resolution']
    for i, dim in enumerate(dims):
        x = 0.62 + i * 0.1
        rect = FancyBboxPatch((x - 0.04, 0.32), 0.08, 0.06,
                              boxstyle='round,pad=0.01', facecolor=COLORS['primary'], 
                              edgecolor='none', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, 0.35, dim, fontsize=8, ha='center', va='center', 
                color=COLORS['white'])
    
    # Results
    ax.text(0.5, 0.18, 'Same accuracy, 50% less training cost, any hardware target!',
            fontsize=11, ha='center', color=COLORS['accent'], fontweight='bold')
    
    ax.text(0.5, 0.08, 'EfficientNet: Scale depth x width x resolution together\nalpha x beta^2 x gamma^2 = 2 (FLOPs constraint)',
            fontsize=9, ha='center', color=COLORS['light'])
    
    plt.tight_layout()
    fig.savefig('08_neural_architecture_search_2/overview.png', dpi=150, 
                facecolor='#0f172a', edgecolor='none', bbox_inches='tight')
    plt.close()

def create_lecture_09():
    """Knowledge Distillation."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    
    add_title_box(ax, 'Knowledge Distillation', 'Teacher-Student Learning')
    
    # Teacher-Student diagram
    # Teacher
    rect_t = FancyBboxPatch((0.1, 0.5), 0.2, 0.15,
                            boxstyle='round,pad=0.02', facecolor=COLORS['danger'], 
                            edgecolor='none', alpha=0.8)
    ax.add_patch(rect_t)
    ax.text(0.2, 0.575, 'Teacher', fontsize=12, ha='center', va='center', 
            color=COLORS['white'], fontweight='bold')
    ax.text(0.2, 0.52, 'ResNet-152\n60M params', fontsize=9, ha='center', 
            va='center', color=COLORS['light'])
    
    # Arrow
    ax.annotate('', xy=(0.55, 0.575), xytext=(0.32, 0.575),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=3))
    ax.text(0.43, 0.62, 'Soft Labels\n(T=4)', fontsize=9, ha='center', 
            color=COLORS['accent'])
    
    # Student
    rect_s = FancyBboxPatch((0.55, 0.52), 0.15, 0.11,
                            boxstyle='round,pad=0.02', facecolor=COLORS['accent'], 
                            edgecolor='none', alpha=0.8)
    ax.add_patch(rect_s)
    ax.text(0.625, 0.575, 'Student', fontsize=12, ha='center', va='center', 
            color=COLORS['white'], fontweight='bold')
    ax.text(0.625, 0.54, 'MobileNet\n3M params', fontsize=9, ha='center', 
            va='center', color=COLORS['light'])
    
    # Soft vs Hard labels
    ax.text(0.3, 0.38, 'Why Soft Labels?', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    ax.text(0.15, 0.3, 'Hard: [1, 0, 0, 0]', fontsize=9, color=COLORS['light'], 
            family='monospace')
    ax.text(0.15, 0.24, 'Soft: [0.7, 0.2, 0.05, 0.05]', fontsize=9, 
            color=COLORS['accent'], family='monospace')
    ax.text(0.15, 0.18, 'Contains class similarity info!', fontsize=9, 
            color=COLORS['warning'])
    
    # Loss formula
    ax.text(0.7, 0.38, 'Distillation Loss', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    ax.text(0.7, 0.28, 'L = a * KL(soft) + (1-a) * CE(hard)', fontsize=10, 
            color=COLORS['accent'], ha='center', family='monospace')
    
    # Results
    ax.text(0.5, 0.08, 'DistilBERT: 40% smaller, 60% faster, 97% of BERT performance',
            fontsize=10, ha='center', color=COLORS['accent'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['dark'], alpha=0.8))
    
    plt.tight_layout()
    fig.savefig('09_knowledge_distillation/overview.png', dpi=150, 
                facecolor='#0f172a', edgecolor='none', bbox_inches='tight')
    plt.close()

def create_lecture_10():
    """MCUNet & TinyML."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    
    add_title_box(ax, 'TinyML & MCUNet', 'ML on Microcontrollers')
    
    # Memory comparison
    devices = [
        ('Server GPU', '80GB', 80000, COLORS['danger']),
        ('Smartphone', '6GB', 6000, COLORS['warning']),
        ('Raspberry Pi', '4GB', 4000, COLORS['primary']),
        ('MCU (STM32)', '320KB', 0.32, COLORS['accent']),
    ]
    
    ax.text(0.25, 0.72, 'Memory Comparison', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    for i, (device, mem, val, color) in enumerate(devices):
        y = 0.62 - i * 0.1
        width = min(val / 80000 * 0.35, 0.35)
        rect = FancyBboxPatch((0.08, y - 0.025), max(width, 0.01), 0.05,
                              boxstyle='round,pad=0.01', facecolor=color, 
                              edgecolor='none', alpha=0.8)
        ax.add_patch(rect)
        ax.text(0.06, y, device, fontsize=9, ha='right', va='center', 
                color=COLORS['white'])
        ax.text(0.44, y, mem, fontsize=9, ha='left', va='center', color=color)
    
    ax.text(0.25, 0.2, 'MCUs have 1000x less\nmemory than phones!', 
            fontsize=10, ha='center', color=COLORS['warning'], fontweight='bold')
    
    # MCUNet approach
    ax.text(0.75, 0.72, 'MCUNet: Co-Design', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    steps = [
        ('TinyNAS', 'Search arch for MCU'),
        ('TinyEngine', 'Optimized inference'),
    ]
    
    for i, (name, desc) in enumerate(steps):
        y = 0.58 - i * 0.15
        rect = FancyBboxPatch((0.58, y - 0.04), 0.34, 0.1,
                              boxstyle='round,pad=0.02', facecolor=COLORS['dark'], 
                              edgecolor=COLORS['primary'], alpha=0.8, linewidth=2)
        ax.add_patch(rect)
        ax.text(0.75, y + 0.02, name, fontsize=10, ha='center', va='center', 
                color=COLORS['accent'], fontweight='bold')
        ax.text(0.75, y - 0.02, desc, fontsize=9, ha='center', va='center', 
                color=COLORS['light'])
    
    # Result
    ax.text(0.5, 0.08, 'First ImageNet on MCU: 62.2% accuracy with 320KB SRAM',
            fontsize=10, ha='center', color=COLORS['accent'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['dark'], alpha=0.8))
    
    plt.tight_layout()
    fig.savefig('10_mcunet_tinyml/overview.png', dpi=150, facecolor='#0f172a', 
                edgecolor='none', bbox_inches='tight')
    plt.close()

def create_lecture_11():
    """Efficient Transformers."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    
    add_title_box(ax, 'Efficient Transformers', 'Solving O(N^2) Attention')
    
    # Attention complexity
    ax.text(0.5, 0.72, 'Standard Attention: O(N^2) memory and compute', 
            fontsize=11, color=COLORS['danger'], ha='center', fontweight='bold')
    
    # Sequence length scaling
    lengths = [('512', '1MB'), ('2K', '16MB'), ('8K', '256MB'), ('32K', '4GB')]
    ax.text(0.25, 0.62, 'Memory vs Sequence Length', fontsize=10, 
            fontweight='bold', color=COLORS['white'], ha='center')
    
    for i, (seq, mem) in enumerate(lengths):
        y = 0.52 - i * 0.08
        ax.text(0.15, y, f'N={seq}:', fontsize=9, color=COLORS['light'], va='center')
        ax.text(0.35, y, mem, fontsize=9, color=COLORS['warning'], 
                va='center', ha='right')
    
    # Solutions
    ax.text(0.75, 0.62, 'Efficient Solutions', fontsize=10, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    solutions = [
        ('FlashAttention', 'O(N^2) compute, O(N) memory'),
        ('Sparse Attention', 'O(N * sqrt(N))'),
        ('Linear Attention', 'O(N)'),
        ('MQA/GQA', 'Reduce KV cache'),
    ]
    
    for i, (name, desc) in enumerate(solutions):
        y = 0.52 - i * 0.1
        ax.text(0.58, y, name, fontsize=9, fontweight='bold', 
                color=COLORS['accent'], va='center')
        ax.text(0.58, y - 0.03, desc, fontsize=8, color=COLORS['light'], va='center')
    
    # FlashAttention insight
    ax.text(0.5, 0.12, 'FlashAttention: Same math, better memory access patterns',
            fontsize=10, ha='center', color=COLORS['white'], fontweight='bold')
    ax.text(0.5, 0.05, 'Result: 2-4x faster, exact computation, O(N) memory',
            fontsize=10, ha='center', color=COLORS['accent'])
    
    plt.tight_layout()
    fig.savefig('11_efficient_transformers/overview.png', dpi=150, 
                facecolor='#0f172a', edgecolor='none', bbox_inches='tight')
    plt.close()

def create_lecture_12():
    """Efficient Training."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    
    add_title_box(ax, 'Efficient Training', 'Memory & Speed Optimization')
    
    # Memory breakdown
    ax.text(0.25, 0.72, 'Training Memory', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    components = [
        ('Weights', 'W', COLORS['primary']),
        ('Gradients', 'W', COLORS['secondary']),
        ('Optimizer', '2-8W', COLORS['warning']),
        ('Activations', 'Huge!', COLORS['danger']),
    ]
    
    for i, (name, size, color) in enumerate(components):
        y = 0.6 - i * 0.1
        rect = FancyBboxPatch((0.08, y - 0.03), 0.25, 0.07,
                              boxstyle='round,pad=0.01', facecolor=color, 
                              edgecolor='none', alpha=0.7)
        ax.add_patch(rect)
        ax.text(0.1, y, name, fontsize=9, va='center', color=COLORS['white'])
        ax.text(0.31, y, size, fontsize=9, va='center', ha='right', 
                color=COLORS['white'])
    
    ax.text(0.25, 0.22, 'Total: 12-16x model size!', fontsize=10, 
            ha='center', color=COLORS['danger'], fontweight='bold')
    
    # Optimization techniques
    ax.text(0.72, 0.72, 'Optimization Techniques', fontsize=11, 
            fontweight='bold', color=COLORS['white'], ha='center')
    
    techniques = [
        ('Mixed Precision', '2x memory, 2-3x speed'),
        ('Gradient Checkpoint', '3-5x memory, +30% time'),
        ('8-bit Optimizer', '2x optimizer memory'),
        ('LoRA', '10-100x fewer params'),
    ]
    
    for i, (tech, benefit) in enumerate(techniques):
        y = 0.6 - i * 0.1
        ax.text(0.55, y, tech, fontsize=9, fontweight='bold', 
                color=COLORS['accent'], va='center')
        ax.text(0.55, y - 0.03, benefit, fontsize=8, color=COLORS['light'], 
                va='center')
    
    # torch.compile
    ax.text(0.5, 0.08, 'torch.compile() = 1.5-2x speedup with one line of code!',
            fontsize=10, ha='center', color=COLORS['accent'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['dark'], alpha=0.8))
    
    plt.tight_layout()
    fig.savefig('12_efficient_training/overview.png', dpi=150, 
                facecolor='#0f172a', edgecolor='none', bbox_inches='tight')
    plt.close()

def create_lecture_13():
    """On-Device Training."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    
    add_title_box(ax, 'On-Device Training', 'Training on Edge Devices')
    
    # Why on-device
    benefits = [
        ('Privacy', 'Data stays on device'),
        ('Personalization', 'Adapt to user'),
        ('Offline', 'No cloud needed'),
    ]
    
    ax.text(0.25, 0.72, 'Why On-Device?', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    for i, (name, desc) in enumerate(benefits):
        y = 0.6 - i * 0.1
        circle = Circle((0.1, y), 0.02, facecolor=COLORS['accent'], 
                        edgecolor='none', alpha=0.8)
        ax.add_patch(circle)
        ax.text(0.14, y, name, fontsize=10, fontweight='bold', 
                va='center', color=COLORS['white'])
        ax.text(0.14, y - 0.035, desc, fontsize=9, va='center', 
                color=COLORS['light'])
    
    # TinyTL
    ax.text(0.72, 0.72, 'TinyTL: Bias-Only Training', fontsize=11, 
            fontweight='bold', color=COLORS['white'], ha='center')
    
    ax.text(0.72, 0.6, 'Freeze all weights\nOnly train biases\n= 10x less memory!', 
            fontsize=10, color=COLORS['light'], ha='center')
    
    # Memory comparison
    ax.text(0.72, 0.4, 'Memory Reduction', fontsize=10, fontweight='bold', 
            color=COLORS['accent'], ha='center')
    
    mem_data = [('Full training', '100%'), ('TinyTL', '11%'), 
                ('+ Quantization', '8%')]
    for i, (method, mem) in enumerate(mem_data):
        y = 0.32 - i * 0.07
        ax.text(0.58, y, method, fontsize=9, color=COLORS['light'], va='center')
        ax.text(0.86, y, mem, fontsize=9, color=COLORS['accent'], 
                va='center', ha='right')
    
    # Federated Learning
    ax.text(0.5, 0.08, 'Federated Learning: Train across devices without sharing data',
            fontsize=10, ha='center', color=COLORS['accent'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['dark'], alpha=0.8))
    
    plt.tight_layout()
    fig.savefig('13_on_device_training/overview.png', dpi=150, 
                facecolor='#0f172a', edgecolor='none', bbox_inches='tight')
    plt.close()

def create_lecture_14():
    """Distributed Training."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    
    add_title_box(ax, 'Distributed Training', 'Scaling to Large Models')
    
    # Why distributed
    ax.text(0.5, 0.72, 'GPT-3 (175B) needs 350GB+ memory - no single GPU can fit it!', 
            fontsize=10, color=COLORS['warning'], ha='center', fontweight='bold')
    
    # Parallelism types
    types = [
        ('Data Parallel', 'Same model, different data', COLORS['primary']),
        ('Tensor Parallel', 'Split matrices', COLORS['secondary']),
        ('Pipeline Parallel', 'Split layers', COLORS['accent']),
    ]
    
    for i, (name, desc, color) in enumerate(types):
        x = 0.18 + i * 0.32
        rect = FancyBboxPatch((x - 0.12, 0.5), 0.24, 0.12,
                              boxstyle='round,pad=0.02', facecolor=color, 
                              edgecolor='none', alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, 0.58, name, fontsize=10, ha='center', va='center', 
                color=COLORS['white'], fontweight='bold')
        ax.text(x, 0.52, desc, fontsize=8, ha='center', va='center', 
                color=COLORS['light'])
    
    # ZeRO
    ax.text(0.5, 0.4, 'ZeRO Optimization', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    zero_stages = [
        ('DDP', 'Everything replicated', '1x'),
        ('ZeRO-1', 'Partition optimizer', '4x'),
        ('ZeRO-2', '+ Partition gradients', '8x'),
        ('ZeRO-3', '+ Partition params', 'Linear'),
    ]
    
    for i, (stage, desc, saving) in enumerate(zero_stages):
        x = 0.12 + i * 0.22
        y = 0.28
        ax.text(x, y + 0.04, stage, fontsize=9, fontweight='bold', 
                ha='center', color=COLORS['accent'])
        ax.text(x, y, desc, fontsize=8, ha='center', color=COLORS['light'])
        ax.text(x, y - 0.04, f'Memory: {saving}', fontsize=8, ha='center', 
                color=COLORS['warning'])
    
    # 3D parallelism
    ax.text(0.5, 0.08, '3D Parallelism: Combine all three for maximum scale (GPT-3, PaLM)',
            fontsize=10, ha='center', color=COLORS['accent'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['dark'], alpha=0.8))
    
    plt.tight_layout()
    fig.savefig('14_distributed_training/overview.png', dpi=150, 
                facecolor='#0f172a', edgecolor='none', bbox_inches='tight')
    plt.close()

def create_lecture_15():
    """Efficient Vision Models."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    
    add_title_box(ax, 'Efficient Vision Models', 'MobileNet, EfficientNet & Beyond')
    
    # Depthwise Separable
    ax.text(0.3, 0.72, 'Depthwise Separable Conv', fontsize=11, 
            fontweight='bold', color=COLORS['white'], ha='center')
    
    ax.text(0.3, 0.62, 'Standard: C_in x C_out x K^2 x HW', fontsize=9, 
            color=COLORS['danger'], ha='center', family='monospace')
    ax.text(0.3, 0.55, 'DW Sep: C_in x (K^2 + C_out) x HW', fontsize=9, 
            color=COLORS['accent'], ha='center', family='monospace')
    ax.text(0.3, 0.48, 'Speedup: ~9x fewer FLOPs!', fontsize=10, 
            color=COLORS['warning'], ha='center', fontweight='bold')
    
    # Model evolution
    ax.text(0.75, 0.72, 'Evolution', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    models = [
        ('MobileNetV1', 'Depthwise separable'),
        ('MobileNetV2', 'Inverted residuals'),
        ('MobileNetV3', 'NAS + squeeze-excite'),
        ('EfficientNet', 'Compound scaling'),
    ]
    
    for i, (name, feature) in enumerate(models):
        y = 0.6 - i * 0.1
        ax.text(0.6, y, name, fontsize=9, fontweight='bold', 
                color=COLORS['primary'], va='center')
        ax.text(0.6, y - 0.03, feature, fontsize=8, color=COLORS['light'], 
                va='center')
    
    # Comparison table
    ax.text(0.5, 0.22, 'Model Comparison', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    comparison = 'ResNet-50: 25M, 4.1G FLOPs  |  MobileNetV3: 5.4M, 0.2G FLOPs'
    ax.text(0.5, 0.12, comparison, fontsize=9, ha='center', color=COLORS['light'])
    
    ax.text(0.5, 0.05, 'ConvNeXt: Modern CNN matching ViT with fewer tricks',
            fontsize=10, ha='center', color=COLORS['accent'], fontweight='bold')
    
    plt.tight_layout()
    fig.savefig('15_efficient_vision_models/overview.png', dpi=150, 
                facecolor='#0f172a', edgecolor='none', bbox_inches='tight')
    plt.close()

def create_lecture_16():
    """Efficient LLMs."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    
    add_title_box(ax, 'Efficient LLM Inference', 'KV Cache, Speculative Decoding & More')
    
    # LLM is memory-bound
    ax.text(0.5, 0.72, 'LLM Inference is Memory-Bound, not Compute-Bound', 
            fontsize=10, color=COLORS['warning'], ha='center', fontweight='bold')
    
    # KV Cache
    ax.text(0.25, 0.62, 'KV Cache Memory', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    kv_data = [
        ('LLaMA-7B, 2K ctx', '1GB'),
        ('LLaMA-7B, 32K ctx', '16GB'),
        ('LLaMA-70B, 32K ctx', '160GB'),
    ]
    
    for i, (config, mem) in enumerate(kv_data):
        y = 0.52 - i * 0.08
        ax.text(0.1, y, config, fontsize=9, color=COLORS['light'], va='center')
        ax.text(0.4, y, mem, fontsize=9, color=COLORS['danger'], 
                va='center', ha='right')
    
    # Solutions
    ax.text(0.75, 0.62, 'Optimization Stack', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    solutions = [
        ('MQA/GQA', 'Share KV heads'),
        ('PagedAttention', 'vLLM memory mgmt'),
        ('Speculative', '2-3x faster decode'),
        ('Continuous Batch', '4x throughput'),
    ]
    
    for i, (name, desc) in enumerate(solutions):
        y = 0.52 - i * 0.09
        ax.text(0.6, y, name, fontsize=9, fontweight='bold', 
                color=COLORS['accent'], va='center')
        ax.text(0.6, y - 0.03, desc, fontsize=8, color=COLORS['light'], 
                va='center')
    
    # Speculative decoding
    ax.text(0.5, 0.18, 'Speculative Decoding', fontsize=10, fontweight='bold', 
            color=COLORS['white'], ha='center')
    ax.text(0.5, 0.1, 'Small model drafts 5 tokens -> Large model verifies in 1 pass',
            fontsize=9, ha='center', color=COLORS['light'])
    ax.text(0.5, 0.04, 'Same quality, 2-3x faster!', fontsize=10, ha='center', 
            color=COLORS['accent'], fontweight='bold')
    
    plt.tight_layout()
    fig.savefig('16_efficient_llms/overview.png', dpi=150, 
                facecolor='#0f172a', edgecolor='none', bbox_inches='tight')
    plt.close()

def create_lecture_17():
    """Efficient Diffusion Models."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    
    add_title_box(ax, 'Efficient Diffusion Models', 'From 1000 Steps to 1 Step')
    
    # Steps comparison
    methods = [
        ('DDPM', 1000, '60s', COLORS['danger']),
        ('DDIM', 50, '3s', COLORS['warning']),
        ('DPM-Solver', 20, '1s', COLORS['primary']),
        ('LCM', 4, '0.3s', COLORS['accent']),
        ('SDXL Turbo', 1, '0.1s', COLORS['secondary']),
    ]
    
    ax.text(0.5, 0.72, 'Sampling Steps Evolution', fontsize=11, 
            fontweight='bold', color=COLORS['white'], ha='center')
    
    max_steps = 1000
    for i, (name, steps, time, color) in enumerate(methods):
        y = 0.6 - i * 0.09
        width = (steps / max_steps) * 0.5
        rect = FancyBboxPatch((0.15, y - 0.025), max(width, 0.02), 0.05,
                              boxstyle='round,pad=0.01', facecolor=color, 
                              edgecolor='none', alpha=0.8)
        ax.add_patch(rect)
        ax.text(0.13, y, name, fontsize=9, ha='right', va='center', 
                color=COLORS['white'])
        ax.text(0.67, y, f'{steps} steps', fontsize=8, ha='left', 
                va='center', color=color)
        ax.text(0.82, y, time, fontsize=8, ha='left', va='center', 
                color=COLORS['light'])
    
    # Key techniques
    ax.text(0.5, 0.2, 'Key Techniques', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    techniques = 'Latent Space (8x smaller) + Distillation + Consistency Models'
    ax.text(0.5, 0.12, techniques, fontsize=9, ha='center', color=COLORS['light'])
    
    ax.text(0.5, 0.05, '1-step generation is 1000x faster than original DDPM!',
            fontsize=10, ha='center', color=COLORS['accent'], fontweight='bold')
    
    plt.tight_layout()
    fig.savefig('17_efficient_diffusion_models/overview.png', dpi=150, 
                facecolor='#0f172a', edgecolor='none', bbox_inches='tight')
    plt.close()

def create_lecture_18():
    """Quantum ML."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    
    add_title_box(ax, 'Quantum Machine Learning', 'Future of Efficient ML?')
    
    # Quantum basics
    ax.text(0.3, 0.72, 'Quantum Advantage', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    ax.text(0.3, 0.62, 'Classical: n bits = 1 state', fontsize=10, 
            color=COLORS['light'], ha='center')
    ax.text(0.3, 0.55, 'Quantum: n qubits = 2^n states', fontsize=10, 
            color=COLORS['accent'], ha='center', fontweight='bold')
    
    # Potential speedups
    ax.text(0.3, 0.42, 'Potential Speedups', fontsize=10, fontweight='bold', 
            color=COLORS['warning'], ha='center')
    
    speedups = [
        ('Matrix inversion', 'O(N^3) -> O(log N)'),
        ('Sampling', 'Native quantum'),
        ('Optimization', 'Quantum tunneling'),
    ]
    
    for i, (task, speedup) in enumerate(speedups):
        y = 0.34 - i * 0.07
        ax.text(0.15, y, task, fontsize=9, color=COLORS['light'], va='center')
        ax.text(0.45, y, speedup, fontsize=9, color=COLORS['accent'], 
                va='center', ha='right')
    
    # Current status
    ax.text(0.75, 0.72, 'Current Reality', fontsize=11, fontweight='bold', 
            color=COLORS['white'], ha='center')
    
    status = [
        ('Hardware', 'NISQ era (noisy)'),
        ('Qubits', '50-1000'),
        ('Error rate', '0.1-1% per gate'),
        ('ML advantage', 'Not yet proven'),
    ]
    
    for i, (aspect, val) in enumerate(status):
        y = 0.6 - i * 0.1
        ax.text(0.6, y, aspect, fontsize=9, fontweight='bold', 
                color=COLORS['primary'], va='center')
        ax.text(0.6, y - 0.03, val, fontsize=8, color=COLORS['light'], 
                va='center')
    
    # Timeline
    ax.text(0.5, 0.08, 'Timeline: Educational now | Niche apps in 5-15 years | True advantage 15+ years',
            fontsize=9, ha='center', color=COLORS['light'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['dark'], alpha=0.8))
    
    plt.tight_layout()
    fig.savefig('18_quantum_ml/overview.png', dpi=150, 
                facecolor='#0f172a', edgecolor='none', bbox_inches='tight')
    plt.close()

def main():
    """Generate all images."""
    print("Generating lecture images...")
    
    create_lecture_01()
    print("  01_introduction/overview.png")
    
    create_lecture_02()
    print("  02_basics/overview.png")
    
    create_lecture_03()
    print("  03_pruning_sparsity_1/overview.png")
    
    create_lecture_04()
    print("  04_pruning_sparsity_2/overview.png")
    
    create_lecture_05()
    print("  05_quantization_1/overview.png")
    
    create_lecture_06()
    print("  06_quantization_2/overview.png")
    
    create_lecture_07()
    print("  07_neural_architecture_search_1/overview.png")
    
    create_lecture_08()
    print("  08_neural_architecture_search_2/overview.png")
    
    create_lecture_09()
    print("  09_knowledge_distillation/overview.png")
    
    create_lecture_10()
    print("  10_mcunet_tinyml/overview.png")
    
    create_lecture_11()
    print("  11_efficient_transformers/overview.png")
    
    create_lecture_12()
    print("  12_efficient_training/overview.png")
    
    create_lecture_13()
    print("  13_on_device_training/overview.png")
    
    create_lecture_14()
    print("  14_distributed_training/overview.png")
    
    create_lecture_15()
    print("  15_efficient_vision_models/overview.png")
    
    create_lecture_16()
    print("  16_efficient_llms/overview.png")
    
    create_lecture_17()
    print("  17_efficient_diffusion_models/overview.png")
    
    create_lecture_18()
    print("  18_quantum_ml/overview.png")
    
    print("\nDone! All images generated.")

if __name__ == '__main__':
    main()

