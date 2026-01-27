"""
Q-Learning Model Performance Visualizations

This module provides a collection of visualization functions for analyzing
Q-Learning agent performance, including convergence analysis, reward tracking,
state visitation patterns, and policy evaluation.

Functions:
    - plot_actions_over_time: Visualize agent actions over time
    - plot_water_levels: Plot water volume with max threshold
    - plot_water_levels_multipart: Multi-panel view of extended time periods
    - plot_cumulative_rewards: Cumulative reward progression
    - plot_daily_statistics: Daily min/max/mean water levels
    - plot_state_heatmap: 2D state visitation heatmap
    - plot_multiple_heatmaps: Create multiple heatmaps from state pairs
    - plot_action_distribution: Bar chart of action frequencies
    - plot_water_stability: Water level variance analysis
    - plot_reward_distribution: Distribution of rewards
    - create_performance_dashboard: Comprehensive multi-panel dashboard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Union, Dict
import warnings

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10


def plot_actions_over_time(
    action_history: List[int],
    title: str = "Agent Actions Over Time",
    num_steps: Optional[int] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> None:
    """
    Visualize agent actions over time with markers.
    
    Args:
        action_history: List of discrete action indices
        title: Plot title
        num_steps: Number of steps to plot (None = all)
        figsize: Figure size as (width, height)
    """
    actions = action_history[:num_steps] if num_steps else action_history
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(actions, marker='o', linewidth=2, markersize=6, color='#2E86AB')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Timestep (hours)", fontsize=11)
    ax.set_ylabel("Action Index", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_water_levels(
    water_levels: np.ndarray,
    max_volume: float,
    title: str = "Water Volume Over Time",
    num_steps: Optional[int] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> None:
    """
    Plot water volume over time with maximum threshold line.
    
    Args:
        water_levels: Array of water volumes
        max_volume: Maximum allowed water volume
        title: Plot title
        num_steps: Number of steps to plot (None = all)
        figsize: Figure size as (width, height)
    """
    levels = water_levels[:num_steps] if num_steps else water_levels
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(levels, linewidth=2, color='#06A77D', label='Water Volume')
    ax.axhline(max_volume, color='#D62828', linestyle='--', linewidth=2, 
               alpha=0.7, label='Maximum Volume')
    
    # Fill between 0 and the water level
    ax.fill_between(range(len(levels)), 0, levels, alpha=0.2, color='#06A77D')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Timestep (hours)", fontsize=11)
    ax.set_ylabel("Water Volume (m³)", fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_cumulative_rewards(
    rewards: Union[List[float], Dict[str, List[float]]],
    title: str = "Cumulative Reward Over Time",
    figsize: Tuple[int, int] = (14, 5),
    labels: Optional[List[str]] = None
) -> None:
    """
    Plot cumulative rewards over time (single or multiple lines).
    
    Args:
        rewards: Either a list of rewards, or a dict mapping labels to reward lists
        title: Plot title
        figsize: Figure size as (width, height)
        labels: Optional list of labels for multiple reward series (ignored if rewards is a dict)
    """
    colors = ['#F77F00', '#06A77D', '#D62828', '#F18F01', '#C1121F', '#073B4C']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle dict input (preferred for multiple lines)
    if isinstance(rewards, dict):
        for idx, (label, reward_list) in enumerate(rewards.items()):
            cumulative = np.cumsum(reward_list)
            color = colors[idx % len(colors)]
            ax.plot(cumulative, linewidth=2.5, label=label, color=color)
            ax.fill_between(range(len(cumulative)), 0, cumulative, alpha=0.15, color=color)
    
    # Handle list input (single line)
    else:
        cumulative = np.cumsum(rewards)
        ax.plot(cumulative, linewidth=2.5, color=colors[0])
        ax.fill_between(range(len(cumulative)), 0, cumulative, alpha=0.2, color=colors[0])
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Timestep (hours)", fontsize=11)
    ax.set_ylabel("Cumulative Reward", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add legend if multiple lines
    if isinstance(rewards, dict) or (labels and len(labels) > 1):
        ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.show()


def plot_daily_statistics(
    water_levels: np.ndarray,
    max_volume: float,
    hours_per_day: int = 24,
    title: str = "Daily Water Level Statistics",
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """
    Plot daily maximum, minimum, and mean water levels.
    
    Args:
        water_levels: Array of water volumes
        max_volume: Maximum allowed water volume
        hours_per_day: Number of hours per day (default 24)
        title: Plot title
        figsize: Figure size as (width, height)
    """
    water_levels = np.array(water_levels)
    
    # Handle incomplete last day
    remainder = len(water_levels) % hours_per_day
    if remainder != 0:
        water_levels = water_levels[:-remainder]
    
    reshaped = water_levels.reshape(-1, hours_per_day)
    
    daily_max = reshaped.max(axis=1)
    daily_min = reshaped.min(axis=1)
    daily_mean = reshaped.mean(axis=1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    days = np.arange(len(daily_max))
    ax.fill_between(days, daily_min, daily_max, alpha=0.3, color='#A23B72', 
                     label='Daily Range (Min-Max)')
    ax.plot(days, daily_max, linewidth=2, marker='o', markersize=4, 
            color='#D62828', label='Daily Max')
    ax.plot(days, daily_mean, linewidth=2, linestyle='--', marker='s', 
            markersize=4, color='#F77F00', label='Daily Mean')
    ax.plot(days, daily_min, linewidth=2, marker='^', markersize=4, 
            color='#06A77D', label='Daily Min')
    
    ax.axhline(max_volume, color='red', linestyle=':', linewidth=2, 
               alpha=0.5, label='Max Capacity')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel("Water Volume (m³)", fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_reward_distribution(
    rewards: List[float],
    bins: int = 30,
    title: str = "Reward Distribution",
    figsize: Tuple[int, int] = (12, 5)
) -> None:
    """
    Plot histogram and KDE of reward distribution.
    
    Args:
        rewards: List of rewards per timestep
        bins: Number of histogram bins
        title: Plot title
        figsize: Figure size as (width, height)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    ax1.hist(rewards, bins=bins, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(rewards):.2f}')
    ax1.axvline(np.median(rewards), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(rewards):.2f}')
    ax1.set_title("Reward Distribution", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Reward Value", fontsize=10)
    ax1.set_ylabel("Frequency", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Box plot
    ax2.boxplot(rewards, vert=True, patch_artist=True,
                boxprops=dict(facecolor='#2E86AB', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))
    ax2.set_title("Reward Statistics", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Reward Value", fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add text with statistics
    stats_text = f"Mean: {np.mean(rewards):.2f}\nStd: {np.std(rewards):.2f}\n"
    stats_text += f"Min: {np.min(rewards):.2f}\nMax: {np.max(rewards):.2f}"
    ax2.text(1.25, np.median(rewards), stats_text, fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def plot_action_distribution(
    action_history: List[int],
    title: str = "Action Distribution",
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot bar chart of action frequencies.
    
    Args:
        action_history: List of actions taken
        title: Plot title
        figsize: Figure size as (width, height)
    """
    unique_actions = sorted(set(action_history))
    action_counts = [action_history.count(a) for a in unique_actions]
    action_percentages = [count / len(action_history) * 100 for count in action_counts]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_actions)))
    bars = ax.bar(unique_actions, action_counts, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, count, pct in zip(bars, action_counts, action_percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Action Index", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_xticks(unique_actions)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def plot_state_heatmap(agent, dim_x, dim_y):
    """
    dim_x, dim_y: strings, één van:
        "Volume", "Price", "Hour", "Week", "Month"
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # 1. Mapping van naam → index in state_visits
    dim_map = {
        "Volume": 0,
        "Price": 1,
        "Hour": 2,
        "Week": 3,
        "Month": 4
    }

    # 2. Bins per dimensie
    bins_map = {
        "Volume": agent.volume_bins[:-1],
        "Price": agent.price_bins[:-1],
        "Hour": agent.hour_bins[:-1],
        "Week": agent.week_bins[:-1],
        "Month": agent.month_bins[:-1]
    }

    ix = dim_map[dim_x]
    iy = dim_map[dim_y]

    # 3. Alle dimensies behalve x en y weg-summen
    all_dims = {0, 1, 2, 3, 4}
    axes_to_sum = tuple(all_dims - {ix, iy})

    heatmap = agent.state_visits.sum(axis=axes_to_sum)

    # 4. Plotten met correcte kleuren
    plt.figure(figsize=(7, 5))

    im = plt.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        cmap="YlOrRd"  # 🔥 goede heatmap-kleuren
    )

    plt.colorbar(im, label="Visits")
    plt.title(f"{dim_x} × {dim_y} visits")
    plt.xlabel(f"{dim_x} bin")
    plt.ylabel(f"{dim_y} bin")

    # 5. Labels veilig formatteren (±inf)
    def format_bin(b):
        if np.isneginf(b):
            return "-∞"
        if np.isposinf(b):
            return "∞"
        return str(int(b))

    xbins = bins_map[dim_x]
    ybins = bins_map[dim_y]

    plt.xticks(
        ticks=np.arange(len(xbins)),
        labels=[format_bin(b) for b in xbins]
    )
    plt.yticks(
        ticks=np.arange(len(ybins)),
        labels=[format_bin(b) for b in ybins]
    )

    # 6. Getallen in de heatmap (kleur afhankelijk van intensiteit)
    max_val = heatmap.max() if heatmap.max() > 0 else 1

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            value = int(heatmap[i, j])
            text_color = "white" if value > 0.6 * max_val else "black"

            plt.text(
                j, i, str(value),
                ha='center', va='center',
                fontsize=8,
                color=text_color
            )

    plt.tight_layout()
    plt.show()

def plot_learning_curve(
    episode_rewards: List[float],
    window_size: int = 10,
    title: str = "Learning Curve - Training Progress",
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """
    Plot learning curve showing episode rewards with rolling average.
    
    Args:
        episode_rewards: List of total rewards per episode
        window_size: Size of rolling average window (default: 10)
        title: Plot title
        figsize: Figure size as (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    episodes = np.arange(len(episode_rewards))
    
    # Plot raw episode rewards with transparency
    ax.plot(episodes, episode_rewards, linewidth=1, alpha=0.3, 
           color='#2E86AB', label='Per-Episode Reward')
    ax.scatter(episodes, episode_rewards, s=10, alpha=0.2, color='#2E86AB')
    
    # Plot rolling average
    rolling_avg = pd.Series(episode_rewards).rolling(window=window_size, center=True).mean()
    ax.plot(episodes, rolling_avg, linewidth=2.5, color='#F77F00', 
           label=f'Rolling Average (window={window_size})')
    
    # Fill between rolling avg and zero for visualization
    ax.fill_between(episodes, rolling_avg, alpha=0.2, color='#F77F00')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Total Episode Reward", fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_learning_curve_with_phases(
    episode_rewards: List[float],
    window_size: int = 10,
    title: str = "Learning Progress - Training Phases",
    figsize: Tuple[int, int] = (16, 6)
) -> None:
    """
    Plot learning curve with multiple metrics tracking training progress.
    
    Args:
        episode_rewards: List of total rewards per episode
        window_size: Size of rolling average window (default: 10)
        title: Plot title
        figsize: Figure size as (width, height)
    """
    episode_rewards = np.array(episode_rewards)
    episodes = np.arange(len(episode_rewards))
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Raw rewards with rolling average
    ax = axes[0, 0]
    ax.plot(episodes, episode_rewards, linewidth=1, alpha=0.3, 
           color='#2E86AB', label='Per-Episode')
    rolling_avg = pd.Series(episode_rewards).rolling(window=window_size, center=True).mean()
    ax.plot(episodes, rolling_avg, linewidth=2.5, color='#F77F00', 
           label=f'Rolling Avg (window={window_size})')
    ax.fill_between(episodes, rolling_avg, alpha=0.2, color='#F77F00')
    ax.set_title("Episode Rewards", fontsize=12, fontweight='bold')
    ax.set_ylabel("Reward", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2. Cumulative learning progress
    ax = axes[0, 1]
    cumulative_rewards = np.cumsum(episode_rewards)
    ax.plot(episodes, cumulative_rewards, linewidth=2.5, color='#06A77D')
    ax.fill_between(episodes, cumulative_rewards, alpha=0.2, color='#06A77D')
    ax.set_title("Cumulative Rewards", fontsize=12, fontweight='bold')
    ax.set_ylabel("Total Cumulative Reward", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. Learning rate (improvement per episode)
    ax = axes[1, 0]
    improvement = np.diff(episode_rewards, prepend=episode_rewards[0])
    ax.bar(episodes, improvement, color=['#06A77D' if x > 0 else '#D62828' 
                                         for x in improvement], 
          alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title("Episode-to-Episode Improvement", fontsize=12, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Reward Change", fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Learning metrics summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate metrics
    total_episodes = len(episode_rewards)
    best_episode = np.argmax(episode_rewards)
    best_reward = episode_rewards[best_episode]
    avg_reward = np.mean(episode_rewards)
    final_avg = np.mean(episode_rewards[-window_size:])
    improvement_pct = ((final_avg - avg_reward) / abs(avg_reward) * 100) if avg_reward != 0 else 0
    
    metrics_text = f"""
LEARNING PROGRESS SUMMARY

Episodes:
  Total: {total_episodes}
  Best Episode: #{best_episode + 1}

Rewards:
  Best: {best_reward:.2f}
  Average: {avg_reward:.2f}
  Final Avg: {final_avg:.2f}
  Improvement: {improvement_pct:+.1f}%

Training Status:
  {"✓ Learning" if final_avg > avg_reward else "✗ Struggling"}
  {"→ Converging" if np.std(episode_rewards[-window_size:]) < np.std(episode_rewards[:window_size]) else "→ Exploring"}
    """
    
    ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def create_performance_dashboard(
    water_levels: np.ndarray,
    rewards: List[float],
    action_history: List[int],
    max_volume: float,
    agent = None,
    figsize: Tuple[int, int] = (18, 14)
) -> None:
    """
    Create a comprehensive multi-panel dashboard of model performance.
    
    Args:
        water_levels: Array of water volumes
        rewards: List of rewards per timestep
        action_history: List of actions taken
        max_volume: Maximum allowed water volume
        agent: QAgent instance (optional, for state heatmap)
        figsize: Figure size as (width, height)
    """
    water_levels = np.array(water_levels)
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Water levels over time
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(water_levels, linewidth=2, color='#06A77D')
    ax1.axhline(max_volume, color='#D62828', linestyle='--', linewidth=2, alpha=0.7)
    ax1.fill_between(range(len(water_levels)), 0, water_levels, alpha=0.2, color='#06A77D')
    ax1.set_title("Water Volume Over Time", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Volume (m³)", fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Action distribution
    ax2 = fig.add_subplot(gs[0, 2])
    unique_actions = sorted(set(action_history))
    action_counts = [action_history.count(a) for a in unique_actions]
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_actions)))
    ax2.bar(unique_actions, action_counts, color=colors, edgecolor='black')
    ax2.set_title("Action Distribution", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_xlabel("Action", fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Cumulative rewards
    ax3 = fig.add_subplot(gs[1, :2])
    cumulative = np.cumsum(rewards)
    ax3.plot(cumulative, linewidth=2.5, color='#F77F00')
    ax3.fill_between(range(len(cumulative)), 0, cumulative, alpha=0.2, color='#F77F00')
    ax3.set_title("Cumulative Reward", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Cumulative Reward", fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Reward distribution
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(rewards, bins=20, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label='Mean')
    ax4.set_title("Reward Distribution", fontsize=12, fontweight='bold')
    ax4.set_ylabel("Frequency", fontsize=10)
    ax4.set_xlabel("Reward", fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Daily statistics
    hours_per_day = 24
    remainder = len(water_levels) % hours_per_day
    if remainder != 0:
        wl_clean = water_levels[:-remainder]
    else:
        wl_clean = water_levels
    
    reshaped = wl_clean.reshape(-1, hours_per_day)
    daily_max = reshaped.max(axis=1)
    daily_min = reshaped.min(axis=1)
    daily_mean = reshaped.mean(axis=1)
    
    ax5 = fig.add_subplot(gs[2, :2])
    days = np.arange(len(daily_max))
    ax5.fill_between(days, daily_min, daily_max, alpha=0.3, color='#A23B72', label='Range')
    ax5.plot(days, daily_mean, linewidth=2, marker='o', markersize=4, color='#F77F00', label='Mean')
    ax5.axhline(max_volume, color='red', linestyle=':', linewidth=2, alpha=0.5, label='Max')
    ax5.set_title("Daily Water Level Statistics", fontsize=12, fontweight='bold')
    ax5.set_xlabel("Day", fontsize=10)
    ax5.set_ylabel("Volume (m³)", fontsize=10)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics text
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    stats_text = f"""
PERFORMANCE SUMMARY

Water Level:
  Mean: {water_levels.mean():.2f} m³
  Min: {water_levels.min():.2f} m³
  Max: {water_levels.max():.2f} m³
  Std: {water_levels.std():.2f} m³

Rewards:
  Total: {np.sum(rewards):.2f}
  Mean: {np.mean(rewards):.2f}
  Min: {np.min(rewards):.2f}
  Max: {np.max(rewards):.2f}

Actions:
  Total: {len(action_history)}
  Unique: {len(set(action_history))}
    """
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle("Model Performance Dashboard", 
                fontsize=16, fontweight='bold', y=0.995)
    plt.show()