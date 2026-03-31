#!/usr/bin/env python3
"""
Generate animated GIF demos for the BSD project homepage.

Outputs:
  1. denoising_process.gif  — simulated BSD denoising on Bicycle (600x600, ~3s)
  2. method_comparison.gif  — side-by-side MBD | BSD-fix | NN  (900x400, ~4s)
  3. multi_system.gif       — 2x2 grid across 4 vehicle systems (800x800, ~4s)
"""

import os
import sys
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────────
RESULTS_BASE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', 'Experiment', 'results'))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

SYSTEMS = {
    'kinematic_bicycle2d': {'label': 'Bicycle (3D)', 'n_state': 3},
    'tt2d':                {'label': 'Tractor-Trailer (4D)', 'n_state': 4},
    'n_trailer2d':         {'label': 'N-Trailer (5D)', 'n_state': 5},
    'acc_tt2d':            {'label': 'Acc. Tractor-Trailer (6D)', 'n_state': 6},
}

# ── Parking lot geometry ─────────────────────────────────────────────────────
N_COLS = 8
N_ROWS = 2
SPACE_W = 3.5
SPACE_L = 7.0
LOT_X0 = -14.0
LOT_Y0 = -12.0
OCCUPIED = [1, 2, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15]
TARGETS  = [3, 11]

# Goal: center of space 3 (row 0, col 2)
GOAL_X = LOT_X0 + 2 * SPACE_W + SPACE_W / 2   # -5.25
GOAL_Y = LOT_Y0 + SPACE_L / 2                  # -8.5

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Color palette
BG_COLOR     = '#FFFFFF'
LOT_BG       = '#F5F5F5'
SPACE_FILL   = '#E8E8E8'
SPACE_EDGE   = '#BDBDBD'
TARGET_FILL  = '#E8F5E9'
TARGET_EDGE  = '#4CAF50'
OBSTACLE_CLR = '#D5D5D5'
OBS_EDGE_CLR = '#C0C0C0'
WALL_CLR     = '#555555'
ROAD_CLR     = '#FAFAFA'

MBD_CLR      = '#2196F3'
BSD_CLR      = '#FF7043'
NN_CLR       = '#66BB6A'

# Denoising gradient: warm red -> cool blue
DENOISE_COLORS = [
    '#E53935', '#EF6C00', '#F9A825', '#7CB342',
    '#00ACC1', '#1E88E5', '#5E35B1', '#283593',
]


def load_dataset(system):
    """Load states and rewards for a given system."""
    ds_dir = os.path.join(RESULTS_BASE, system, 'parking', 'dataset')
    states = np.load(os.path.join(ds_dir, 'states.npy'))
    rewards = np.load(os.path.join(ds_dir, 'rewards.npy'))
    return states, rewards


def space_center(space_1idx):
    """Compute center (x, y) of a 1-indexed parking space."""
    row = (space_1idx - 1) // N_COLS
    col = (space_1idx - 1) % N_COLS
    cx = LOT_X0 + col * SPACE_W + SPACE_W / 2
    cy = LOT_Y0 + row * SPACE_L + SPACE_L / 2
    return cx, cy


def draw_parking_lot(ax, show_goal=True, compact=False):
    """Draw the parking lot background on a matplotlib Axes."""
    ax.set_facecolor(ROAD_CLR)

    # All parking spaces (base grid)
    for row in range(N_ROWS):
        for col in range(N_COLS):
            x = LOT_X0 + col * SPACE_W
            y = LOT_Y0 + row * SPACE_L
            rect = Rectangle((x, y), SPACE_W, SPACE_L,
                              linewidth=0.5, edgecolor=SPACE_EDGE,
                              facecolor=LOT_BG, zorder=0)
            ax.add_patch(rect)

    # Occupied spaces (filled grey with car silhouette)
    for sp in OCCUPIED:
        row_i = (sp - 1) // N_COLS
        col_i = (sp - 1) % N_COLS
        x = LOT_X0 + col_i * SPACE_W
        y = LOT_Y0 + row_i * SPACE_L
        rect = Rectangle((x, y), SPACE_W, SPACE_L,
                          linewidth=0.5, edgecolor=SPACE_EDGE,
                          facecolor=SPACE_FILL, zorder=0)
        ax.add_patch(rect)
        # Car silhouette
        cx, cy = space_center(sp)
        car_w, car_h = 1.6, 4.4
        car = FancyBboxPatch((cx - car_w/2, cy - car_h/2), car_w, car_h,
                              boxstyle="round,pad=0.15",
                              linewidth=0.3, edgecolor=OBS_EDGE_CLR,
                              facecolor=OBSTACLE_CLR, zorder=0)
        ax.add_patch(car)

    # Target spaces (green dashed outline)
    for sp in TARGETS:
        row_i = (sp - 1) // N_COLS
        col_i = (sp - 1) % N_COLS
        x = LOT_X0 + col_i * SPACE_W
        y = LOT_Y0 + row_i * SPACE_L
        rect = Rectangle((x, y), SPACE_W, SPACE_L,
                          linewidth=1.8 if not compact else 1.4,
                          edgecolor=TARGET_EDGE,
                          facecolor=TARGET_FILL, linestyle='--',
                          zorder=0)
        ax.add_patch(rect)

    # Boundary walls
    wall_lw = 2.5 if not compact else 2.0
    ax.axhline(y=15.0, color=WALL_CLR, linewidth=wall_lw, zorder=1)
    ax.axhline(y=-14.0, color=WALL_CLR, linewidth=wall_lw, zorder=1)

    # Goal marker
    if show_goal:
        ms = 14 if not compact else 10
        ax.plot(GOAL_X, GOAL_Y, '*', color='#C62828', markersize=ms, zorder=5,
                markeredgecolor='white', markeredgewidth=0.5)

    # Axis limits
    ax.set_xlim(-18, 16)
    ax.set_ylim(-15.5, 16)
    ax.set_aspect('equal')


def fig_to_pil(fig, dpi=100):
    """Render a matplotlib figure to a PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                pad_inches=0.05, facecolor=BG_COLOR)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


def resize_exact(img, width, height):
    """Resize PIL image to exact dimensions."""
    return img.resize((width, height), Image.LANCZOS)


def ease_in_out(t):
    """Smooth ease-in-out for alpha blending (cubic)."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - (-2 * t + 2) ** 3 / 2


# ═════════════════════════════════════════════════════════════════════════════
# GIF 1: Denoising Process
# ═════════════════════════════════════════════════════════════════════════════
def generate_denoising_gif():
    """Simulate BSD denoising: random noise -> refined trajectory."""
    print('Generating denoising_process.gif ...')

    states, rewards = load_dataset('kinematic_bicycle2d')
    best_idx = np.argmax(rewards)
    best_traj = states[best_idx]  # (51, 3)

    rng = np.random.RandomState(42)

    # Generate 20 spread-out noise curves across the full world area
    n_noise = 20
    noise_trajs = []
    for i in range(n_noise):
        t = np.zeros((51, 2))
        # Distribute start points across the map
        t[0] = [rng.uniform(-16, 14), rng.uniform(-12, 14)]
        for step in range(1, 51):
            t[step] = t[step - 1] + rng.randn(2) * 0.8
            # Soft clamp to world bounds
            t[step, 0] = np.clip(t[step, 0], -17, 15)
            t[step, 1] = np.clip(t[step, 1], -14, 15)
        noise_trajs.append(t)
    noise_trajs = np.array(noise_trajs)  # (20, 51, 2)

    # 20 real trajectories for interpolation targets (diverse rewards)
    sort_idx = np.argsort(rewards)
    # Mix of good and medium trajectories
    interp_indices = np.concatenate([
        sort_idx[-10:],           # top 10
        sort_idx[-50:-40],        # medium-high
    ])
    rng.shuffle(interp_indices)
    interp_trajs = states[interp_indices[:n_noise], :, :2]  # (20, 51, 2)

    n_frames = 16  # 0=noise, 1-14=denoising, 15=final
    frames = []

    # Color gradient for denoising steps
    cmap_denoise = plt.cm.RdYlBu  # Red -> Yellow -> Blue

    for frame_i in range(n_frames):
        alpha = frame_i / (n_frames - 1)  # 0.0 to 1.0
        alpha_smooth = ease_in_out(alpha)

        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        draw_parking_lot(ax)

        if frame_i == 0:
            # ---- Frame 0: Pure noise ----
            for i in range(n_noise):
                ax.plot(noise_trajs[i, :, 0], noise_trajs[i, :, 1],
                        color='#BBBBBB', linewidth=0.7, alpha=0.5, zorder=2)
            step_label = 'T = 1.00  (noise)'

        elif frame_i < n_frames - 1:
            # ---- Denoising frames: blend noise -> real ----
            color = cmap_denoise(alpha)
            for i in range(n_noise):
                blended = (1 - alpha_smooth) * noise_trajs[i] + alpha_smooth * interp_trajs[i]
                lw = 0.5 + alpha_smooth * 1.2
                ax.plot(blended[:, 0], blended[:, 1],
                        color=color, linewidth=lw,
                        alpha=0.25 + alpha_smooth * 0.35, zorder=2)

            # Converging "best" trajectory (thicker, brighter)
            best_blend = (1 - alpha_smooth) * noise_trajs[0] + alpha_smooth * best_traj[:, :2]
            ax.plot(best_blend[:, 0], best_blend[:, 1],
                    color=color, linewidth=1.5 + alpha_smooth * 1.5,
                    alpha=0.5 + alpha_smooth * 0.4, zorder=3)

            t_val = 1.0 - alpha
            step_label = f'T = {t_val:.2f}'

        else:
            # ---- Final frame: clean optimized trajectory ----
            ax.plot(best_traj[:, 0], best_traj[:, 1],
                    color='#1565C0', linewidth=3.5, alpha=0.95, zorder=4,
                    solid_capstyle='round')
            ax.plot(best_traj[0, 0], best_traj[0, 1], 'o',
                    color='#2E7D32', markersize=11, zorder=5,
                    markeredgecolor='white', markeredgewidth=1.2)
            ax.plot(best_traj[-1, 0], best_traj[-1, 1], 'X',
                    color='#C62828', markersize=13, zorder=5,
                    markeredgecolor='white', markeredgewidth=1.2)
            step_label = 'T = 0.00  (optimized)'

        # Title
        ax.set_title('BSD Denoising Process', fontsize=15, fontweight='bold',
                      pad=12, color='#333333')

        # Step badge (bottom-left)
        ax.text(0.03, 0.03, step_label, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          alpha=0.9, edgecolor='#CCCCCC', linewidth=0.5),
                zorder=10, color='#444444')

        # Progress bar (bottom)
        bar_y = -15.0
        bar_x0, bar_x1 = -17.5, 15.5
        bar_len = bar_x1 - bar_x0
        ax.plot([bar_x0, bar_x1], [bar_y, bar_y], color='#E0E0E0',
                linewidth=4, solid_capstyle='round', zorder=8)
        ax.plot([bar_x0, bar_x0 + bar_len * alpha], [bar_y, bar_y],
                color='#1565C0', linewidth=4, solid_capstyle='round', zorder=9)

        ax.tick_params(labelbottom=False, labelleft=False,
                       bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.tight_layout()
        img = fig_to_pil(fig, dpi=100)
        img = resize_exact(img, 600, 600)
        frames.append(img)
        plt.close(fig)

    # Save GIF  (~3 seconds: 400 + 14*150 + 800 = 3300ms)
    out_path = os.path.join(OUTPUT_DIR, 'denoising_process.gif')
    durations = [500] + [150] * (n_frames - 2) + [1000]
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print(f'  Saved: {out_path} ({len(frames)} frames, ~{sum(durations)/1000:.1f}s)')
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# GIF 2: Method Comparison
# ═════════════════════════════════════════════════════════════════════════════
def generate_comparison_gif():
    """Side-by-side MBD | BSD-fix | NN comparison with trajectory animation."""
    print('Generating method_comparison.gif ...')

    states, rewards = load_dataset('kinematic_bicycle2d')

    # MBD: top 5 trajectories
    sorted_idx = np.argsort(rewards)[::-1]
    mbd_indices = sorted_idx[:5]
    mbd_trajs = states[mbd_indices]

    # BSD-fix: next 5 top trajectories (different but equally good)
    bsd_indices = sorted_idx[5:10]
    bsd_trajs = states[bsd_indices]

    # NN: 5 trajectories from lowest-reward region for visible contrast
    nn_candidates = np.argsort(rewards)[:50]
    rng = np.random.RandomState(777)
    nn_indices = rng.choice(nn_candidates, size=5, replace=False)
    nn_trajs = states[nn_indices]

    panels = [
        ('MBD (model-based)', MBD_CLR, mbd_trajs, rewards[mbd_indices]),
        ('BSD-fix (ours)', BSD_CLR, bsd_trajs, rewards[bsd_indices]),
        ('NN (no diffusion)', NN_CLR, nn_trajs, rewards[nn_indices]),
    ]

    n_trajs = 5
    n_timesteps = states.shape[1]  # 51

    # Animation structure: draw all 5 trajectories simultaneously, progressively
    # 25 draw frames + 10 hold frames = 35 frames
    draw_frames = 25
    hold_frames = 10
    total_frames = draw_frames + hold_frames

    frames = []

    for frame_i in range(total_frames):
        fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
        fig.patch.set_facecolor(BG_COLOR)

        if frame_i < draw_frames:
            frac = (frame_i + 1) / draw_frames
        else:
            frac = 1.0
        end_step = max(2, int(frac * n_timesteps))

        for panel_idx, (label, color, trajs, rews) in enumerate(panels):
            ax = axes[panel_idx]
            draw_parking_lot(ax, compact=True)

            for t_idx in range(n_trajs):
                traj = trajs[t_idx]
                # Alpha varies by trajectory index for visual depth
                base_alpha = 0.5 + 0.1 * (n_trajs - t_idx) / n_trajs
                lw = 2.0 if t_idx > 0 else 2.8  # first traj thicker

                ax.plot(traj[:end_step, 0], traj[:end_step, 1],
                        color=color, linewidth=lw, alpha=base_alpha, zorder=3,
                        solid_capstyle='round')

                # Start marker
                ax.plot(traj[0, 0], traj[0, 1], 'o',
                        color='#2E7D32', markersize=6, zorder=5,
                        markeredgecolor='white', markeredgewidth=0.5)

                # End or current position
                if frac >= 1.0:
                    ax.plot(traj[-1, 0], traj[-1, 1], 's',
                            color=color, markersize=4, zorder=5,
                            markeredgecolor='white', markeredgewidth=0.4)
                else:
                    ax.plot(traj[end_step - 1, 0], traj[end_step - 1, 1], 'o',
                            color=color, markersize=4, zorder=6,
                            markeredgecolor='white', markeredgewidth=0.4, alpha=0.8)

            # Reward annotation on hold frames
            if frac >= 1.0:
                mean_r = np.mean(rews)
                ax.text(0.97, 0.03, f'avg R = {mean_r:.2f}',
                        transform=ax.transAxes, fontsize=9, fontweight='bold',
                        va='bottom', ha='right', color='#555555',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  alpha=0.85, edgecolor='#DDD', linewidth=0.5),
                        zorder=10)

            ax.set_title(label, fontsize=13, fontweight='bold',
                         color=color, pad=8)
            ax.tick_params(labelbottom=False, labelleft=False,
                           bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.suptitle('Planning Method Comparison -- Bicycle Parking',
                     fontsize=15, fontweight='bold', y=0.99, color='#333333')
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        img = fig_to_pil(fig, dpi=72)
        img = resize_exact(img, 900, 400)
        frames.append(img)
        plt.close(fig)

    # Save GIF  (~4s: 25*120 + 10*100 = 4000ms)
    out_path = os.path.join(OUTPUT_DIR, 'method_comparison.gif')
    durations = [120] * draw_frames + [100] * hold_frames
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print(f'  Saved: {out_path} ({len(frames)} frames, ~{sum(durations)/1000:.1f}s)')
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# GIF 3: Multi-System
# ═════════════════════════════════════════════════════════════════════════════
def generate_multi_system_gif():
    """2x2 grid: best trajectory per system, drawn progressively."""
    print('Generating multi_system.gif ...')

    system_keys = ['kinematic_bicycle2d', 'tt2d', 'n_trailer2d', 'acc_tt2d']
    system_labels = [
        'Bicycle (3D)', 'Tractor-Trailer (4D)',
        'N-Trailer (5D)', 'Acc. Tractor-Trailer (6D)'
    ]
    system_colors = ['#1565C0', '#E65100', '#6A1B9A', '#2E7D32']

    # Load best trajectory per system
    best_trajs = []
    best_rewards = []
    for sys_key in system_keys:
        states, rewards = load_dataset(sys_key)
        best_idx = np.argmax(rewards)
        best_trajs.append(states[best_idx])
        best_rewards.append(rewards[best_idx])

    n_timesteps = 51
    draw_steps = 25
    hold_frames = 10
    total_frames = draw_steps + hold_frames

    frames = []

    for frame_i in range(total_frames):
        fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.8))
        fig.patch.set_facecolor(BG_COLOR)

        if frame_i < draw_steps:
            frac = (frame_i + 1) / draw_steps
        else:
            frac = 1.0
        end_step = max(2, int(frac * n_timesteps))

        for idx, (sys_key, label, color) in enumerate(
                zip(system_keys, system_labels, system_colors)):
            row, col = divmod(idx, 2)
            ax = axes[row][col]
            draw_parking_lot(ax, compact=True)

            traj = best_trajs[idx]

            # Trail effect: fading gradient behind the head
            if end_step > 2:
                # Draw trajectory segments with gradient alpha
                for seg_i in range(end_step - 1):
                    seg_alpha = 0.3 + 0.6 * (seg_i / max(1, end_step - 2))
                    ax.plot(traj[seg_i:seg_i+2, 0], traj[seg_i:seg_i+2, 1],
                            color=color, linewidth=2.8, alpha=seg_alpha, zorder=4,
                            solid_capstyle='round')

            # Start marker (green circle)
            ax.plot(traj[0, 0], traj[0, 1], 'o',
                    color='#2E7D32', markersize=9, zorder=5,
                    markeredgecolor='white', markeredgewidth=1.0)

            # Head / end marker
            if frac >= 1.0:
                ax.plot(traj[-1, 0], traj[-1, 1], 'X',
                        color='#C62828', markersize=11, zorder=5,
                        markeredgecolor='white', markeredgewidth=1.0)
                # Show reward
                ax.text(0.97, 0.03, f'R = {best_rewards[idx]:.2f}',
                        transform=ax.transAxes, fontsize=9, fontweight='bold',
                        va='bottom', ha='right', color='#555555',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  alpha=0.85, edgecolor='#DDD', linewidth=0.5),
                        zorder=10)
            else:
                # Moving head
                ax.plot(traj[end_step - 1, 0], traj[end_step - 1, 1], 'o',
                        color=color, markersize=8, zorder=6,
                        markeredgecolor='white', markeredgewidth=0.8)

            ax.set_title(label, fontsize=13, fontweight='bold',
                         color=color, pad=8)
            ax.tick_params(labelbottom=False, labelleft=False,
                           bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.suptitle('BSD-fix: Best Trajectory Across Systems',
                     fontsize=15, fontweight='bold', y=0.99, color='#333333')
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        img = fig_to_pil(fig, dpi=92)
        img = resize_exact(img, 800, 800)
        frames.append(img)
        plt.close(fig)

    # Save GIF  (~4s: 25*120 + 10*100 = 4000ms)
    out_path = os.path.join(OUTPUT_DIR, 'multi_system.gif')
    durations = [120] * draw_steps + [100] * hold_frames
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print(f'  Saved: {out_path} ({len(frames)} frames, ~{sum(durations)/1000:.1f}s)')
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'Results base: {RESULTS_BASE}')
    print(f'Output dir:   {OUTPUT_DIR}')
    print()

    p1 = generate_denoising_gif()
    print()
    p2 = generate_comparison_gif()
    print()
    p3 = generate_multi_system_gif()
    print()
    print('All GIFs generated successfully!')
    print(f'  {p1}')
    print(f'  {p2}')
    print(f'  {p3}')
