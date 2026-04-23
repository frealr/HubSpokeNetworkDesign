import sys
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def _out_path(mat_path, suffix):
    stem = os.path.splitext(os.path.basename(mat_path))[0]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots_blo')
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, stem + suffix)


# ── Airport coordinates (IATA → (name, lat, lon)) ────────────────────────────
AIRPORTS = {
    'MAD': ('Madrid',    40.4719, -3.5626),
    'BCN': ('Barcelona', 41.2971,  2.0785),
    'PMI': ('Palma',     39.5517,  2.7388),
    'AGP': ('Málaga',    36.6749, -4.4991),
}
NODE_ORDER = ['MAD', 'BCN', 'PMI', 'AGP']   # nodes 1..4

PROJ = ccrs.PlateCarree()

# Line-style cycle to distinguish OD pairs beyond color
_LSTYLES = ['-', '--', '-.', ':']


def _add_spain_features(ax):
    ax.add_feature(cfeature.OCEAN.with_scale('10m'),   color='#c8e6f5', zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('10m'),    color='#f5f0e8', zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.6,
                   edgecolor='#555555', zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6,
                   edgecolor='#555555', zorder=2)


def _great_circle_pts(lon1, lat1, lon2, lat2, n_pts=120):
    """Return (lons, lats) arrays tracing the great-circle arc via SLERP."""
    r1 = np.radians([lat1, lon1])
    r2 = np.radians([lat2, lon2])

    # Cartesian unit vectors
    def to_xyz(la, lo):
        return np.array([np.cos(la)*np.cos(lo),
                         np.cos(la)*np.sin(lo),
                         np.sin(la)])

    v1 = to_xyz(*r1)
    v2 = to_xyz(*r2)
    omega = np.arccos(np.clip(np.dot(v1, v2), -1, 1))

    if omega < 1e-10:          # coincident points
        return np.array([lon1, lon2]), np.array([lat1, lat2])

    t = np.linspace(0, 1, n_pts)
    pts = (np.sin((1 - t) * omega)[:, None] * v1 +
           np.sin(t * omega)[:, None] * v2) / np.sin(omega)

    lats = np.degrees(np.arcsin(np.clip(pts[:, 2], -1, 1)))
    lons = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    return lons, lats


def plot_network_topology(mat_path, show=True):
    data = sio.loadmat(mat_path)
    s  = np.ravel(data['s'])
    sh = np.ravel(data['sh'])
    a  = data['a']
    n  = len(NODE_ORDER)

    is_hub   = sh > 1e-2
    is_spoke = (~is_hub) & (s > 1e-2)

    fig = plt.figure(figsize=(9, 8))

    # Main axes: mainland Spain + Balearics
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.90],
                      projection=ccrs.AlbersEqualArea(
                          central_longitude=-3.5, central_latitude=40.0,
                          standard_parallels=(36.0, 44.0)))
    ax.set_extent([-10.0, 5.5, 34.8, 44.5], crs=PROJ)
    _add_spain_features(ax)
    ax.gridlines(draw_labels=False, linewidth=0.3, color='gray',
                 alpha=0.4, linestyle='--')

    # Inset: Canary Islands – lower-left rectangle
    ax_can = fig.add_axes([0.05, 0.05, 0.25, 0.22],
                          projection=ccrs.AlbersEqualArea(
                              central_longitude=-15.5, central_latitude=28.1,
                              standard_parallels=(27.0, 29.5)))
    ax_can.set_extent([-18.5, -13.0, 27.4, 29.6], crs=PROJ)
    _add_spain_features(ax_can)
    for sp in ax_can.spines.values():
        sp.set_edgecolor('#333333')
        sp.set_linewidth(1.2)

    # ── Links: orthodromic arcs ───────────────────────────────────────────────
    for i in range(n):
        for j in range(i + 1, n):
            if a[i, j] > 1e-2 or a[j, i] > 1e-2:
                _, lat1, lon1 = AIRPORTS[NODE_ORDER[i]]
                _, lat2, lon2 = AIRPORTS[NODE_ORDER[j]]
                lons, lats = _great_circle_pts(lon1, lat1, lon2, lat2)
                ax.plot(lons, lats, transform=PROJ,
                        color='steelblue', linewidth=1.8, alpha=0.75, zorder=3)

    # ── Airports ─────────────────────────────────────────────────────────────
    for i, iata in enumerate(NODE_ORDER):
        _, lat, lon = AIRPORTS[iata]
        if is_hub[i]:
            color, marker, ms, lcol = 'red',       'o', 10, 'darkred'
        elif is_spoke[i]:
            color, marker, ms, lcol = 'steelblue', 'o',  8, 'navy'
        else:
            color, marker, ms, lcol = 'gray',      'x',  7, 'gray'

        ax.plot(lon, lat, marker=marker, color=color, markersize=ms,
                markeredgecolor='white', markeredgewidth=0.8,
                transform=PROJ, zorder=5)
        ax.text(lon + 0.15, lat + 0.15, iata, transform=PROJ,
                fontsize=9, fontweight='bold', color=lcol, zorder=6, clip_on=True)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mlines.Line2D([], [], color='red',       marker='o', ms=8,
                      linestyle='None', label='Hub'),
        mlines.Line2D([], [], color='steelblue', marker='o', ms=8,
                      linestyle='None', label='Spoke'),
        mlines.Line2D([], [], color='steelblue', linewidth=1.8,
                      alpha=0.75, label='Link'),
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              fontsize=9, framealpha=0.9)

    title = mat_path.split('/')[-1].replace('.mat', '')
    ax.set_title(f'Network topology – {title}', fontsize=10, pad=8)

    out_path = _out_path(mat_path, '_map.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    if show:
        plt.show()
    plt.close(fig)


def plot_sh_trajectory(mat_path, show=True):
    data = sio.loadmat(mat_path)
    s_traj  = data['s_traj']    # (niters, n)
    sh_traj = data['sh_traj']   # (niters, n)
    niters, n = sh_traj.shape
    iters = np.arange(1, niters + 1)

    colors  = plt.cm.tab10(np.linspace(0, 0.9, n))
    markers = ['o', 's', '^', 'D', 'v', 'P']

    fig, ax = plt.subplots(figsize=(8, 5))
    title = mat_path.split('/')[-1].replace('.mat', '')
    fig.suptitle(title, fontsize=9)

    for i in range(n):
        values = np.where(sh_traj[:, i] > 1e-2,
                          s_traj[:, i] + sh_traj[:, i],
                          sh_traj[:, i])
        label = NODE_ORDER[i] if i < len(NODE_ORDER) else f'Node {i+1}'
        ax.plot(iters, values,
                color=colors[i],
                linestyle=_LSTYLES[i % len(_LSTYLES)],
                marker=markers[i % len(markers)],
                markersize=4, markevery=max(1, niters // 15),
                linewidth=1.6, label=label)

    ax.set_xlabel('Outer iteration')
    ax.set_ylabel('s + sh  (hub active)  /  sh  (otherwise)')
    ax.set_title('Hub/station capacity trajectory')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    out_path = _out_path(mat_path, '_sh_traj.png')
    plt.savefig(out_path, dpi=150)
    print(f'Saved: {out_path}')
    if show:
        plt.show()
    plt.close(fig)


def plot_f_trajectory(mat_path, show=True):
    data = sio.loadmat(mat_path)
    f_traj = data['f_traj']     # (niters, n, n)
    niters, n, _ = f_traj.shape
    iters = np.arange(1, niters + 1)

    od_pairs = [(o, d) for o in range(n) for d in range(n) if o != d]
    n_pairs  = len(od_pairs)

    # Tab20 gives 20 visually distinct colors → no repeats for ≤12 pairs
    colors  = plt.cm.tab20(np.linspace(0, 1, n_pairs, endpoint=False))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', '*', '<', '>', 'p']

    fig, ax = plt.subplots(figsize=(10, 6))
    title = mat_path.split('/')[-1].replace('.mat', '')
    fig.suptitle(title, fontsize=9)

    for idx, (o, d) in enumerate(od_pairs):
        o_iata = NODE_ORDER[o] if o < len(NODE_ORDER) else str(o + 1)
        d_iata = NODE_ORDER[d] if d < len(NODE_ORDER) else str(d + 1)
        ax.plot(iters, f_traj[:, o, d],
                color=colors[idx],
                linestyle=_LSTYLES[(idx // 10) % len(_LSTYLES)],
                marker=markers[idx % len(markers)],
                markersize=4, markevery=max(1, niters // 15),
                linewidth=1.4, label=f'{o_iata}→{d_iata}')

    ax.set_xlabel('Outer iteration')
    ax.set_ylabel('f$^{od}$  (market share)')
    ax.set_title('Market share trajectory per OD pair')
    ax.legend(ncol=3, fontsize=8, loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    out_path = _out_path(mat_path, '_f_traj.png')
    plt.savefig(out_path, dpi=150)
    print(f'Saved: {out_path}')
    if show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        import glob
        candidates = glob.glob('./4node_hs_prueba_v0_blo/*python.mat')
        if not candidates:
            print('Usage: python plot_blo_traj.py <path_to_mat>')
            sys.exit(1)
        mat_path = candidates[0]
        print(f'No file given, using: {mat_path}')
    else:
        mat_path = sys.argv[1]

    plot_sh_trajectory(mat_path)
    plot_f_trajectory(mat_path)
    plot_network_topology(mat_path)
