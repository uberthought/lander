import os
import sys

_plotter = None
_disabled = False


def _make_plotter():
    import matplotlib.pyplot as plt
    from offline_training import WORLD_DIM_LABELS

    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    fig.canvas.manager.set_window_title('lander training')

    ax_overall, ax_per_dim, ax_actor = axes

    ax_overall.set_title('World SNR (overall)')
    ax_overall.set_ylabel('dB')
    overall_line, = ax_overall.plot([], [], label='World SNR')
    ax_overall.legend(loc='lower right')

    ax_per_dim.set_title('World per-dim SNR')
    ax_per_dim.set_ylabel('dB')
    per_dim_lines = {
        label: ax_per_dim.plot([], [], label=label)[0]
        for label in WORLD_DIM_LABELS
    }
    ax_per_dim.legend(loc='lower right', ncol=4, fontsize=7)

    ax_actor.set_title('Actor-Q')
    ax_actor.set_xlabel('iteration')
    pred_line, = ax_actor.plot([], [], label='pred')
    tgt_line, = ax_actor.plot([], [], label='tgt')
    ax_actor.legend(loc='lower right')

    fig.tight_layout()

    return {
        'fig': fig,
        'axes': axes,
        'overall_line': overall_line,
        'per_dim_lines': per_dim_lines,
        'pred_line': pred_line,
        'tgt_line': tgt_line,
        'history': {
            'x': [],
            'overall': [],
            'per_dim': {label: [] for label in WORLD_DIM_LABELS},
            'pred': [],
            'tgt': [],
        },
        'i': 0,
    }


def record(world_snr, per_dim, actor_pred, actor_tgt):
    global _plotter, _disabled
    if _disabled:
        return
    if os.environ.get('LANDER_NO_PLOT'):
        _disabled = True
        return
    try:
        if _plotter is None:
            _plotter = _make_plotter()

        p = _plotter
        p['i'] += 1
        h = p['history']
        h['x'].append(p['i'])
        h['overall'].append(world_snr)
        for label, snr in per_dim:
            h['per_dim'][label].append(snr)
        h['pred'].append(actor_pred)
        h['tgt'].append(actor_tgt)

        p['overall_line'].set_data(h['x'], h['overall'])
        for label, line in p['per_dim_lines'].items():
            line.set_data(h['x'], h['per_dim'][label])
        p['pred_line'].set_data(h['x'], h['pred'])
        p['tgt_line'].set_data(h['x'], h['tgt'])

        for ax in p['axes']:
            ax.relim()
            ax.autoscale_view()

        p['fig'].canvas.draw_idle()
        p['fig'].canvas.flush_events()
    except Exception as e:
        print(f"[live_plot] disabling (error: {e})", file=sys.stderr)
        _disabled = True
