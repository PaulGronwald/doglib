import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def test_projection_registered():
    import triplot  # noqa: F401
    assert "tripartite" in matplotlib.projections.get_projection_names()


def test_subplot_kw_works():
    import triplot  # noqa: F401
    fig, ax = plt.subplots(subplot_kw={"projection": "tripartite"})
    from triplot.axes import TripartiteAxes
    assert isinstance(ax, TripartiteAxes)
    plt.close(fig)


def test_default_log_scales():
    import triplot  # noqa: F401
    fig, ax = plt.subplots(subplot_kw={"projection": "tripartite"})
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    plt.close(fig)


def test_aspect_defaults_to_auto():
    """Default aspect is 'auto' — interactive zoom feels natural, and the
    diagonals are rotated in pixel space so they still render with the
    correct slope at any aspect. Users who want pinned 45° diagonals opt
    in with aspect='equal'."""
    import triplot  # noqa: F401
    fig, ax = plt.subplots(subplot_kw={"projection": "tripartite"})
    assert ax.get_aspect() == "auto"
    plt.close(fig)


def test_aspect_equal_is_opt_in():
    """aspect='equal' still works when requested explicitly."""
    import triplot  # noqa: F401
    fig, ax = plt.subplots(
        subplot_kw={"projection": "tripartite", "aspect": "equal"},
    )
    assert ax.get_aspect() == 1.0
    plt.close(fig)


def test_triplot_plot_oneliner():
    """triplot.plot(freq, pv) returns (fig, ax) and draws the curve."""
    import numpy as np
    import triplot

    f = np.logspace(0, 3, 50)
    v = 10 / np.sqrt(1 + (f / 50) ** 2)
    fig, ax = triplot.plot(f, v)
    from triplot.axes import TripartiteAxes
    assert isinstance(ax, TripartiteAxes)
    assert len(ax.lines) == 1
    plt.close(fig)


def test_triplot_subplots_no_data():
    """triplot.subplots() makes a tripartite ax with no curve plotted yet."""
    import triplot
    fig, ax = triplot.subplots(units="SI", label_mode="midpoint")
    from triplot.axes import TripartiteAxes
    assert isinstance(ax, TripartiteAxes)
    assert ax._label_mode == "midpoint"
    assert len(ax.lines) == 0
    plt.close(fig)


def test_title_autopad_in_edge_mode():
    """In label_mode='edge', a title is pushed up past the top spine so it
    doesn't collide with disp labels living above the top edge.
    Measured in display pixels: the gap between the axes' top and the
    title's bottom should be clearly > the matplotlib default (~6pt)."""
    import triplot
    fig_e, ax_e = triplot.subplots(label_mode="edge")
    ax_e.set_title("test")
    fig_m, ax_m = triplot.subplots(label_mode="midpoint")
    ax_m.set_title("test")
    fig_e.canvas.draw()
    fig_m.canvas.draw()
    gap_edge = ax_e.title.get_window_extent().ymin - ax_e.bbox.ymax
    gap_mid = ax_m.title.get_window_extent().ymin - ax_m.bbox.ymax
    # Edge-mode padding is 20 pt, default ~6 pt — expect >10 pt extra.
    assert gap_edge - gap_mid > 10
    plt.close(fig_e)
    plt.close(fig_m)


def test_title_explicit_pad_respected():
    """If the user passes pad explicitly, we don't override."""
    import triplot
    fig, ax = triplot.subplots(label_mode="edge")
    ax.set_title("test", pad=2)
    fig.canvas.draw()
    gap = ax.title.get_window_extent().ymin - ax.bbox.ymax
    # pad=2 is tighter than the default ~6 — should be a small positive gap.
    assert gap < 8
    plt.close(fig)


def test_legend_default_upper_left_in_edge_mode():
    """Edge-label clutter makes 'best' pick bad corners; default to upper left."""
    import numpy as np
    import triplot
    fig, ax = triplot.subplots(label_mode="edge")
    ax.plot([1, 2, 3], [1, 2, 3], label="a")
    leg = ax.legend()
    # legend._loc is an int; 2 corresponds to 'upper left' in matplotlib.
    from matplotlib.legend import Legend
    assert leg._get_loc() == Legend.codes["upper left"]
    plt.close(fig)


def test_legend_explicit_loc_respected():
    """If the user passes loc=, we don't override it."""
    import triplot
    fig, ax = triplot.subplots(label_mode="edge")
    ax.plot([1, 2, 3], [1, 2, 3], label="a")
    leg = ax.legend(loc="lower right")
    from matplotlib.legend import Legend
    assert leg._get_loc() == Legend.codes["lower right"]
    plt.close(fig)


def test_aspect_auto_allows_squishing():
    """aspect='auto' lets the axes stretch independently so the plot can be
    squished into a wide or tall figure without forcing a square box."""
    import numpy as np
    import triplot  # noqa: F401

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(
        1, 1, 1, projection="tripartite", aspect="auto",
    )
    assert ax.get_aspect() == "auto"
    # A draw must still succeed — labels compute rotation in display space
    # so they follow the actual on-screen slope regardless of aspect.
    freq = np.logspace(0, 3, 50)
    pv = 10 / (1 + (freq / 50) ** 2) ** 0.5
    ax.plot(freq, pv)
    ax.set_xlim(1, 1000)
    ax.set_ylim(0.01, 100)
    fig.canvas.draw()
    plt.close(fig)
