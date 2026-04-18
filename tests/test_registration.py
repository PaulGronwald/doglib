import matplotlib
import matplotlib.pyplot as plt


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
