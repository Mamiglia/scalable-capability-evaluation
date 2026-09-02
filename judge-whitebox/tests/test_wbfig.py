import os
import pathlib
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wbfig
import wbconfig


def test_rcparams_font_size():
    """Font size set to 8."""
    assert matplotlib.rcParams["font.size"] == 8


def test_rcparams_dpi():
    """Savefig DPI set to 400."""
    assert matplotlib.rcParams["savefig.dpi"] == 400


def test_save_creates_png_and_svg(monkeypatch):
    """save() creates both .png and .svg files in out/figures/."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Monkeypatch cfg to return our temp directory
        def mock_cfg(path=None):
            return {"paths": {"out": tmpdir}}

        monkeypatch.setattr(wbfig.wbconfig, "cfg", mock_cfg)

        fig = wbfig.new_fig(3.5, 3.5)
        ax = fig.add_subplot(111)
        ax.plot([1, 2, 3], [1, 2, 3])

        wbfig.save(fig, "test_figure")

        figures_dir = pathlib.Path(tmpdir) / "figures"
        assert (figures_dir / "test_figure.png").exists()
        assert (figures_dir / "test_figure.svg").exists()


def test_new_fig_returns_figure_with_constrained_layout():
    """new_fig() returns constrained-layout figure."""
    fig = wbfig.new_fig(3.5, 3.5)
    assert fig.get_layout_engine() is not None
    assert fig.get_size_inches().tolist() == [3.5, 3.5]
    plt.close(fig)
