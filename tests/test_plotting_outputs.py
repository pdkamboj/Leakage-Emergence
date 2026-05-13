from __future__ import annotations

from leakage_emergence.plotting import FIGURE_CAPTIONS, generate_all_figures


def test_generate_all_figures_writes_png_pdf_and_caption_file(tmp_path) -> None:
    paths = generate_all_figures(tmp_path)
    generated = {path.name for path in paths}

    for item in FIGURE_CAPTIONS:
        assert f"{item.stem}.png" in generated
        assert f"{item.stem}.pdf" in generated
        assert (tmp_path / f"{item.stem}.png").is_file()
        assert (tmp_path / f"{item.stem}.pdf").is_file()

    caption_path = tmp_path / "figure_captions.tex"
    assert "figure_captions.tex" in generated
    assert caption_path.is_file()
    caption_text = caption_path.read_text(encoding="utf-8")
    assert r"\includegraphics" in caption_text
    assert "fig:leakage-norms" in caption_text
    assert "finite-dimensional" in caption_text

