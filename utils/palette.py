"""THE TWO COLOURS USED BY THE WHOLE DASHBOARD  (edit only this file + .streamlit/config.toml).

BG      -> soft sage-green background for the whole page, sidebar and header
ACCENT  -> soft light-blue surface colour for cards, buttons and panels
INK     -> plain dark text colour that stays readable on both colours.
           If your ACCENT is dark, change INK_ON_ACCENT to a light colour such as "#FFFFFF".

PLACEHOLDER VALUES: the website could not be opened while this was built, so replace
BG and ACCENT with the exact hex codes from the site (right-click the section ->
Inspect -> background-color).
"""

BG = "#DCE7E1"          # soft sage-green page background for comfortable viewing
ACCENT = "#D6E6EF"      # soft light-blue surfaces for cards and controls
INK = "#20353C"         # deep blue-grey for readable text
INK_ON_ACCENT = "#20353C"   # dark text on the light sage surface


def css_vars() -> str:
    return (":root{"
            f"--ust-bg:{BG};--ust-accent:{ACCENT};--ust-ink:{INK};--ust-on-accent:{INK_ON_ACCENT};"
            "}")


def _make_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("ust_two_colour", [BG, ACCENT, INK])


CMAP = _make_cmap()
