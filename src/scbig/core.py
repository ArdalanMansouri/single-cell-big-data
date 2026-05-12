from unidecode import unidecode
import re
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import numpy as np

def clean_variable_names(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    
    """ This function accepts a pandas DataFrame, removes any [] signs and the
    unnecessary prefix from column names, and returns the cleaned DataFrame
    along with a list of the cleaned column names for inspection.

    Args:
        df (pd.DataFrame): The input DataFrame with original column names.

    Returns:
        A tuple containing the cleaned DataFrame and a list of the cleaned 
        column names.
    """
    
    def _clean(name: str) -> str:
        # name = re.sub(r'[\[\]]', '', name) # if want to remove brackets
        name = unidecode(name.replace('Individual cells Selected - ', ''))
        name = name.strip()
        return name
    
    df.columns = [_clean(col) for col in df.columns]
    cleaned_columns = list(df.columns)
    
    return df, cleaned_columns

def scatter_plot_gen(
    df: pd.DataFrame,
    x_cols: list[str],
    y_col: str | list[str],
    mode: str = "x_vs_y",
    render_mode: str = "interactive",
    # plotting params
    s: int = 8,
    alpha: float = 0.4,
    legend_size: int = 18,
    x_label_size: int = 25,
    y_label_size: int = 25,
    tick_size: int = 18,
    draw_regression: bool = False,
    trendline_color: str = "red",
    width: int = 800,
    height: int = 600,
    spine_thickness: int = 2,
    # saving
    save_figs: bool = False,
    output_folder: str = ".",
    file_prefix: str = "scatter",
    library_name: str = "",
    dpi: int = 300,

) -> None:
    """Generate scatter plots for combinations of x and y columns using Plotly.
    Loops through every (x_col, y_col) pair implied by `mode` and renders or
    saves one figure per pair.

    Args:
        df: DataFrame containing the columns to plot.
        x_cols: List of column names to use as x-axes.
        y_col: Single column name (or list) to use as y-axis/axes.
            In "x_vs_y" mode a single string is expected.
        mode: How to generate (x, y) pairs.
            - "x_vs_y"  : each col in x_cols paired with the single y_col.
            - "pairwise": every ordered pair within x_cols (x != y).
            - "explicit": x_cols[i] paired with y_col[i] (both same length).
        render_mode: "interactive" — show full Plotly HTML widget (default);
                     "static"      — render a lightweight PNG-style static image
                                    (lower file size and faster render time).
        s: Marker size.
        alpha: Marker opacity (0–1).
        legend_size: Font size for the legend.
        x_label_size: Font size for the x-axis label.
        y_label_size: Font size for the y-axis label.
        tick_size: Font size for axis tick labels.
        draw_regression: If True, overlay a linear regression trend-line.
        trendline_color: Color of the regression trend-line (default "red").
            Accepts any valid CSS color name or hex code (e.g. "#1f77b4").
        width: Width of each figure in pixels (default 800).
        height: Height of each figure in pixels (default 600).
        spine_thickness: Line width of the axis border spines on all four 
            edges (default 2).
        save_figs: If True, save figures as PNG files; otherwise display them.
        output_folder: Directory to save figures (created if missing).
        file_prefix: Prefix for saved filenames.
        library_name: Optional extra prefix prepended to filenames.
        dpi: Resolution (DPI) for saved PNG files.
    """

    # ------ Build list of (x, y) pairs according to mode ------
    if mode == "x_vs_y":
        if not isinstance(y_col, str):
            raise ValueError("In 'x_vs_y' mode, y_col must be a single string.")
        pairs = [(x, y_col) for x in x_cols]

    elif mode == "pairwise":
        pairs = [(x, y) for x in x_cols for y in x_cols if x != y]

    elif mode == "explicit":
        y_cols = y_col if isinstance(y_col, list) else [y_col]
        if len(x_cols) != len(y_cols):
            raise ValueError(
                "In 'explicit' mode, x_cols and y_col must have the same length."
            )
        pairs = list(zip(x_cols, y_cols))

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'x_vs_y', \
                         'pairwise', or 'explicit'.")

    # ------ Prepare output folder ------
    if save_figs:
        os.makedirs(output_folder, exist_ok=True)

    # ------ Loop and plot ------
    for x_col, y_col_i in pairs:
        trendline = "ols" if draw_regression else None

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col_i,
            opacity=alpha,
            trendline=trendline,
            trendline_color_override=trendline_color if draw_regression else None,
            labels={x_col: x_col, y_col_i: y_col_i},
        )

        fig.update_traces(marker=dict(size=s))

        spine = dict(
            showline=True, 
            linecolor="black", linewidth=spine_thickness, 
            mirror=True
        )

        fig.update_layout(
            xaxis=dict(
                title_font=dict(size=x_label_size),
                tickfont=dict(size=tick_size),
                showgrid=False,
                zeroline=False,
                **spine,
            ),
            yaxis=dict(
                title_font=dict(size=y_label_size),
                tickfont=dict(size=tick_size),
                showgrid=False,
                zeroline=False,
                **spine,
            ),
            legend=dict(font=dict(size=legend_size)),
            width=width,
            height=height,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        if save_figs:
            parts = [
                p for p in [library_name, file_prefix,
                            x_col, "vs", y_col_i]
                            if p
            ]
            if render_mode == "static":
                filename = "_".join(parts) + ".png"
                filepath = os.path.join(output_folder, filename)
                img_bytes = fig.to_image(
                    format="png", width=width, height=height, scale=dpi / 96
                )
                with open(filepath, "wb") as f:
                    f.write(img_bytes)
            else:
                filename = "_".join(parts) + ".html"
                filepath = os.path.join(output_folder, filename)
                fig.write_html(filepath)
            print(f"Saved: {filepath}")
        else:
            if render_mode == "static":
                from IPython.display import Image, display as ipy_display
                img_bytes = fig.to_image(format="png", width=width, height=height)
                ipy_display(Image(img_bytes))
            else:
                fig.show()


def hist_plot_gen(
    df: pd.DataFrame,
    cols: list[str],
    render_mode: str = "interactive",
    # histogram params
    nbins: int = None,
    color: str = "#1f77b4",
    alpha: float = 0.8,
    # font / layout params
    legend_size: int = 18,
    x_label_size: int = 25,
    y_label_size: int = 25,
    tick_size: int = 18,
    width: int = 800,
    height: int = 600,
    spine_thickness: int = 2,
    discont_y_axis: list[float] | None = None,
    show_break_marks: bool = True,
    # saving
    save_figs: bool = False,
    output_folder: str = ".",
    file_prefix: str = "hist",
    library_name: str = "",
    dpi: int = 300,
) -> None:
    """Generate histograms for one or more columns using Plotly.
    Loops through each column in `cols` and renders or saves one figure per column.

    Args:
        df: DataFrame containing the columns to plot.
        cols: List of column names to plot as histograms.
        render_mode: "interactive" — show full Plotly HTML widget (default);
                     "static"      — render a lightweight PNG-style static image
                                    (lower file size and faster render time).
        nbins: Number of histogram bins. If None, Plotly chooses automatically.
        color: Fill color of the histogram bars (default "#1f77b4").
            Accepts any valid CSS color name or hex code.
        alpha: Opacity of the bars (0–1, default 0.8).
        legend_size: Font size for the legend.
        x_label_size: Font size for the x-axis label.
        y_label_size: Font size for the y-axis label.
        tick_size: Font size for axis tick labels.
        width: Width of each figure in pixels (default 800).
        height: Height of each figure in pixels (default 600).
        spine_thickness: Line width of the axis border spines on all four
            edges (default 2).
        discont_y_axis: Optional two-element list [break_bottom, break_top].
            When provided the y-axis is split: the bottom panel shows counts
            from 0 to break_bottom and the top panel from break_top to
            max_count. Use this when a few tall bars dwarf all others.
            Example: discont_y_axis=[500, 2000]
        show_break_marks: If True (default), draw diagonal slash marks at the
            y-axis break gap to indicate the discontinuity. Set to False to
            hide them.
        save_figs: If True, save figures (PNG for static, HTML for interactive);
            otherwise display them.
        output_folder: Directory to save figures (created if missing).
        file_prefix: Prefix for saved filenames.
        library_name: Optional extra prefix prepended to filenames.
        dpi: Resolution (DPI) for saved PNG files.
    """

    # ------ Prepare output folder ------
    if save_figs:
        os.makedirs(output_folder, exist_ok=True)

    spine = dict(showline=True, linecolor="black", linewidth=spine_thickness, mirror=True)

    # ------ Loop and plot ------
    for col in cols:
        # ------ Compute data ranges ------
        values = pd.to_numeric(df[col], errors="coerce").dropna().values
        col_min, col_max = float(values.min()), float(values.max())
        _bins = nbins if nbins is not None else "auto"
        counts, _ = np.histogram(values, bins=_bins)
        max_count = int(counts.max()) if counts.size else 1
        _nbinsx = nbins if nbins is not None else 0

        if discont_y_axis is not None:
            # ------ Broken-axis figure ------
            break_lo = float(discont_y_axis[0])
            break_hi = float(discont_y_axis[1])

            bottom_range = break_lo
            top_range = max_count - break_hi
            total = (bottom_range + top_range) or 1
            bottom_frac = bottom_range / total
            top_frac = top_range / total

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[top_frac, bottom_frac],
                vertical_spacing=0.03,
            )

            # add same histogram to both panels (bingroup keeps bins identical)
            for r in (1, 2):
                fig.add_trace(
                    go.Histogram(
                        x=df[col],
                        nbinsx=_nbinsx,
                        opacity=alpha,
                        marker_color=color,
                        bingroup=col,
                        showlegend=False,
                        name=col,
                    ),
                    row=r, col=1,
                )

            # generate round-number tick values within a range
            def _nice_ticks(lo: float, hi: float, n: int = 5,
                            include_lo: bool = False) -> list[int]:
                span = hi - lo
                if span <= 0:
                    return []
                raw_step = span / n
                mag = 10 ** np.floor(np.log10(raw_step))
                nice_step = int(max(round(raw_step / mag) * mag, 1))
                start = int(np.ceil(lo / nice_step) * nice_step) if include_lo \
                        else int(np.ceil((lo + 1) / nice_step) * nice_step)
                return list(range(start, int(hi), nice_step))

            def _fmt_tick(v: int) -> str:
                if v >= 10_000:
                    k = v / 1000
                    return f"{int(k)}K" if k == int(k) else f"{k:.1f}K"
                return str(v)

            bot_ticks = _nice_ticks(0, break_lo, include_lo=True)
            top_ticks = _nice_ticks(break_hi, max_count)
            bot_text  = [_fmt_tick(v) for v in bot_ticks]
            top_text  = [_fmt_tick(v) for v in top_ticks]

            # spines: bottom subplot → bottom / left / right (no top)
            #         top subplot    → top (via shape) / left / right (no bottom)
            fig.update_layout(
                # ------ bottom subplot (row 2) ------
                xaxis2=dict(
                    title_text=col,
                    title_font=dict(size=x_label_size),
                    tickfont=dict(size=tick_size),
                    range=[col_min, col_max],
                    showgrid=False,
                    zeroline=False,
                    showline=True,
                    linecolor="black",
                    linewidth=spine_thickness,
                    mirror=False,       # bottom edge only, no top edge
                ),
                yaxis2=dict(
                    title_text="",
                    tickfont=dict(size=tick_size),
                    range=[0, break_lo],
                    tickmode="array",
                    tickvals=bot_ticks,
                    ticktext=bot_text,
                    showgrid=False,
                    zeroline=False,
                    showline=True,
                    linecolor="black",
                    linewidth=spine_thickness,
                    mirror=True,        # left + right edges
                ),
                # ------ top subplot (row 1) ------
                xaxis=dict(
                    range=[col_min, col_max],
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    showline=False,     # no bottom edge on top subplot
                    mirror=False,
                ),
                yaxis=dict(
                    tickfont=dict(size=tick_size),
                    range=[break_hi, max_count],
                    tickmode="array",
                    tickvals=top_ticks,
                    ticktext=top_text,
                    showgrid=False,
                    zeroline=False,
                    showline=True,
                    linecolor="black",
                    linewidth=spine_thickness,
                    mirror=True,        # left + right edges
                ),
                legend=dict(font=dict(size=legend_size)),
                width=width,
                height=height,
                plot_bgcolor="white",
                paper_bgcolor="white",
                bargap=0.05,
            )

            # top border of top subplot (paper coords y=1 is the very top)
            fig.add_shape(
                type="line",
                xref="paper", yref="paper",
                x0=0, x1=1, y0=1, y1=1,
                line=dict(color="black", width=spine_thickness),
            )

            # single "Count" label centred across both subplots
            y_top_domain_top = fig.layout.yaxis.domain[1]    # top of top subplot
            y_bot_domain_bot = fig.layout.yaxis2.domain[0]   # bottom of bottom subplot
            y_label_center   = (y_top_domain_top + y_bot_domain_bot) / 2
            x_left_domain    = fig.layout.xaxis.domain[0]
            fig.add_annotation(
                x=x_left_domain,
                y=y_label_center,
                xref="paper", yref="paper",
                text="Count",
                showarrow=False,
                textangle=-90,
                font=dict(size=y_label_size),
                xanchor="right",
                yanchor="middle",
                xshift=-(y_label_size * 1.6),   # push left of the tick labels
            )

            # diagonal break marks: one slash centered on each break edge,
            # placed on both the left and right spines
            if show_break_marks:
                y_break_top = fig.layout.yaxis.domain[0]    # bottom edge of top subplot
                y_break_bot = fig.layout.yaxis2.domain[1]   # top edge of bottom subplot
                x_left  = fig.layout.xaxis.domain[0]        # left spine in paper coords
                x_right = fig.layout.xaxis.domain[1]        # right spine in paper coords
                mh, mw = 0.013, 0.018   # half-height and half-width of each slash
                for x_spine in (x_left, x_right):
                    for y_c in (y_break_top, y_break_bot):
                        fig.add_shape(
                            type="line",
                            xref="paper", yref="paper",
                            x0=x_spine - mw, y0=y_c - mh,
                            x1=x_spine + mw, y1=y_c + mh,
                            line=dict(color="black", width=spine_thickness),
                        )

        else:
            # ------ Normal single-panel figure ------
            fig = px.histogram(
                df,
                x=col,
                nbins=nbins,
                opacity=alpha,
                labels={col: col},
            )

            fig.update_traces(marker_color=color)

            fig.update_layout(
                xaxis=dict(
                    title_font=dict(size=x_label_size),
                    tickfont=dict(size=tick_size),
                    showgrid=False,
                    zeroline=False,
                    range=[col_min, col_max],
                    **spine,
                ),
                yaxis=dict(
                    title_text="Count",
                    title_font=dict(size=y_label_size),
                    tickfont=dict(size=tick_size),
                    showgrid=False,
                    zeroline=False,
                    range=[0, max_count],
                    **spine,
                ),
                legend=dict(font=dict(size=legend_size)),
                width=width,
                height=height,
                plot_bgcolor="white",
                paper_bgcolor="white",
                bargap=0.05,
            )

        if save_figs:
            parts = [p for p in [library_name, file_prefix, col] if p]
            if render_mode == "static":
                filename = "_".join(parts) + ".png"
                filepath = os.path.join(output_folder, filename)
                img_bytes = fig.to_image(format="png", width=width, height=height, scale=dpi / 96)
                with open(filepath, "wb") as f:
                    f.write(img_bytes)
            else:
                filename = "_".join(parts) + ".html"
                filepath = os.path.join(output_folder, filename)
                fig.write_html(filepath)
            print(f"Saved: {filepath}")
        else:
            if render_mode == "static":
                from IPython.display import Image, display as ipy_display
                img_bytes = fig.to_image(format="png", width=width, height=height)
                ipy_display(Image(img_bytes))
            else:
                fig.show()
