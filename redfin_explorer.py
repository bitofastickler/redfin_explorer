"""
Redfin Explorer Dash App (single-file)

Features
- Filters + charts for exploring listings
- Highlights "good deals" as the bottom X percentile of $/sqft
- Deals-only toggle
- Currency-readable price readout (and medians formatted as currency)
- Sliders update while dragging (so medians/plots feel "live")
- Zoom/pan-friendly map that preserves view between filter changes

Run:
  pip install -U dash pandas plotly
  python redfin_dash_app_v4.py

Then open:
  http://127.0.0.1:8050
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html, dash_table
from dash.dash_table import FormatTemplate
from dash.dash_table.Format import Format, Group, Scheme, Symbol


# --------- Config ---------
DEFAULT_CSV = os.environ.get("REDFIN_CSV", "redfin_2026-01-03_anamosa.csv") # your .csv file name goes here, just make sure it's co-located with your .py file
PORT = int(os.environ.get("PORT", "8050"))


def load_redfin(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Drop obvious non-data rows (Redfin sometimes includes a MLS disclaimer row)
    # Keep rows that have at least a price or address.
    if "PRICE" in df.columns or "ADDRESS" in df.columns:
        df = df[df.get("PRICE").notna() | df.get("ADDRESS").notna()].copy()

    # Normalize URL column name if present
    rename = {}
    for col in df.columns:
        if str(col).strip().upper().startswith("URL (SEE"):
            rename[col] = "URL"
    df = df.rename(columns=rename)

    # Standardize numerics (if present)
    numeric_cols = ["PRICE", "SQUARE FEET", "$/SQUARE FEET", "BEDS", "BATHS", "LOT SIZE", "YEAR BUILT", "DAYS ON MARKET"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute $/sqft if missing
    if "$/SQUARE FEET" not in df.columns and {"PRICE", "SQUARE FEET"}.issubset(df.columns):
        df["$/SQUARE FEET"] = np.where(
            (df["PRICE"].notna()) & (df["SQUARE FEET"].notna()) & (df["SQUARE FEET"] > 0),
            df["PRICE"] / df["SQUARE FEET"],
            np.nan,
        )

    # Ensure columns exist
    for c in ["BEDS", "SQUARE FEET", "PRICE", "$/SQUARE FEET", "YEAR BUILT"]:
        if c not in df.columns:
            df[c] = np.nan

    # Derived fields
    current_year = datetime.now().year
    df["AGE"] = np.where(df["YEAR BUILT"].notna(), current_year - df["YEAR BUILT"], np.nan)

    # Label for hovers
    def _safe_str(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str)

    zip_col = "ZIP OR POSTAL CODE" if "ZIP OR POSTAL CODE" in df.columns else None
    zip_str = _safe_str(df[zip_col]) if zip_col else pd.Series([""] * len(df), index=df.index)

    df["LABEL"] = (
        _safe_str(df.get("ADDRESS", pd.Series([""] * len(df), index=df.index)))
        + ", "
        + _safe_str(df.get("CITY", pd.Series([""] * len(df), index=df.index)))
        + ", "
        + _safe_str(df.get("STATE OR PROVINCE", pd.Series([""] * len(df), index=df.index)))
        + " "
        + zip_str.str.replace(r"\.0$", "", regex=True)
    ).str.strip(" ,")

    # Home-ish rows
    df["IS_HOME"] = df["BEDS"].notna() & df["SQUARE FEET"].notna() & df["PRICE"].notna()

    return df


def make_app(df: pd.DataFrame) -> Dash:
    app = Dash(__name__)
    app.title = "Redfin Explorer"

    def _minmax(col: str, default=(0.0, 1.0)):
        if col not in df.columns:
            return default
        s = df[col].dropna()
        if len(s) == 0:
            return default
        return (float(s.min()), float(s.max()))

    price_min, price_max = _minmax("PRICE", (0, 1_000_000))
    sqft_min, sqft_max = _minmax("SQUARE FEET", (0, 4000))
    beds_min, beds_max = _minmax("BEDS", (0, 6))
    baths_min, baths_max = _minmax("BATHS", (0, 4))
    year_min, year_max = _minmax("YEAR BUILT", (1900, datetime.now().year))
    dom_min, dom_max = _minmax("DAYS ON MARKET", (0, 180))

    status_options = sorted([x for x in df.get("STATUS", pd.Series(dtype=str)).dropna().unique()])
    city_options = sorted([x for x in df.get("CITY", pd.Series(dtype=str)).dropna().unique()])
    ptype_options = sorted([x for x in df.get("PROPERTY TYPE", pd.Series(dtype=str)).dropna().unique()])

    default_status = status_options if status_options else []
    default_city = []  # empty means "all"
    default_ptype = [x for x in ptype_options if "vacant" not in str(x).lower()] or ptype_options

    app.layout = html.Div(
        style={"fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif", "padding": "14px"},
        children=[
            html.H2("Redfin Explorer (Dash + Plotly)", style={"margin": "0 0 10px 0"}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "360px 1fr", "gap": "14px", "alignItems": "start"},
                children=[
                    # Sidebar
                    html.Div(
                        style={
                            "border": "1px solid #ddd",
                            "borderRadius": "12px",
                            "padding": "12px",
                            "boxShadow": "0 1px 8px rgba(0,0,0,0.04)",
                            "position": "sticky",
                            "top": "10px",
                        },
                        children=[
                            html.Div(
                                style={"display": "flex", "gap": "10px", "alignItems": "center", "flexWrap": "wrap"},
                                children=[
                                    dcc.Checklist(
                                        id="home-only",
                                        options=[{"label": "Homes only (beds + sqft + price present)", "value": "home"}],
                                        value=["home"],
                                        inputStyle={"marginRight": "6px"},
                                    ),
                                    dcc.Checklist(
                                        id="deals-only",
                                        options=[{"label": "Deals only", "value": "deals"}],
                                        value=[],
                                        inputStyle={"marginRight": "6px"},
                                    ),
                                ],
                            ),
                            html.Hr(),
                            html.Label("Status"),
                            dcc.Dropdown(
                                id="status",
                                options=[{"label": s, "value": s} for s in status_options],
                                value=default_status,
                                multi=True,
                                placeholder="All",
                            ),
                            html.Div(style={"height": "8px"}),
                            html.Label("City"),
                            dcc.Dropdown(
                                id="city",
                                options=[{"label": c, "value": c} for c in city_options],
                                value=default_city,
                                multi=True,
                                placeholder="All",
                            ),
                            html.Div(style={"height": "8px"}),
                            html.Label("Property type"),
                            dcc.Dropdown(
                                id="ptype",
                                options=[{"label": p, "value": p} for p in ptype_options],
                                value=default_ptype,
                                multi=True,
                                placeholder="All",
                            ),
                            html.Hr(),
                            html.Div(id="price_readout", style={"fontWeight": "700", "marginBottom": "4px"}),
                            html.Label("Price ($)"),
                            dcc.RangeSlider(
                                id="price",
                                min=price_min,
                                max=price_max,
                                step=max(1000, (price_max - price_min) / 200) if price_max > price_min else 1000,
                                value=[price_min, price_max],
                                tooltip={"placement": "bottom", "always_visible": False},
                                allowCross=False,
                                updatemode="drag",  # <-- live updates while dragging
                            ),
                            html.Div(style={"height": "8px"}),
                            html.Label("Square feet"),
                            dcc.RangeSlider(
                                id="sqft",
                                min=sqft_min,
                                max=sqft_max,
                                step=max(10, (sqft_max - sqft_min) / 200) if sqft_max > sqft_min else 10,
                                value=[sqft_min, sqft_max],
                                tooltip={"placement": "bottom", "always_visible": False},
                                allowCross=False,
                                updatemode="drag",
                            ),
                            html.Div(style={"height": "8px"}),
                            html.Label("Beds"),
                            dcc.RangeSlider(
                                id="beds",
                                min=beds_min,
                                max=beds_max,
                                step=1,
                                value=[beds_min, beds_max],
                                tooltip={"placement": "bottom", "always_visible": False},
                                allowCross=False,
                                updatemode="drag",
                            ),
                            html.Div(style={"height": "8px"}),
                            html.Label("Baths"),
                            dcc.RangeSlider(
                                id="baths",
                                min=baths_min,
                                max=baths_max,
                                step=0.5,
                                value=[baths_min, baths_max],
                                tooltip={"placement": "bottom", "always_visible": False},
                                allowCross=False,
                                updatemode="drag",
                            ),
                            html.Div(style={"height": "8px"}),
                            html.Label("Year built"),
                            dcc.RangeSlider(
                                id="year",
                                min=year_min,
                                max=year_max,
                                step=1,
                                value=[year_min, year_max],
                                tooltip={"placement": "bottom", "always_visible": False},
                                allowCross=False,
                                updatemode="drag",
                            ),
                            html.Div(style={"height": "8px"}),
                            html.Label("Days on market"),
                            dcc.RangeSlider(
                                id="dom",
                                min=dom_min,
                                max=dom_max,
                                step=1,
                                value=[dom_min, dom_max],
                                tooltip={"placement": "bottom", "always_visible": False},
                                allowCross=False,
                                updatemode="drag",
                            ),
                            html.Hr(),
                            html.Label("Deal highlight (bottom X percentile of $/sqft)"),
                            dcc.Slider(
                                id="deal_pct",
                                min=5,
                                max=50,
                                step=1,
                                value=20,
                                marks={5: "5%", 10: "10%", 20: "20%", 30: "30%", 40: "40%", 50: "50%"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            html.Div(id="summary", style={"marginTop": "12px", "fontSize": "14px"}),
                        ],
                    ),

                    # Main content
                    html.Div(
                        children=[
                            html.Div(
                                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
                                children=[
                                    dcc.Graph(id="fig_ppsf", config={"displayModeBar": False}),
                                    dcc.Graph(id="fig_scatter", config={"displayModeBar": False}),
                                ],
                            ),
                            html.Div(
                                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
                                children=[
                                    dcc.Graph(id="fig_age", config={"displayModeBar": False}),
                                    dcc.Graph(id="fig_map", config={"displayModeBar": True, "scrollZoom": True}),
                                ],
                            ),
                            html.H3("Best deals (lowest $/sqft)", style={"marginTop": "18px"}),
                            dash_table.DataTable(
                                id="tbl",
                                markdown_options={"link_target": "_blank"},
                                page_size=12,
                                sort_action="native",
                                filter_action="none",
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "padding": "8px",
                                    "fontSize": "13px",
                                    "whiteSpace": "normal",
                                    "height": "auto",
                                },
                                style_header={"fontWeight": "700"},
                                style_data_conditional=[
                                    {"if": {"filter_query": "{DEAL} = 'YES'"}, "fontWeight": "700"}
                                ],
                            ),
                            html.Div(
                                style={"marginTop": "8px", "fontSize": "12px", "color": "#555"},
                                children=[
                                    "Tip: For true *price trends over time*, save multiple Redfin exports over weeks/months and append snapshots. ",
                                    "This dashboard is a snapshot explorer (but easy to extend to time-series).",
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        Output("fig_ppsf", "figure"),
        Output("fig_scatter", "figure"),
        Output("fig_age", "figure"),
        Output("fig_map", "figure"),
        Output("tbl", "data"),
        Output("tbl", "columns"),
        Output("summary", "children"),
        Output("price_readout", "children"),
        Input("home-only", "value"),
        Input("deals-only", "value"),
        Input("status", "value"),
        Input("city", "value"),
        Input("ptype", "value"),
        Input("price", "value"),
        Input("sqft", "value"),
        Input("beds", "value"),
        Input("baths", "value"),
        Input("year", "value"),
        Input("dom", "value"),
        Input("deal_pct", "value"),
    )
    def update(home_only, deals_only, status, city, ptype, price, sqft, beds, baths, year, dom, deal_pct):
        dff = df.copy()

        if home_only and "home" in home_only:
            dff = dff[dff["IS_HOME"]]

        if status and "STATUS" in dff.columns:
            dff = dff[dff["STATUS"].isin(status)]
        if city and "CITY" in dff.columns:
            dff = dff[dff["CITY"].isin(city)]
        if ptype and "PROPERTY TYPE" in dff.columns:
            dff = dff[dff["PROPERTY TYPE"].isin(ptype)]

        # Range filters (keep NaNs so missing fields don't vanish)
        dff = dff[(dff["PRICE"].between(price[0], price[1], inclusive="both")) | dff["PRICE"].isna()]
        dff = dff[(dff["SQUARE FEET"].between(sqft[0], sqft[1], inclusive="both")) | dff["SQUARE FEET"].isna()]
        dff = dff[(dff["BEDS"].between(beds[0], beds[1], inclusive="both")) | dff["BEDS"].isna()]
        dff = dff[(dff["BATHS"].between(baths[0], baths[1], inclusive="both")) | dff["BATHS"].isna()]
        dff = dff[(dff["YEAR BUILT"].between(year[0], year[1], inclusive="both")) | dff["YEAR BUILT"].isna()]

        if "DAYS ON MARKET" in dff.columns:
            dff = dff[(dff["DAYS ON MARKET"].between(dom[0], dom[1], inclusive="both")) | dff["DAYS ON MARKET"].isna()]

        # Deals: bottom X percentile of $/sqft within current filtered set
        ppsf = dff["$/SQUARE FEET"].dropna()
        thr = float(np.percentile(ppsf, deal_pct)) if len(ppsf) > 0 else np.nan
        dff["DEAL"] = np.where(dff["$/SQUARE FEET"].notna() & (dff["$/SQUARE FEET"] <= thr), "YES", "NO")

        # Optional: show only deals
        if deals_only and "deals" in deals_only:
            dff = dff[dff["DEAL"] == "YES"].copy()

        # Figures
        fig_ppsf = px.histogram(
            dff.dropna(subset=["$/SQUARE FEET"]),
            x="$/SQUARE FEET",
            nbins=30,
            title=f"$/sqft distribution (deal threshold ≈ {thr:,.0f} $/sqft)" if np.isfinite(thr) else "$/sqft distribution",
        )
        fig_ppsf.update_xaxes(tickformat="$,.0f")
        fig_ppsf.update_layout(margin=dict(l=10, r=10, t=50, b=10))

        fig_scatter = px.scatter(
            dff.dropna(subset=["PRICE", "SQUARE FEET", "$/SQUARE FEET"]),
            x="SQUARE FEET",
            y="PRICE",
            color="DEAL",
            hover_name="LABEL",
            hover_data={
                "BEDS": True,
                "BATHS": True,
                "YEAR BUILT": True,
                "DAYS ON MARKET": True,
                "$/SQUARE FEET": ":.0f",
                "DEAL": True,
            },
            title="Price vs square feet (deals highlighted)",
        )
        fig_scatter.update_yaxes(tickformat="$,.0f")
        fig_scatter.update_xaxes(tickformat=",.0f")
        fig_scatter.update_layout(margin=dict(l=10, r=10, t=50, b=10))

        dff_age = dff.dropna(subset=["$/SQUARE FEET", "AGE"])
        fig_age = px.scatter(
            dff_age,
            x="AGE",
            y="$/SQUARE FEET",
            color="DEAL",
            hover_name="LABEL",
            hover_data={"PRICE": ":.0f", "SQUARE FEET": True, "YEAR BUILT": True, "BEDS": True, "BATHS": True},
            title="$/sqft vs home age",
        )
        fig_age.update_yaxes(tickformat="$,.0f")
        fig_age.update_xaxes(tickformat=",.0f")
        fig_age.update_layout(margin=dict(l=10, r=10, t=50, b=10))

        # Map: center on filtered points + preserve zoom/pan between updates
        if {"LATITUDE", "LONGITUDE"}.issubset(dff.columns):
            map_df = dff.dropna(subset=["LATITUDE", "LONGITUDE"])
        else:
            map_df = pd.DataFrame()

        if len(map_df) > 0:
            fig_map = px.scatter_mapbox(
                map_df,
                lat="LATITUDE",
                lon="LONGITUDE",
                color="DEAL",
                hover_name="LABEL",
                hover_data={"PRICE": ":.0f", "$/SQUARE FEET": ":.0f", "BEDS": True, "BATHS": True, "STATUS": True},
                height=450,
                title="Map (deals highlighted)",
            )
            center = {"lat": float(map_df["LATITUDE"].mean()), "lon": float(map_df["LONGITUDE"].mean())}
            fig_map.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(center=center, zoom=12),
                uirevision="keep-map-view",
                margin=dict(l=10, r=10, t=50, b=10),
            )
        else:
            fig_map = px.scatter_mapbox(title="Map (no lat/long in filtered data)")
            fig_map.update_layout(
                mapbox_style="open-street-map",
                uirevision="keep-map-view",
                margin=dict(l=10, r=10, t=50, b=10),
            )

        # Table
        table_cols = [
            "ADDRESS",
            "CITY",
            "STATE OR PROVINCE",
            "ZIP OR POSTAL CODE",
            "STATUS",
            "PRICE",
            "SQUARE FEET",
            "$/SQUARE FEET",
            "BEDS",
            "BATHS",
            "YEAR BUILT",
            "AGE",
            "DAYS ON MARKET",
            "DEAL",
            "URL",
        ]
        table_cols = [c for c in table_cols if c in dff.columns]

        tbl = dff.copy()
        tbl["_deal_rank"] = np.where(tbl["DEAL"] == "YES", 0, 1)
        tbl = tbl.sort_values(by=["_deal_rank", "$/SQUARE FEET"], ascending=[True, True]).drop(columns=["_deal_rank"])

        tbl_out = tbl[table_cols].head(250).copy()

        # Make URL clickable
        if "URL" in tbl_out.columns:
            tbl_out["URL"] = tbl_out["URL"].fillna("").astype(str)
            tbl_out["URL"] = tbl_out["URL"].apply(lambda u: f"[link]({u})" if u.startswith("http") else "")

        data = tbl_out.to_dict("records")

        # Column formatting
        columns = []
        for c in tbl_out.columns:
            coldef = {"name": c, "id": c}
            if c == "URL":
                coldef["presentation"] = "markdown"
            if c == "PRICE":
                coldef["type"] = "numeric"
                coldef["format"] = FormatTemplate.money(0)
            elif c == "$/SQUARE FEET":
                coldef["type"] = "numeric"
                coldef["format"] = Format(
                    precision=0,
                    scheme=Scheme.fixed,
                    group=Group.yes,
                    symbol=Symbol.yes,
                    symbol_prefix="$",
                )
            elif c == "SQUARE FEET":
                coldef["type"] = "numeric"
                coldef["format"] = Format(precision=0, scheme=Scheme.fixed, group=Group.yes)
            elif c in ("BEDS", "YEAR BUILT", "AGE", "DAYS ON MARKET"):
                coldef["type"] = "numeric"
                coldef["format"] = Format(precision=0, scheme=Scheme.fixed, group=Group.yes)
            elif c == "BATHS":
                coldef["type"] = "numeric"
                coldef["format"] = Format(precision=1, scheme=Scheme.fixed, group=Group.yes)
            columns.append(coldef)

        # Summary (computed AFTER filters so it updates with them)
        n = len(dff)
        n_deals = int((dff["DEAL"] == "YES").sum()) if "DEAL" in dff.columns else 0
        med_price = float(dff["PRICE"].median()) if dff["PRICE"].notna().any() else np.nan
        med_ppsf = float(dff["$/SQUARE FEET"].median()) if dff["$/SQUARE FEET"].notna().any() else np.nan

        summary = html.Div(
            [
                html.Div(f"Filtered listings: {n:,}"),
                html.Div(f"Deals highlighted: {n_deals:,} (bottom {deal_pct}% of $/sqft)"),
                html.Div(f"Median price: ${med_price:,.0f}" if np.isfinite(med_price) else "Median price: n/a"),
                html.Div(f"Median $/sqft: ${med_ppsf:,.0f}" if np.isfinite(med_ppsf) else "Median $/sqft: n/a"),
            ]
        )

        price_text = f"Price: ${price[0]:,.0f} – ${price[1]:,.0f}"

        return fig_ppsf, fig_scatter, fig_age, fig_map, data, columns, summary, price_text

    return app


if __name__ == "__main__":
    df = load_redfin(DEFAULT_CSV)
    app = make_app(df)

    # Dash 3+: app.run()
    # Dash 2 (legacy): app.run_server()
    try:
        app.run(debug=True, port=PORT)
    except AttributeError:
        app.run_server(debug=True, port=PORT)
