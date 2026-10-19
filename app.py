from __future__ import annotations

import io
import math
import re
import time
from datetime import date, datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", page_icon="📈", layout="wide")

COUNTRIES = {
    "Mexico (MX)": "MEX", "Brazil (BR)": "BRA", "Argentina (AR)": "ARG",
    "Colombia (CO)": "COL", "Chile (CL)": "CHL", "Peru (PE)": "PER",
    "Venezuela (VE)": "VEN", "Ecuador (EC)": "ECU", "Uruguay (UY)": "URY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Life expectancy at birth is the average number of years a newborn would live if current age-specific mortality rates persisted."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Unemployment is the share of the labor force without work but available for and seeking employment, using national estimates."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US$ per person", "GDP per capita is gross domestic product divided by population. It is not average wealth or household income."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Crude birth rate is annual live births per 1,000 people, not a total birth count."),
    "Net migration": ("SM.POP.NETM", "people (net)", "Net migration is the difference between the number of immigrants and emigrants over the period; it is not an emigration count."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicide rate is the number of victims per 100,000 people, as defined by the source."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Consumer price inflation is the annual percentage change in the consumer price index."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt is reported as a percentage of GDP."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US$", "Total reserves including gold are reported in current US dollars."),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services is exports minus imports of goods and services, expressed as a percentage of GDP."),
}
CONTEXTS = {
    "Business": ("revenue", "dollars", "A business model can describe how revenue changes with time or sales volume.", "0.08*x**2 + 3*x + 20"),
    "Engineering": ("position", "meters", "Position describes location; its first derivative is velocity and its second derivative is acceleration.", "-0.5*x**2 + 8*x"),
    "Medical": ("concentration", "mg/L", "A concentration curve can describe a measured quantity over time; it is illustrative, not medical advice.", "80*exp(-0.25*x)"),
    "Sports science": ("performance", "points", "A performance model may rise and then level off or decline.", "-0.4*x**2 + 5*x + 10"),
    "Biology": ("population", "organisms", "A population model can represent growth over time, within the model's assumptions.", "100*exp(0.12*x)"),
    "Bioengineering": ("signal", "signal units", "A signal model can describe periodic or changing measured output.", "2*sin(x) + 0.2*x"),
    "Economic forecasting": ("index", "index points", "An economic index model describes a trend; extrapolations are uncertain and not forecasts guaranteed to occur.", "0.03*x**2 + 1.2*x + 100"),
    "Weather": ("temperature", "°C", "A temperature model can show changing measurements; a simple curve cannot capture all weather dynamics.", "8*sin(2*pi*x/24) + 15"),
    "Custom": ("output", "units", "Describe what the dependent variable means in your own context.", "x**2"),
}
FUNCTIONS = {
    "Polynomial": "0.5*x**3 - 3*x**2 + 2*x + 5",
    "Rational": "(x**2 + 1)/(x - 2)",
    "Exponential": "2*exp(0.3*x)",
    "Logarithmic": "5*log(x)",
    "Trigonometric": "3*sin(x) + 0.5*x",
}
WB_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?date={start}:{end}&format=json&per_page=20000"

def get_source_data(country: str, indicator: str, start_year: int, end_year: int) -> tuple[pd.DataFrame, str | None]:
    """Retrieve authentic annual values from the World Bank API. No synthetic fallback."""
    url = WB_URL.format(country=country, indicator=indicator, start=start_year, end=end_year)
    try:
        response = requests.get(url, timeout=25, headers={"User-Agent": "Hispanic-Heritage-Data-Explorer/1.0"})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            return pd.DataFrame(columns=["Year", "Value"]), "The API returned no observations for this selection."
        rows = []
        for item in payload[1]:
            if item and item.get("date") and item.get("value") is not None:
                try:
                    rows.append({"Year": int(item["date"]), "Value": float(item["value"])})
                except (TypeError, ValueError):
                    continue
        frame = pd.DataFrame(rows, columns=["Year", "Value"]).drop_duplicates("Year").sort_values("Year")
        if frame.empty:
            return frame, "The API returned an empty/null series for this country and indicator."
        return frame.reset_index(drop=True), None
    except requests.Timeout:
        return pd.DataFrame(columns=["Year", "Value"]), "World Bank request timed out. Try again later."
    except (requests.RequestException, ValueError, KeyError) as exc:
        return pd.DataFrame(columns=["Year", "Value"]), f"World Bank request failed: {exc}"

@st.cache_data(ttl=3600, show_spinner=False)
def cached_source_data(country: str, indicator: str, start_year: int, end_year: int) -> tuple[pd.DataFrame, str | None]:
    return get_source_data(country, indicator, start_year, end_year)

def safe_expression(source: str, x: sp.Symbol) -> sp.Expr:
    """Parse a restricted mathematical expression; reject unknown symbols and unsafe syntax."""
    if len(source) > 180:
        raise ValueError("Expression is too long (maximum 180 characters).")
    allowed = {
        "x": x, "pi": sp.pi, "E": sp.E, "e": sp.E,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
        "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
        "abs": sp.Abs, "Abs": sp.Abs,
    }
    try:
        expr = sp.sympify(source, locals=allowed, evaluate=True)
    except Exception as exc:
        raise ValueError(f"Could not parse the expression: {exc}") from exc
    if expr.free_symbols - {x}:
        raise ValueError("Use only x and supported functions: sin, cos, tan, exp, log, sqrt, abs, pi.")
    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValueError("Expression contains an undefined or infinite constant.")
    return expr

def lambdify_real(expr: sp.Expr, x: sp.Symbol):
    fn = sp.lambdify(x, expr, modules=["numpy"])
    def evaluate(values):
        with np.errstate(all="ignore"):
            result = np.asarray(fn(values), dtype=float)
        if result.ndim == 0:
            result = np.full_like(np.asarray(values, dtype=float), float(result), dtype=float)
        return result
    return evaluate

def fit_polynomial(data: pd.DataFrame, degree: int):
    clean = data[["Year", "Value"]].dropna().copy()
    clean["Year"] = pd.to_numeric(clean["Year"], errors="coerce")
    clean["Value"] = pd.to_numeric(clean["Value"], errors="coerce")
    clean = clean.dropna().sort_values("Year")
    clean = clean[clean["Year"].between(1800, 2200)]
    clean = clean.drop_duplicates("Year")
    if len(clean) < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree + 1} distinct valid year/value observations; only {len(clean)} are available.")
    base_year = int(clean["Year"].min())
    t = clean["Year"].to_numpy(dtype=float) - base_year
    y = clean["Value"].to_numpy(dtype=float)
    coeffs = np.polyfit(t, y, degree)
    poly = np.poly1d(coeffs)
    pred = poly(t)
    residuals = y - pred
    r2 = float(r2_score(y, pred)) if len(y) > 1 else float("nan")
    mae = float(mean_absolute_error(y, pred))
    return base_year, coeffs, poly, pred, residuals, r2, mae, clean

def format_polynomial(coeffs, base_year, degree):
    terms = []
    for i, coef in enumerate(coeffs):
        power = degree - i
        if abs(coef) < 1e-12:
            continue
        if power == 0:
            term = f"{coef:.8g}"
        elif power == 1:
            term = f"{coef:.8g}t"
        else:
            term = f"{coef:.8g}t^{power}"
        terms.append(term)
    return "f(year) = " + (" + ".join(terms).replace("+ -", "- ") if terms else "0") + f", where t = year − {base_year}"

def validate_editable_data(frame: pd.DataFrame, indicator_name: str):
    errors = []
    if not {"Year", "Value"}.issubset(frame.columns):
        return ["Working table must have Year and Value columns."]
    data = frame[["Year", "Value"]].copy()
    years = pd.to_numeric(data["Year"], errors="coerce")
    values = pd.to_numeric(data["Value"], errors="coerce")
    if years.isna().any() or not np.all(np.isclose(years.dropna(), np.round(years.dropna()))):
        errors.append("Every Year must be an integer.")
    if years.duplicated().any():
        errors.append("Year values must be unique; remove duplicate years.")
    if values.isna().any():
        errors.append("Every Value must be numeric; blank values are not valid in the working table.")
    if ((years < 1800) | (years > 2200)).any():
        errors.append("Years must be between 1800 and 2200.")
    if indicator_name == "Life expectancy at birth" and ((values <= 0) | (values > 125)).any():
        errors.append("Life expectancy values must be greater than 0 and no more than 125 years.")
    if indicator_name in ("Unemployment, national estimate", "Consumer price inflation") and (values < -100).any():
        errors.append("Values below -100 are not accepted for this indicator.")
    if indicator_name == "Crude birth rate" and (values < 0).any():
        errors.append("Crude birth rate cannot be negative.")
    if indicator_name == "Intentional homicide rate" and (values < 0).any():
        errors.append("Homicide rate cannot be negative.")
    return errors

def build_chart(data, base_year, poly, degree, horizon, sample_increment, single_results, batch_results, indicator_label, units):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    years = data["Year"].to_numpy(dtype=float)
    vals = data["Value"].to_numpy(dtype=float)
    ax.scatter(years, vals, label="Observed / working data", s=24, zorder=4)
    left, right = float(np.min(years)), float(np.max(years))
    smooth_end = right + horizon
    curve_years = np.linspace(left, max(right, smooth_end), 700)
    curve_vals = poly(curve_years - base_year)
    ax.plot(curve_years[curve_years <= right], curve_vals[curve_years <= right], label="Fitted curve (historical range)", linewidth=2)
    if horizon > 0:
        mask = curve_years >= right
        ax.plot(curve_years[mask], curve_vals[mask], "--", label="Extrapolated segment", linewidth=2)
        ax.axvline(right, linestyle=":", linewidth=1.3, label=f"Historical boundary ({int(right)})")
    sampled = np.arange(int(left), int(right) + 1, sample_increment)
    sampled = sampled[np.isin(sampled, years.astype(int))]
    if len(sampled):
        sampled_vals = [poly(y - base_year) for y in sampled]
        ax.scatter(sampled, sampled_vals, marker="x", s=50, label=f"Model markers every {sample_increment} year(s)")
    if single_results:
        sy = [r["Year"] for r in single_results]
        sv = [r["Estimate"] for r in single_results]
        ax.scatter(sy, sv, marker="*", s=140, label="Calculated year(s)", zorder=5)
    if batch_results:
        by = [r["Year"] for r in batch_results]
        bv = [r["Estimate"] for r in batch_results]
        ax.scatter(by, bv, marker="D", s=38, label="Batch estimates", zorder=5)
    ax.set_title(f"{indicator_label}: {units}")
    ax.set_xlabel("Year")
    ax.set_ylabel(units)
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig

def make_excel(summary: dict, data: pd.DataFrame, single: pd.DataFrame, batch: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        data.to_excel(writer, sheet_name="Working Data", index=False)
        single.to_excel(writer, sheet_name="Single-Year Results", index=False)
        batch.to_excel(writer, sheet_name="Batch Results", index=False)
        workbook = writer.book
        header = workbook.add_format({"bold": True, "text_wrap": True, "valign": "top", "bg_color": "#D9EAF7", "border": 1})
        for ws_name, ws in writer.sheets.items():
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, max(1, ({"Summary": 1, "Working Data": len(data), "Single-Year Results": len(single), "Batch Results": len(batch)}[ws_name])), max(0, len(({"Summary": list(summary.keys()), "Working Data": list(data.columns), "Single-Year Results": list(single.columns), "Batch Results": list(batch.columns)}[ws_name])) - 1))
            ws.set_row(0, 30, header)
            ws.set_column(0, 12, 20)
    return output.getvalue()

def make_pdf(summary: dict, equation: str, metrics: str, chart_bytes: bytes, data: pd.DataFrame, single: pd.DataFrame, batch: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=landscape(letter), rightMargin=28, leftMargin=28, topMargin=26, bottomMargin=26)
    styles = getSampleStyleSheet()
    story = [Paragraph("Hispanic Heritage Historical Data Explorer", styles["Title"]), Spacer(1, 8)]
    meta = " | ".join(f"{k}: {v}" for k, v in summary.items())
    story += [Paragraph(meta.replace("&", "&amp;"), styles["BodyText"]), Spacer(1, 8),
              Paragraph("Model equation", styles["Heading2"]), Paragraph(equation.replace("&", "&amp;"), styles["BodyText"]),
              Paragraph(metrics.replace("&", "&amp;"), styles["BodyText"]), Spacer(1, 8)]
    if chart_bytes:
        story.append(Image(io.BytesIO(chart_bytes), width=660, height=335))
    story.append(Paragraph("Working data preview (first 20 rows)", styles["Heading2"]))
    story.append(_pdf_table(data.head(20)))
    if len(single):
        story += [Paragraph("Single-year calculated results", styles["Heading2"]), _pdf_table(single.head(30))]
    if len(batch):
        story += [Paragraph("Batch calculated results", styles["Heading2"]), _pdf_table(batch.head(30))]
    story += [Spacer(1, 8), Paragraph("Limitations: polynomial fits describe the selected data, not causation. Extrapolation can be unstable, especially at high degree or far outside the observed range. MAE and R-squared are in-sample unless explicitly labeled otherwise.", styles["BodyText"])]
    doc.build(story)
    return output.getvalue()

def _pdf_table(frame):
    if frame is None or frame.empty:
        return Table([["No results"]])
    cols = list(frame.columns)
    rows = [[str(c)[:30] for c in cols]]
    for _, row in frame.iterrows():
        rows.append([str(v)[:42] for v in row.tolist()])
    table = Table(rows, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), .35, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return table

st.title("Hispanic Heritage Historical Data Explorer")
st.caption("Explore authentic World Bank historical indicators, fit polynomial models, and investigate functions through Algebra 2, Precalculus, and Calculus.")
st.info("Brazil is included as a Latin American comparison country. Country comparisons are descriptive; coverage and definitions may differ by indicator.")

with st.sidebar:
    st.header("Historical data")
    country_label = st.selectbox("Featured country", list(COUNTRIES.keys()), index=0)
    indicator_label = st.selectbox("Indicator", list(INDICATORS.keys()), index=0)
    current_year = date.today().year
    start_year, end_year = current_year - 70, current_year - 1
    st.caption(f"Requested period: {start_year}–{end_year} (current year minus 70 through minus 1)")
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_increment = st.slider("Graph sample increment (years)", 1, 10, 5)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    comparison_labels = st.multiselect("Comparison countries (same indicator)", [c for c in COUNTRIES if c != country_label], default=["Brazil (BR)"] if country_label != "Brazil (BR)" else ["Mexico (MX)"])
    printer_mode = st.checkbox("Printer-friendly view", value=False)

indicator_code, units, definition = INDICATORS[indicator_label]
source_country = COUNTRIES[country_label]
with st.spinner("Retrieving World Bank data…"):
    original_data, source_error = cached_source_data(source_country, indicator_code, start_year, end_year)
retrieval_date = date.today().isoformat()
source_url = WB_URL.format(country=source_country, indicator=indicator_code, start=start_year, end=end_year)

if source_error:
    st.warning(source_error)
if original_data.empty:
    st.error("No source observations are available to fit for this selection. Choose another country or indicator, or retry when the API is available.")
    st.markdown(f"[Open World Bank API request]({source_url})")
    st.stop()

coverage = f"{int(original_data.Year.min())}–{int(original_data.Year.max())}"
st.markdown(f"**Source:** [World Bank API]({source_url}) · **Retrieved:** {retrieval_date} · **Available observations:** {len(original_data)} · **Actual coverage:** {coverage}")
st.caption(definition + f" Units: {units}. The app requests a 70-year window but displays the actual coverage and count returned by the source.")

if "working_key" not in st.session_state:
    st.session_state.working_key = None
key = f"{source_country}|{indicator_code}|{start_year}|{end_year}"
if st.session_state.working_key != key:
    st.session_state.working_data = original_data.copy()
    st.session_state.working_key = key
    st.session_state.user_modified = False
    st.session_state.single_results = []
    st.session_state.batch_results = []
    st.session_state.last_model_signature = None
    st.session_state.animation_x = float(original_data["Year"].median())
if "single_results" not in st.session_state:
    st.session_state.single_results = []
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "user_modified" not in st.session_state:
    st.session_state.user_modified = False

top1, top2 = st.columns([3, 1])
with top1:
    st.subheader("Editable working data")
    st.caption("Edit the table below. Original source data stays unchanged. Click Reset to source to discard edits.")
with top2:
    if st.button("Reset to source", use_container_width=True):
        st.session_state.working_data = original_data.copy()
        st.session_state.user_modified = False
        st.session_state.single_results = []
        st.session_state.batch_results = []
        st.rerun()

edited = st.data_editor(st.session_state.working_data, num_rows="dynamic", use_container_width=True, key=f"editor_{key}")
if not edited.equals(st.session_state.working_data):
    st.session_state.working_data = edited.copy()
    st.session_state.user_modified = True
    st.session_state.single_results = []
    st.session_state.batch_results = []
working = st.session_state.working_data.copy()
validation_errors = validate_editable_data(working, indicator_label)
if st.session_state.user_modified:
    st.warning("USER-MODIFIED SCENARIO: model and exports use your edited working table, not only the original source data.")
if validation_errors:
    for err in validation_errors:
        st.error(err)
    st.stop()
working["Year"] = pd.to_numeric(working["Year"]).astype(int)
working["Value"] = pd.to_numeric(working["Value"]).astype(float)
working = working.sort_values("Year").reset_index(drop=True)
if len(working) < degree + 1:
    st.error(f"Degree {degree} requires at least {degree + 1} distinct years. Current valid rows: {len(working)}. Lower the degree or restore/add valid rows.")
    st.stop()

try:
    base_year, coeffs, poly, fitted, residuals, r2, mae, clean = fit_polynomial(working, degree)
except (ValueError, np.linalg.LinAlgError) as exc:
    st.error(f"Unable to fit model: {exc}")
    st.stop()
equation = format_polynomial(coeffs, base_year, degree)
signature = (source_country, indicator_code, degree, tuple(working["Year"]), tuple(np.round(working["Value"], 10)))
if st.session_state.last_model_signature != signature:
    st.session_state.single_results = []
    st.session_state.batch_results = []
    st.session_state.last_model_signature = signature

st.subheader("Polynomial model")
st.code(equation)
metric_cols = st.columns(5)
metric_cols[0].metric("Degree", degree)
metric_cols[1].metric("Observations", len(clean))
metric_cols[2].metric("In-sample R²", f"{r2:.4f}" if np.isfinite(r2) else "N/A")
metric_cols[3].metric("In-sample MAE", f"{mae:.4g} {units}")
metric_cols[4].metric("Mean observed value", f"{clean.Value.mean():.4g} {units}")
st.caption(f"Centered time base year: {base_year}. Historical fitted range: {int(clean.Year.min())}–{int(clean.Year.max())}. Metrics are in-sample and are not proof of predictive accuracy.")
resid_frame = pd.DataFrame({"Year": clean.Year, "Observed": clean.Value, "Fitted": fitted, "Residual (observed − fitted)": residuals})
with st.expander("Inspect fitted values and residuals"):
    st.dataframe(resid_frame, use_container_width=True, hide_index=True)

st.subheader("Calculate estimates")
left, right = st.columns(2)
with left:
    with st.form("single_year_form"):
        single_year = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=int(np.clip(round(float(clean.Year.median())), 1950, 2100)), step=1)
        calculate_single = st.form_submit_button("Calculate Value")
    if calculate_single:
        estimate = float(poly(single_year - base_year))
        actual_rows = clean.loc[clean.Year == single_year, "Value"]
        classification = "within fitted range" if clean.Year.min() <= single_year <= clean.Year.max() else "outside fitted range (extrapolation)"
        st.session_state.single_results = [{"Year": int(single_year), "Estimate": estimate, "Units": units, "Range classification": classification, "Observed value at year": float(actual_rows.iloc[0]) if not actual_rows.empty else None, "Status": "Fitted value at observed year" if not actual_rows.empty else "Model estimate only"}]
with right:
    batch_mode = st.radio("Batch input", ["Year range", "Custom years"], horizontal=True)
    with st.form("batch_form"):
        if batch_mode == "Year range":
            b1, b2, b3 = st.columns(3)
            batch_start = b1.number_input("Start", min_value=1800, max_value=2200, value=max(1950, int(clean.Year.min())), step=1)
            batch_end = b2.number_input("End", min_value=1800, max_value=2200, value=min(2100, int(clean.Year.max()) + 10), step=1)
            batch_step = b3.number_input("Step (1–10)", min_value=1, max_value=10, value=1, step=1)
            custom_years = ""
        else:
            custom_years = st.text_input("Comma-separated years", value="2000, 2010, 2020")
            batch_start, batch_end, batch_step = 0, 0, 1
        calculate_batch = st.form_submit_button("Calculate Batch")
    if calculate_batch:
        requested, invalid = [], []
        if batch_mode == "Year range":
            if batch_end < batch_start:
                invalid.append("End year must be greater than or equal to start year.")
            else:
                requested = list(range(int(batch_start), int(batch_end) + 1, int(batch_step)))
        else:
            for token in custom_years.split(","):
                token = token.strip()
                if not token:
                    continue
                if not re.fullmatch(r"[+-]?\d+", token):
                    invalid.append(f"'{token}' is not an integer year.")
                else:
                    yr = int(token)
                    if yr < 1800 or yr > 2200:
                        invalid.append(f"{yr} is outside the supported year range 1800–2200.")
                    else:
                        requested.append(yr)
        if invalid:
            for message in invalid:
                st.error(message)
        else:
            requested = list(dict.fromkeys(requested))
            if not requested:
                st.error("Enter at least one valid year.")
            else:
                st.session_state.batch_results = [
                    {"Year": y, "Estimate": float(poly(y - base_year)), "Units": units,
                     "Range classification": "within fitted range" if clean.Year.min() <= y <= clean.Year.max() else "outside fitted range (extrapolation)",
                     "Observed value at year": float(clean.loc[clean.Year == y, "Value"].iloc[0]) if (clean.Year == y).any() else None}
                    for y in requested
                ]
if st.session_state.single_results:
    st.markdown("**Single-year result**")
    st.dataframe(pd.DataFrame(st.session_state.single_results), use_container_width=True, hide_index=True)
if st.session_state.batch_results:
    st.markdown("**Batch results**")
    st.dataframe(pd.DataFrame(st.session_state.batch_results), use_container_width=True, hide_index=True)
    st.caption("Batch values use the same centered polynomial as the single-year calculator. Outside-range values are extrapolations, not observations.")

# Optional chronological holdout: fit on earlier rows, score on the last 20% only if enough data.
if len(clean) >= max(12, degree + 5):
    split = max(degree + 1, int(len(clean) * 0.8))
    train, test = clean.iloc[:split], clean.iloc[split:]
    if len(test):
        hold_base = int(train.Year.min())
        hold_coeff = np.polyfit(train.Year.to_numpy() - hold_base, train.Value.to_numpy(), degree)
        hold_pred = np.polyval(hold_coeff, test.Year.to_numpy() - hold_base)
        st.caption(f"Chronological holdout (last {len(test)} observations; trained on earlier years): MAE = {mean_absolute_error(test.Value, hold_pred):.4g} {units}. This is separate from the in-sample metrics.")

# Comparison series
comparison_frames = []
for comp_label in comparison_labels:
    comp_code = COUNTRIES[comp_label]
    comp_df, comp_error = cached_source_data(comp_code, indicator_code, start_year, end_year)
    if comp_error:
        st.warning(f"{comp_label}: {comp_error}")
    if not comp_df.empty:
        comparison_frames.append((comp_label, comp_df))
if comparison_frames:
    st.subheader("Country comparison")
    fig_cmp, ax_cmp = plt.subplots(figsize=(10, 4.2))
    ax_cmp.plot(clean.Year, clean.Value, marker="o", markersize=3, label=f"{country_label} (working data)")
    for comp_label, comp_df in comparison_frames:
        ax_cmp.plot(comp_df.Year, comp_df.Value, marker=".", markersize=3, label=comp_label)
    ax_cmp.set_title(f"{indicator_label} — country comparison")
    ax_cmp.set_xlabel("Year")
    ax_cmp.set_ylabel(units)
    ax_cmp.grid(True, alpha=.25)
    ax_cmp.legend(fontsize=8)
    fig_cmp.tight_layout()
    st.pyplot(fig_cmp)
    st.caption("Lines may cover different years because source availability varies by country. Missing years are gaps in source coverage, not zero values.")

fig = build_chart(clean, base_year, poly, degree, horizon, sample_increment,
                  st.session_state.single_results, st.session_state.batch_results, indicator_label, units)
if printer_mode:
    st.subheader("Printer-friendly report")
    st.markdown(f"**Country:** {country_label}  \n**Indicator:** {indicator_label}  \n**Units:** {units}  \n**Coverage:** {coverage}  \n**Retrieval date:** {retrieval_date}  \n**Data status:** {'User-modified scenario' if st.session_state.user_modified else 'Source data'}")
    st.markdown(f"**Model:** `{equation}`  \n**In-sample R²:** {r2:.4f}  \n**In-sample MAE:** {mae:.5g} {units}")
st.subheader("Historical data and model plot")
st.pyplot(fig)
st.caption("The fitted curve is shown separately from observed data. The dotted boundary marks the last observed year; the dashed segment is extrapolation. The sample increment changes model marker frequency. Do not treat polynomial extrapolations as validated forecasts.")

# Course pathway and interactive context/function lab
st.divider()
st.header("Function and Calculus Lab")
pathway = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"])
context = st.selectbox("Context", list(CONTEXTS.keys()))
default_dep, default_units, default_desc, context_expr = CONTEXTS[context]
variable_col, units_col = st.columns(2)
with variable_col:
    x_name = st.text_input("Independent variable name", value="time")
    x_units = st.text_input("Independent variable units", value="hours")
with units_col:
    dep_name = st.text_input("Dependent variable name", value=default_dep)
    dep_units = st.text_input("Dependent variable units", value=default_units)
description = st.text_area("Context description", value=default_desc, height=75)
function_source_mode = st.radio("Function source", ["World Bank polynomial fit", "Suggested function", "Choose function family", "Enter my own function"], horizontal=False)
if function_source_mode == "World Bank polynomial fit":
    # Shift x by base year so the fitted model is numerically stable and usable as a symbolic function.
    x = sp.Symbol("x", real=True)
    expr = sum(sp.Float(float(c), 14) * (x - base_year) ** (degree - i) for i, c in enumerate(coeffs))
    function_text = equation
elif function_source_mode == "Suggested function":
    x = sp.Symbol("x", real=True)
    expr = safe_expression(context_expr, x)
    function_text = context_expr
elif function_source_mode == "Choose function family":
    family = st.selectbox("Function family", list(FUNCTIONS.keys()))
    suggested = FUNCTIONS[family]
    function_text = st.text_input("Function expression (editable)", value=suggested)
    x = sp.Symbol("x", real=True)
    try:
        expr = safe_expression(function_text, x)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
else:
    function_text = st.text_input("Function f(x)", value="x**2 - 4*x + 1", help="Use * for multiplication and ** for powers. Supported: sin, cos, tan, exp, log, sqrt, abs, pi.")
    x = sp.Symbol("x", real=True)
    try:
        expr = safe_expression(function_text, x)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

st.latex(sp.latex(sp.Eq(sp.Function("f")(x), expr)))
st.caption("Expression parser supports polynomial, rational, exponential, logarithmic, and trigonometric forms. Use x as the independent variable; x is measured in the units you specify.")
with st.expander("Change model parameters"):
    st.write("For suggested functions, edit the expression directly to change coefficients or parameters. Examples: `a*x**2` is not accepted unless a is replaced by a number; try `2*x**2`, `3*exp(0.2*x)`, or `4*sin(x)`.")

# Analysis interval and current point
interval_default_low, interval_default_high = 0.0, 10.0
range_cols = st.columns(3)
x_min = range_cols[0].number_input("Analysis interval start", value=interval_default_low, min_value=-100000.0, max_value=100000.0, step=1.0)
x_max = range_cols[1].number_input("Analysis interval end", value=interval_default_high, min_value=-100000.0, max_value=100000.0, step=1.0)
if x_max <= x_min:
    st.error("Analysis interval end must be greater than start.")
    st.stop()
point_default = float(np.clip(st.session_state.get("animation_x", (x_min + x_max) / 2), x_min, x_max))
current_x = st.slider("Current input value", min_value=float(x_min), max_value=float(x_max), value=point_default, step=max((x_max-x_min)/200, 0.001), key="current_x_slider")
play_col, speed_col, reset_col = st.columns(3)
with play_col:
    auto_play = st.checkbox("Auto-play point", value=False)
with speed_col:
    playback_speed = st.select_slider("Playback speed", options=["Slow", "Normal", "Fast"], value="Normal", key="playback_speed")
with reset_col:
    if st.button("Reset point"):
        st.session_state.animation_x = (x_min + x_max) / 2
        st.rerun()
if auto_play:
    step_size = (x_max - x_min) / 100
    next_x = current_x + step_size
    if next_x > x_max:
        next_x = x_min
    st.session_state.animation_x = next_x
    time.sleep({"Slow": 0.5, "Normal": 0.22, "Fast": 0.08}[playback_speed])
    st.rerun()

try:
    f_fn = lambdify_real(expr, x)
    d1_expr = sp.diff(expr, x)
    d2_expr = sp.diff(d1_expr, x)
    d1_fn = lambdify_real(d1_expr, x)
    d2_fn = lambdify_real(d2_expr, x)
    f_current = float(f_fn(np.array([current_x]))[0])
    d1_current = float(d1_fn(np.array([current_x]))[0])
    d2_current = float(d2_fn(np.array([current_x]))[0])
except Exception as exc:
    st.error(f"Could not evaluate this function at the selected point: {exc}")
    st.stop()

if not np.isfinite(f_current):
    st.warning("The function is undefined at the selected point (for example, a denominator may be zero or a logarithm may have an invalid input).")
else:
    interpretation = f"At {x_name} = {current_x:.4g} {x_units}, {dep_name} is approximately {f_current:.5g} {dep_units}."
    st.markdown(f"**Current interpretation:** {interpretation}")
    st.markdown(f"**First derivative:** {sp.latex(d1_expr)}")
    st.latex(sp.Eq(sp.Function("f'")(x), d1_expr))
    st.markdown(f"**Second derivative:**")
    st.latex(sp.Eq(sp.Function("f''")(x), d2_expr))
    st.caption(f"f′ measures the rate of change of {dep_name} per {x_units} ({dep_units}/{x_units}); f″ measures how that rate changes ({dep_units}/{x_units}²). These units are illustrative and depend on your context.")

if pathway in ("Precalculus", "Calculus"):
    st.subheader("Rates of change")
    a_col, b_col = st.columns(2)
    with a_col:
        a = st.number_input("Secant interval start a", value=float(x_min), min_value=float(x_min), max_value=float(x_max), key="secant_a")
    with b_col:
        b = st.number_input("Secant interval end b", value=float(x_max), min_value=float(x_min), max_value=float(x_max), key="secant_b")
    if b <= a:
        st.warning("Choose a secant interval with b greater than a.")
    else:
        fa, fb = float(f_fn(np.array([a]))[0]), float(f_fn(np.array([b]))[0])
        avg_rate = (fb-fa)/(b-a) if np.isfinite(fa) and np.isfinite(fb) else float("nan")
        st.metric("Average rate of change", f"{avg_rate:.5g} {dep_units}/{x_units}" if np.isfinite(avg_rate) else "Undefined on this interval")
        st.caption(f"Secant line through ({a:.4g}, {fa:.4g}) and ({b:.4g}, {fb:.4g}) when both function values are defined.")
    st.subheader("Turning points and concavity")
    st.caption("Numerical scan over the selected interval; classifications are approximations and can miss closely spaced roots or singularities. A zero derivative alone does not prove an extremum, and a zero second derivative alone does not prove an inflection point.")
    grid = np.linspace(x_min, x_max, 1200)
    ygrid, dgrid, ddgrid = f_fn(grid), d1_fn(grid), d2_fn(grid)
    finite = np.isfinite(ygrid) & np.isfinite(dgrid) & np.isfinite(ddgrid)
    def approximate_roots(vals, xs, finite_mask):
        roots = []
        for i in range(len(xs)-1):
            if not finite_mask[i] or not finite_mask[i+1]:
                continue
            if vals[i] == 0:
                roots.append(xs[i])
            elif vals[i] * vals[i+1] < 0:
                roots.append((xs[i]+xs[i+1])/2)
        return [r for j, r in enumerate(roots) if j == 0 or abs(r-roots[j-1]) > (x_max-x_min)/300]
    critical = approximate_roots(dgrid, grid, finite)
    inflect_candidates = approximate_roots(ddgrid, grid, finite)
    rows = []
    for root in critical:
        eps = max((x_max-x_min)/10000, 1e-5)
        before, after = float(d1_fn(np.array([root-eps]))[0]), float(d1_fn(np.array([root+eps]))[0])
        if np.isfinite(before) and np.isfinite(after) and before > 0 and after < 0:
            kind = "local maximum (numerical derivative sign change)"
        elif np.isfinite(before) and np.isfinite(after) and before < 0 and after > 0:
            kind = "local minimum (numerical derivative sign change)"
        else:
            kind = "stationary/critical candidate; no max/min sign change confirmed"
        rows.append({"x": root, "f(x)": float(f_fn(np.array([root]))[0]), "classification": kind})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.write("No critical candidates detected by the numerical scan in this interval.")
    inflect_rows = []
    for root in inflect_candidates:
        eps = max((x_max-x_min)/10000, 1e-5)
        before, after = float(d2_fn(np.array([root-eps]))[0]), float(d2_fn(np.array([root+eps]))[0])
        if np.isfinite(before) and np.isfinite(after) and before*after < 0:
            inflect_rows.append({"x": root, "f(x)": float(f_fn(np.array([root]))[0]), "result": "inflection candidate; concavity changes"})
    if inflect_rows:
        st.dataframe(pd.DataFrame(inflect_rows), use_container_width=True, hide_index=True)
    else:
        st.write("No concavity-changing inflection candidates confirmed by the numerical scan.")
    if pathway == "Calculus":
        st.subheader("Calculus details")
        # Numerical interval extrema include endpoints and sampled valid points.
        valid_idx = np.where(np.isfinite(ygrid))[0]
        if len(valid_idx):
            imin, imax = valid_idx[np.argmin(ygrid[valid_idx])], valid_idx[np.argmax(ygrid[valid_idx])]
            st.write(f"Approximate absolute minimum on sampled valid points: x ≈ {grid[imin]:.5g}, f(x) ≈ {ygrid[imin]:.5g}.")
            st.write(f"Approximate absolute maximum on sampled valid points: x ≈ {grid[imax]:.5g}, f(x) ≈ {ygrid[imax]:.5g}.")
        tangent_slope = d1_current
        if np.isfinite(f_current) and np.isfinite(tangent_slope):
            st.latex(sp.latex(sp.Eq(sp.Symbol("y"), sp.Float(f_current, 6) + sp.Float(tangent_slope, 6)*(x-sp.Float(current_x, 6)))))
            st.caption(f"Tangent line at the selected point: slope ≈ {tangent_slope:.5g} {dep_units}/{x_units}.")
        st.write("Domain restrictions depend on the expression: denominators cannot be zero, logarithm arguments must be positive, and square-root arguments must be nonnegative for real-valued results. The plot below breaks at non-finite values.")
else:
    st.subheader("Algebra 2 concepts")
    finite_y = ygrid = f_fn(np.linspace(x_min, x_max, 400))
    finite_y = finite_y[np.isfinite(finite_y)]
    if len(finite_y):
        st.write(f"On the selected viewing interval, sampled output values range from about {np.min(finite_y):.5g} to {np.max(finite_y):.5g} {dep_units}.")
    st.write(f"End behavior depends on the expression and may differ from this finite viewing interval. The graph shows the selected domain window [{x_min:g}, {x_max:g}].")
    st.write("Average rates of change help describe whether the function rises or falls across an interval.")

# Function plot with broken lines across invalid values
plot_x = np.linspace(x_min, x_max, 1000)
plot_y = f_fn(plot_x)
plot_d1 = d1_fn(plot_x)
plot_d2 = d2_fn(plot_x)
show_f = st.checkbox("Show f(x)", value=True)
show_d1 = st.checkbox("Show f′(x)", value=(pathway == "Calculus"))
show_d2 = st.checkbox("Show f″(x)", value=(pathway == "Calculus"))
fig2, ax2 = plt.subplots(figsize=(10, 5))
if show_f:
    ax2.plot(plot_x, np.where(np.isfinite(plot_y), plot_y, np.nan), label="f(x)", linewidth=2)
if show_d1:
    ax2.plot(plot_x, np.where(np.isfinite(plot_d1), plot_d1, np.nan), label="f′(x)", linestyle="--")
if show_d2:
    ax2.plot(plot_x, np.where(np.isfinite(plot_d2), plot_d2, np.nan), label="f″(x)", linestyle=":")
if np.isfinite(f_current):
    ax2.scatter([current_x], [f_current], marker="o", s=70, label=f"Current point ({current_x:.3g}, {f_current:.3g})", zorder=5)
    if np.isfinite(d1_current):
        tangent_x = np.linspace(max(x_min, current_x-(x_max-x_min)*.2), min(x_max, current_x+(x_max-x_min)*.2), 100)
        tangent_y = f_current + d1_current*(tangent_x-current_x)
        ax2.plot(tangent_x, tangent_y, label="Tangent line", alpha=.75)
ax2.axhline(0, color="black", linewidth=.6, alpha=.4)
ax2.axvline(0, color="black", linewidth=.6, alpha=.4)
ax2.set_xlabel(f"{x_name} ({x_units})")
ax2.set_ylabel(f"{dep_name} / derivatives (units vary)")
ax2.grid(True, alpha=.25)
ax2.legend(fontsize=8)
fig2.tight_layout()
st.pyplot(fig2)

# Contextual visual simulation; scales are explicitly illustrative.
st.subheader("Contextual visual simulation")
visual = st.selectbox("Visual", ["Moving object", "Prosthetic-leg walking cycle", "Tank / concentration", "Population / cell colony", "Thermometer / weather", "Business / economic indicator"])
progress = float(np.clip((current_x-x_min)/(x_max-x_min), 0, 1))
safe_progress = int(progress*100)
visual_cols = st.columns([1, 2])
with visual_cols[0]:
    st.metric("Simulation position", f"{safe_progress}%")
    st.caption("Visual scale is illustrative and normalized to the selected interval; it is not a clinical, engineering, or operational simulator.")
with visual_cols[1]:
    if visual == "Moving object":
        st.markdown(f"**Object** → position along the track: {safe_progress}%")
        st.progress(safe_progress)
    elif visual == "Prosthetic-leg walking cycle":
        phase = 2*math.pi*progress
        knee = int(45 + 35*math.sin(phase))
        st.markdown(f"**Walking-cycle phase:** {safe_progress}% · illustrative knee angle: {knee}°")
        st.progress(safe_progress)
        st.caption("This is a simplified classroom animation indicator, not a model of an individual's gait or prosthetic fitting.")
    elif visual == "Tank / concentration":
        level = int(np.clip(abs(f_current)/(1+abs(f_current))*100, 0, 100)) if np.isfinite(f_current) else 0
        st.markdown(f"**Illustrative level:** {level}%")
        st.progress(level)
    elif visual == "Population / cell colony":
        count = int(10 + 90*progress)
        st.markdown(f"**Illustrative colony index:** {count}")
        st.progress(safe_progress)
    elif visual == "Thermometer / weather":
        temp = -10 + 50*progress
        st.markdown(f"**Illustrative temperature scale:** {temp:.1f} °C")
        st.progress(safe_progress)
    else:
        st.markdown(f"**Illustrative business/economic index:** {100*progress:.1f}")
        st.progress(safe_progress)

# Exports use current selection, working data, model, and calculation results.
summary = {
    "Country": country_label, "Country code": source_country, "Indicator": indicator_label,
    "Indicator code": indicator_code, "Units": units, "Definition": definition,
    "Source URL": source_url, "Retrieved": retrieval_date, "Source coverage": coverage,
    "Source observation count": len(original_data), "Working observation count": len(clean),
    "Degree": degree, "Base year": base_year, "Equation": equation, "R-squared (in-sample)": r2,
    "MAE (in-sample)": mae, "Mean": float(clean.Value.mean()),
    "Data status": "User-modified scenario" if st.session_state.user_modified else "Source data",
    "Mathematics pathway": pathway, "Context": context, "Function": str(expr),
}
single_df = pd.DataFrame(st.session_state.single_results)
batch_df = pd.DataFrame(st.session_state.batch_results)
st.divider()
st.header("Export current analysis")
csv_bytes = clean.to_csv(index=False).encode("utf-8")
excel_bytes = make_excel(summary, clean, single_df, batch_df)
chart_buffer = io.BytesIO()
fig.savefig(chart_buffer, format="png", dpi=150, bbox_inches="tight")
pdf_bytes = make_pdf(summary, equation, f"In-sample R² = {r2:.6g}; in-sample MAE = {mae:.6g} {units}; observations = {len(clean)}.", chart_buffer.getvalue(), clean, single_df, batch_df)
download_cols = st.columns(3)
download_cols[0].download_button("Download working data CSV", csv_bytes, file_name="working_data.csv", mime="text/csv", use_container_width=True)
download_cols[1].download_button("Download Excel workbook", excel_bytes, file_name="hispanic_heritage_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
download_cols[2].download_button("Download PDF report", pdf_bytes, file_name="hispanic_heritage_report.pdf", mime="application/pdf", use_container_width=True)

st.divider()
st.subheader("Source and mathematical limitations")
st.markdown(f"- World Bank indicator API: [{indicator_code}]({source_url})\n- Indicator definitions and units are shown above; null source years are omitted rather than filled with fabricated values.\n- A polynomial is a descriptive approximation. High-degree fits can oscillate and extrapolations can become implausible; predictions are not clipped.\n- R² and MAE above are in-sample. The optional chronological holdout score is separate. Neither proves causal relationships.\n- Numerical calculus analysis uses a finite scan and is approximate. It can miss roots, close features, or singularities; verify important results analytically.")
st.caption("For classroom use. Verify important conclusions against the cited source and the mathematical assumptions of the selected model.")
