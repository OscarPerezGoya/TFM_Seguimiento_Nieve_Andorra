from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

import rasterio
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from rasterio.warp import Resampling as RioResampling

import matplotlib.pyplot as plt

from shiny import App, ui, render, reactive

import folium
from folium import plugins

import geopandas as gpd
import shapely


# RUTAS
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MAPS_DIR = BASE_DIR / "mapas_nieve"

CSV_MAPAS = DATA_DIR / "estadisticas_mapas_nieve.csv"
CSV_AGG = DATA_DIR / "estadisticas_agregadas.csv"
ANDORRA_SHP = DATA_DIR / "Frontera_Andorra" / "Frontera_Andorra.shp"


# WKT CRS (tuyo)
NTF_Lambert_Sud_WKT = (
    'PROJCS["NTF_Paris_Lambert_Sud_France",'
    'GEOGCS["GCS_NTF_Paris",DATUM["D_NTF",SPHEROID["Clarke_1880_IGN",6378249.2,293.46602]],'
    'PRIMEM["Paris",2.33722917],UNIT["grad",0.01570796326794897]],'
    'PROJECTION["Lambert_Conformal_Conic"],'
    'PARAMETER["latitude_of_origin",49],PARAMETER["central_meridian",0],'
    'PARAMETER["scale_factor",0.999877499],PARAMETER["false_easting",600000],'
    'PARAMETER["false_northing",200000],UNIT["Meter",1],PARAMETER["standard_parallel_1",49]]'
)


# Utils
def _safe_read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"No existe el CSV: {p}")
    return pd.read_csv(p)

def _month_name_es(m: int) -> str:
    names = {
        1:"enero",2:"febrero",3:"marzo",4:"abril",5:"mayo",6:"junio",
        7:"julio",8:"agosto",9:"septiembre",10:"octubre",11:"noviembre",12:"diciembre"
    }
    return names.get(m, str(m))

def _exists_path(x) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and np.isnan(x):
        return False
    name = Path(str(x)).name
    return (MAPS_DIR / name).exists()

def normalize_season(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    m = re.fullmatch(r"(\d{4})\s*[-/]\s*(\d{4})", s)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return s

def fmt_satellite(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "NA"
    ss = str(s).strip()
    m = re.match(r"LANDSAT\s*([0-9]+)", ss, flags=re.IGNORECASE)
    if m:
        return f"Landsat-{m.group(1)}"
    m = re.match(r"SENTINEL\s*([0-9A-Z\-]+)", ss, flags=re.IGNORECASE)
    if m:
        return f"Sentinel-{m.group(1)}"
    return ss

def _make_valid_geom(g):
    if hasattr(shapely, "make_valid"):
        g = shapely.make_valid(g)
    g = g.buffer(0)
    return g

def load_shp_to_4326(shp_path: Path, wkt: str):
    if not shp_path.exists():
        return None, f"No existe el shapefile: {shp_path}"
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(wkt)
    gdf = gdf.to_crs(epsg=4326)
    gdf["geometry"] = gdf["geometry"].apply(_make_valid_geom)
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geometry.is_valid].copy()
    return gdf, None


_OVERLAY_CACHE = {}

def _safe_to_4326_raster(src_path: Path, max_size: int = 700, resampling=RioResampling.nearest):
    key = (str(src_path), max_size, str(resampling))
    if key in _OVERLAY_CACHE:
        return _OVERLAY_CACHE[key]

    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError(f"El raster no tiene CRS: {src_path}")

        dst_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        scale = max(width / max_size, height / max_size, 1.0)
        dst_width = int(width / scale)
        dst_height = int(height / scale)

        transform2, width2, height2 = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            dst_width=dst_width, dst_height=dst_height
        )

        dst = np.zeros((height2, width2), dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform2,
            dst_crs=dst_crs,
            resampling=resampling,
            num_threads=2
        )

        b = transform_bounds(src.crs, dst_crs, *src.bounds, densify_pts=21)
        west, south, east, north = b[0], b[1], b[2], b[3]
        bounds = [[south, west], [north, east]]

        _OVERLAY_CACHE[key] = (dst, bounds)
        return dst, bounds

def class_raster_to_rgba(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.int16)
    h, w = a.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[a == 1] = [140, 81, 10, 255]
    rgba[a == 2] = [255, 255, 255, 255]
    rgba[a == 3] = [220, 0, 0, 255]
    return rgba


# Colores fijos por cobertura
COVER_COLORS = {
    "arbolado denso": "#006400",
    "arbolado claro": "#90EE90",
    "prados": "#8B4513",
    "canchales": "#FF0000",
    "matorrales": "#FFA500",
    "roquedo": "#555555",
    "roquedos": "#555555",
    "cultivo": "#FFD700",
    "cultivos": "#FFD700",
    "zonas urbanas": "#000000",
    "zonas desnudas": "#D3D3D3",
    "vias comunicacion": "#8A2BE2",
    "vias de comunicacion": "#8A2BE2",
    "aguas": "#1E90FF",
    "agua": "#1E90FF",
    "zonas deportivas": "#FF00FF",
}

def coverage_color(label: str, fallback=None):
    if label is None:
        return fallback
    lab = str(label).strip().lower().replace("_", " ")
    lab = re.sub(r"\s+", " ", lab)
    lab = lab.replace("vías", "vias")

    if lab in COVER_COLORS:
        return COVER_COLORS[lab]

    if "arbolado" in lab and "denso" in lab:
        return COVER_COLORS["arbolado denso"]
    if "arbolado" in lab and "claro" in lab:
        return COVER_COLORS["arbolado claro"]
    if "prado" in lab:
        return COVER_COLORS["prados"]
    if "canchal" in lab:
        return COVER_COLORS["canchales"]
    if "matorral" in lab:
        return COVER_COLORS["matorrales"]
    if "roqued" in lab:
        return COVER_COLORS["roquedo"]
    if "cultiv" in lab:
        return COVER_COLORS["cultivo"]
    if "urbana" in lab or "urban" in lab:
        return COVER_COLORS["zonas urbanas"]
    if "desnuda" in lab or "desnud" in lab:
        return COVER_COLORS["zonas desnudas"]
    if "via" in lab or "comunicacion" in lab:
        return COVER_COLORS["vias comunicacion"]
    if "agua" in lab:
        return COVER_COLORS["aguas"]
    if "deport" in lab:
        return COVER_COLORS["zonas deportivas"]

    return fallback


# Cargo los datos
df_mapas_raw = _safe_read_csv(CSV_MAPAS)
df_agreg = _safe_read_csv(CSV_AGG)

if "Fecha_dt" in df_mapas_raw.columns:
    df_mapas_raw["_Fecha_dt_norm"] = pd.to_datetime(df_mapas_raw["Fecha_dt"], errors="coerce")
elif "Fecha" in df_mapas_raw.columns:
    df_mapas_raw["_Fecha_dt_norm"] = pd.to_datetime(df_mapas_raw["Fecha"], errors="coerce")
else:
    raise ValueError("estadisticas_mapas_nieve.csv debe tener 'Fecha_dt' o 'Fecha'.")

df_mapas_raw["_Dia_raw"] = df_mapas_raw["_Fecha_dt_norm"].dt.date

df_mapas = df_mapas_raw.copy()
df_mapas["Fecha_dt"] = df_mapas_raw["_Fecha_dt_norm"]

for col in ["Nieve_pct_Andorra", "Nieve_km2", "Snowline_final_p20_m", "Nubes_MODIS_%"]:
    if col not in df_mapas.columns:
        df_mapas[col] = np.nan
    df_mapas[col] = pd.to_numeric(df_mapas[col], errors="coerce")

if "Ruta_tif" not in df_mapas.columns:
    raise ValueError("Falta la columna 'Ruta_tif' en estadisticas_mapas_nieve.csv")

df_mapas["Ruta_tif"] = df_mapas["Ruta_tif"].astype(str)
df_mapas.loc[df_mapas["Ruta_tif"].isin(["nan", "None", ""]), "Ruta_tif"] = np.nan

if "Temporada" in df_mapas.columns:
    df_mapas["Temporada"] = df_mapas["Temporada"].apply(normalize_season)
if "Temporada" in df_agreg.columns:
    df_agreg["Temporada"] = df_agreg["Temporada"].apply(normalize_season)

df_mapas["Dia"] = pd.to_datetime(df_mapas["Fecha_dt"], errors="coerce").dt.date

df_mapas_valid = df_mapas.dropna(subset=["Fecha_dt", "Ruta_tif"]).copy()
df_mapas_valid = df_mapas_valid[df_mapas_valid["Ruta_tif"].apply(_exists_path)].copy()
df_mapas_valid["Dia"] = pd.to_datetime(df_mapas_valid["Fecha_dt"]).dt.date
if df_mapas_valid.empty:
    raise ValueError("No hay filas válidas tras filtrar por Fecha_dt y Ruta_tif existente.")

cal = df_mapas_valid.dropna(subset=["Fecha_dt"]).copy()
cal["Year"]  = pd.to_datetime(cal["Fecha_dt"]).dt.year.astype(int)
cal["Month"] = pd.to_datetime(cal["Fecha_dt"]).dt.month.astype(int)
cal["Day"]   = pd.to_datetime(cal["Fecha_dt"]).dt.day.astype(int)
cal_days = cal[["Year","Month","Day"]].drop_duplicates().sort_values(["Year","Month","Day"])
available_years = sorted(cal_days["Year"].unique().tolist())

def month_choices_for_year(y: int):
    months = sorted(cal_days.loc[cal_days["Year"] == y, "Month"].unique().tolist())
    return {str(m): _month_name_es(m) for m in months}, (str(months[0]) if months else None)

def day_choices_for_year_month(y: int, m: int):
    days = sorted(cal_days.loc[(cal_days["Year"] == y) & (cal_days["Month"] == m), "Day"].unique().tolist())
    return {str(d): f"{d:02d}" for d in days}, (str(days[0]) if days else None)

_default_year = int(available_years[0])
_month_choices0, _month_selected0 = month_choices_for_year(_default_year)
if _month_selected0 is None:
    _day_choices0, _day_selected0 = {}, None
else:
    _day_choices0, _day_selected0 = day_choices_for_year_month(_default_year, int(_month_selected0))

season_choices = sorted(df_agreg["Temporada"].dropna().unique().tolist()) if "Temporada" in df_agreg.columns else []

missing_msgs = []
andorra_gdf, err = load_shp_to_4326(ANDORRA_SHP, NTF_Lambert_Sud_WKT)
if err:
    missing_msgs.append(err)

VAR_LABELS = {
    "Nieve_pct_Andorra": "% nieve s/ Andorra",
    "Snowline_final_p20_m": "Cota de nieve",
    "Dias_nieve_suelo": "Duración temporada (días nieve suelo)",
    "Gantt_inicio_fin": "Inicio-Fin robustos (Gantt)",
    "Heatmap_snowline_mensual": "Heatmap snowline mensual",
}


# CSS
KPI_CSS = ui.tags.style("""
.kpi-wrap { display: grid; grid-template-columns: 1fr; gap: 10px; }

.kpi-card {
  background: #fff;
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 14px;
  padding: 10px 12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}
.kpi-top { display:flex; align-items:center; justify-content:space-between; gap: 10px; }
.kpi-title { font-size: 12px; color: rgba(0,0,0,0.65); font-weight: 600; }
.kpi-icon { font-size: 18px; opacity: 0.9; }
.kpi-value { font-size: 26px; font-weight: 800; margin-top: 2px; line-height: 1.05; }
.kpi-sub { font-size: 12px; color: rgba(0,0,0,0.55); margin-top: 2px; }

.kpi-chip-row {
  display:flex;
  flex-wrap: nowrap;
  gap: 10px;
  margin: 0;
}
.kpi-chip {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 10px 14px;
  border-radius: 14px;
  background: #fff;
  border: 1px solid rgba(0,0,0,0.08);
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
  font-size: 13px;
  color: rgba(0,0,0,0.72);
  white-space: nowrap;
}

.kpi-progress {
  width: 100%;
  height: 10px;
  border-radius: 999px;
  background: rgba(0,0,0,0.08);
  overflow: hidden;
  margin-top: 8px;
}
.kpi-progress > div {
  height: 100%;
  border-radius: 999px;
  background: #2563eb;
}

.sidebar h5 { margin-top: 6px; margin-bottom: 8px; }
.navbar .navbar-brand,
.navbar-brand {
  font-size: 34px !important;
  font-weight: 900 !important;
  letter-spacing: 0.2px;
}

.navbar-nav .nav-link {
  border-radius: 999px;
  padding: 8px 16px !important;
  margin: 6px 6px;
  border: 1px solid rgba(0,0,0,0.12);
  background: rgba(255,255,255,0.70);
  font-weight: 800;
  color: rgba(0,0,0,0.72) !important;
}

.navbar-nav .nav-link:hover {
  background: rgba(255,255,255,0.95);
  border-color: rgba(0,0,0,0.18);
}

.navbar-nav .nav-link.active {
  background: #2563eb !important;
  border-color: #2563eb !important;
  color: #ffffff !important;
}

.navbar-nav .nav-link.active:hover {
  background: #1d4ed8 !important;
  border-color: #1d4ed8 !important;
}

.bar-card .card-header{
  padding-top: 8px !important;
  padding-bottom: 6px !important;
}

.bar-card .card-body{
  padding-top: 8px !important;
  padding-bottom: 2px !important;
}
""")


# UI
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.panel_conditional(
            "input.tabs === 'Día'",
            ui.h4("Selector de día"),
            ui.input_selectize("year", "Año", choices=[str(y) for y in available_years], selected=str(_default_year)),
            ui.input_selectize("month", "Mes", choices=_month_choices0, selected=_month_selected0),
            ui.input_selectize("day", "Día", choices=_day_choices0, selected=_day_selected0),
            ui.h5("Resumen del día"),
            ui.output_ui("day_summary_sidebar"),
            ui.output_ui("missing_layers_note"),
            width=360,
        ),
        ui.panel_conditional(
            "input.tabs === 'Temporadas'",
            ui.h4("Selector de temporadas"),
            ui.input_selectize("seasons", "Temporadas (multi-selección)", choices=season_choices, selected=None, multiple=True),
            ui.input_selectize(
                "var_ts", "Variable a graficar",
                choices={k: v for k, v in VAR_LABELS.items()},
                selected="Nieve_pct_Andorra",
            ),
            width=360,
        ),
    ),

    KPI_CSS,

    ui.navset_bar(
        ui.nav_panel(
            "Día",
            ui.layout_columns(
                ui.card(
                    ui.card_header(
                        ui.div(
                            {
                                "style": (
                                    "display:flex; align-items:center; justify-content:space-between; "
                                    "gap:12px; flex-wrap:wrap;"
                                )
                            },
                            ui.tags.div(
                                "Mapa de cubierta de Nieve",
                                style="font-size: 22px; font-weight: 800; padding: 4px 0;"
                            ),
                            ui.output_ui("map_header_chips"),
                        )
                    ),
                    ui.div(
                        {"style": "height: 340px; overflow: hidden;"},
                        ui.output_ui("map_leaflet"),
                    ),
                    full_screen=True,
                ),
                col_widths=(12,),
            ),
            ui.layout_columns(
                ui.div(
                    {"class": "bar-card"},
                    ui.card(
                        ui.card_header(
                            ui.tags.div(
                                "Porcentaje de nieve según cubiertas del suelo",
                                style="font-size: 22px; font-weight: 800; padding: 4px 0;"
                            )
                        ),
                        ui.output_plot("day_cover_bar", height="200px"),
                    ),
                ),
                col_widths=(12,),
            ),
        ),

        ui.nav_panel(
            "Temporadas",
            ui.layout_columns(
                ui.card(
                    ui.output_plot("season_plot", height="600px"),
                    full_screen=True
                ),
                col_widths=(12,),
            ),
        ),
        title="Dashboard Nieve Andorra",
        id="tabs",
    ),
)


# SERVER
def server(input, output, session):

    @reactive.effect
    def _update_month_choices():
        y = int(input.year())
        mchoices, msel = month_choices_for_year(y)
        ui.update_selectize("month", choices=mchoices, selected=msel)
        if msel is None:
            ui.update_selectize("day", choices={}, selected=None)
        else:
            dchoices, dsel = day_choices_for_year_month(y, int(msel))
            ui.update_selectize("day", choices=dchoices, selected=dsel)

    @reactive.effect
    def _update_day_choices():
        mo = input.month()
        if mo is None:
            ui.update_selectize("day", choices={}, selected=None)
            return
        y = int(input.year())
        m = int(mo)
        dchoices, dsel = day_choices_for_year_month(y, m)
        ui.update_selectize("day", choices=dchoices, selected=dsel)

    @reactive.calc
    def selected_date():
        mo = input.month()
        da = input.day()
        if mo is None or da is None:
            return None
        dt = pd.to_datetime(
            f"{int(input.year())}-{int(mo):02d}-{int(da):02d}",
            errors="coerce"
        )
        return dt.date() if pd.notna(dt) else None

    @reactive.calc
    def rows_for_selected_day_valid():
        d = selected_date()
        if d is None:
            return df_mapas_valid.iloc[0:0].copy()
        return df_mapas_valid[df_mapas_valid["Dia"] == d].copy()

    @reactive.calc
    def rows_for_selected_day_csv():
        d = selected_date()
        if d is None:
            return df_mapas_raw.iloc[0:0].copy()
        return df_mapas_raw[df_mapas_raw["_Dia_raw"] == d].copy()

    def _best_row_for_day(g: pd.DataFrame) -> pd.Series:
        if "Nubes_MODIS_%" in g.columns:
            tmp = g.copy()
            tmp["Nubes_MODIS_%"] = pd.to_numeric(tmp["Nubes_MODIS_%"], errors="coerce")
            if tmp["Nubes_MODIS_%"].notna().any():
                return tmp.sort_values("Nubes_MODIS_%", ascending=True).iloc[0]
        return g.iloc[0]

    @reactive.calc
    def best_row_selected_day():
        g = rows_for_selected_day_valid()
        if g.empty:
            return None
        return _best_row_for_day(g)

    @reactive.calc
    def best_row_selected_day_csv():
        g = rows_for_selected_day_csv()
        if g.empty:
            return None
        return _best_row_for_day(g)

    @output
    @render.ui
    def map_header_chips():
        row = best_row_selected_day()
        if row is None:
            return ui.HTML("")
        fecha = ""
        if "Fecha_dt" in row.index and pd.notna(row.get("Fecha_dt")):
            dt = pd.to_datetime(row["Fecha_dt"], errors="coerce")
            fecha = str(dt.date()) if pd.notna(dt) else str(row.get("Fecha_dt"))
        sat = fmt_satellite(row.get("Satelite", np.nan))

        html = f"""
        <div class="kpi-chip-row">
          <span class="kpi-chip">🛰️ <b>{sat}</b></span>
          <span class="kpi-chip">📅 <b>{fecha}</b></span>
        </div>
        """
        return ui.HTML(html)

    @output
    @render.ui
    def missing_layers_note():
        if not missing_msgs:
            return ui.HTML("")
        txt = "<br>".join(missing_msgs)
        return ui.HTML(f"<div style='margin-top:8px; font-size:12px; color:#b45309'>{txt}</div>")

    @output
    @render.ui
    def day_summary_sidebar():
        row = best_row_selected_day()
        if row is None:
            return ui.HTML("Sin selección válida.")

        nieve_pct = row.get("Nieve_pct_Andorra", np.nan)
        snow_km2  = row.get("Nieve_km2", np.nan)
        sl        = row.get("Snowline_final_p20_m", np.nan)

        nieve_pct_txt = f"{int(round(float(nieve_pct)))}%" if np.isfinite(nieve_pct) else "NA"
        snow_km2_txt  = f"{int(round(float(snow_km2)))}" if np.isfinite(snow_km2) else "NA"
        sl_txt        = f"{sl:.0f}" if np.isfinite(sl) else "NA"

        p = float(nieve_pct) if np.isfinite(nieve_pct) else 0.0
        p = max(0.0, min(100.0, p))
        prog_html = f"""<div class="kpi-progress"><div style="width:{p:.2f}%"></div></div>"""

        html = f"""
        <div class="kpi-wrap">
          <div class="kpi-card">
            <div class="kpi-top">
              <div class="kpi-title">% Cubierta Nieve Andorra</div>
              <div class="kpi-icon">❄️</div>
            </div>
            <div class="kpi-value">{nieve_pct_txt}</div>
            <div class="kpi-sub">Porcentaje sobre Andorra</div>
            {prog_html}
          </div>

          <div class="kpi-card">
            <div class="kpi-top">
              <div class="kpi-title">Superficie nieve</div>
              <div class="kpi-icon">🗺️</div>
            </div>
            <div class="kpi-value">{snow_km2_txt} <span style="font-size:14px;font-weight:700;color:rgba(0,0,0,0.6)">km²</span></div>
            <div class="kpi-sub">Área nevada estimada</div>
          </div>

          <div class="kpi-card">
            <div class="kpi-top">
              <div class="kpi-title">Cota nieve</div>
              <div class="kpi-icon">⛰️</div>
            </div>
            <div class="kpi-value">{sl_txt} <span style="font-size:14px;font-weight:700;color:rgba(0,0,0,0.6)">m</span></div>
            <div class="kpi-sub">Snowline (p20)</div>
          </div>
        </div>
        """
        return ui.HTML(html)

    @output
    @render.ui
    def map_leaflet():
        row = best_row_selected_day()
        if row is None:
            return ui.HTML("<div style='padding:16px'>Sin imagen para la fecha seleccionada.</div>")

        tif_path = MAPS_DIR / Path(str(row["Ruta_tif"])).name
        if not tif_path.exists():
            return ui.HTML(f"<div style='padding:16px'>No existe el TIFF: {tif_path}</div>")

        fig = folium.Figure(width="100%", height=340)
        m = folium.Map(location=[42.55, 1.60], zoom_start=10, tiles=None, control_scale=True)
        m.add_to(fig)

        folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True, show=True).add_to(m)

        arr4326, bounds = _safe_to_4326_raster(tif_path, max_size=700, resampling=RioResampling.nearest)
        rgba = class_raster_to_rgba(arr4326)
        folium.raster_layers.ImageOverlay(
            image=rgba / 255.0,
            bounds=bounds,
            name="Clasificación",
            opacity=1.0,
            interactive=True,
            zindex=6,
            show=True,
        ).add_to(m)
        m.fit_bounds(bounds)

        if andorra_gdf is not None and len(andorra_gdf) > 0:
            folium.GeoJson(
                andorra_gdf.__geo_interface__,
                name="Límite Andorra",
                style_function=lambda feat: {"color": "yellow", "weight": 3, "fillOpacity": 0.0},
                show=True,
            ).add_to(m)

        plugins.Fullscreen(position="topleft").add_to(m)
        plugins.MeasureControl(position="topleft", primary_length_unit="kilometers").add_to(m)
        plugins.MousePosition(position="bottomleft").add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        return ui.HTML(fig._repr_html_())

    @output
    @render.plot
    def day_cover_bar():
        row = best_row_selected_day_csv()
        fig, ax = plt.subplots(figsize=(12, 2.1))

        if row is None:
            ax.text(0.5, 0.5, "Sin datos para el día seleccionado.", ha="center", va="center")
            ax.axis("off")
            return fig

        s = row.copy()
        pat = re.compile(r"^Nieve_(.+)_pct$")
        cols = [c for c in s.index if pat.match(str(c))]
        if not cols:
            ax.text(0.5, 0.5, "No existen columnas tipo Nieve_<Cobertura>_pct en el CSV.", ha="center", va="center")
            ax.axis("off")
            return fig

        vals, labels = [], []
        for c in cols:
            v = pd.to_numeric(s.get(c), errors="coerce")
            if pd.notna(v) and float(v) != 0.0:
                m = pat.match(str(c))
                name = (m.group(1) if m else str(c)).replace("_", " ")
                name = re.sub(r"\s+", " ", name).strip()
                labels.append(name)
                vals.append(float(v))

        if not vals:
            ax.text(0.5, 0.5, "Valores Nieve_<Cobertura>_pct vacíos o 0 para este día.", ha="center", va="center")
            ax.axis("off")
            return fig

        order = np.argsort(vals)[::-1]
        vals = [vals[i] for i in order]
        labels = [labels[i] for i in order]

        total = sum(vals)
        x_max = 100.0 if total <= 100.0 else total

        left = 0.0
        for lab, v in zip(labels, vals):
            col = coverage_color(lab, fallback="#808080")
            ax.barh([0], [v], left=left, height=0.55, label=lab, color=col)
            if v >= (0.06 * x_max):
                ax.text(left + v/2, 0, f"{v:.1f}%", ha="center", va="center", fontsize=9, color="white")
            left += v

        ax.set_xlim(0, x_max)
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.grid(axis="x", alpha=0.25)

        ax.text(0.0, 1.02, "%", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=11, fontweight="bold", color="black")

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False)

        fig.subplots_adjust(left=0.03, right=0.98, top=0.86, bottom=0.28)
        return fig

    @reactive.calc
    def selected_seasons():
        ss = input.seasons()
        return list(ss) if ss is not None else []

    @reactive.calc
    def daily_series_for_seasons() -> pd.DataFrame:
        ss = selected_seasons()
        if not ss or "Temporada" not in df_mapas.columns:
            return df_mapas.iloc[0:0].copy()

        tmp = df_mapas[df_mapas["Temporada"].isin(ss)].copy()
        if tmp.empty:
            return tmp

        tmp["Dia_norm"] = pd.to_datetime(tmp["Fecha_dt"], errors="coerce").dt.normalize()
        tmp = tmp.dropna(subset=["Dia_norm"])

        daily = (
            tmp.groupby(["Temporada", "Dia_norm"], as_index=False)
               .agg(
                   Nieve_pct_Andorra=("Nieve_pct_Andorra", "max"),
                   Snowline_final_p20_m=("Snowline_final_p20_m", "mean"),
               )
               .sort_values(["Temporada", "Dia_norm"])
        )

        season_start_month = 8
        daily["m"] = daily["Dia_norm"].dt.month.astype(int)
        daily["d"] = daily["Dia_norm"].dt.day.astype(int)

        base_year = 1999
        daily["dummy_year"] = np.where(daily["m"] >= season_start_month, base_year, base_year + 1).astype(int)

        month_first = pd.to_datetime(dict(year=daily["dummy_year"], month=daily["m"], day=1), errors="coerce")
        last_day = (month_first + pd.offsets.MonthEnd(0)).dt.day
        bad = daily["d"] > last_day
        daily = daily.loc[~bad].copy()

        daily["x_season"] = month_first.loc[~bad].reset_index(drop=True) + pd.to_timedelta(daily["d"].values - 1, unit="D")
        return daily

    @output
    @render.plot
    def season_plot():
        var = input.var_ts()
        ss = selected_seasons()

        fig, ax = plt.subplots(figsize=(14, 9))

        if not ss:
            ax.text(0.5, 0.5, "Selecciona una o varias temporadas.", ha="center", va="center")
            ax.axis("off")
            return fig

        if var in ("Nieve_pct_Andorra", "Snowline_final_p20_m"):
            daily = daily_series_for_seasons()

            if daily.empty or var not in daily.columns:
                ax.text(0.5, 0.5, "Selecciona una o varias temporadas con datos para esa variable.", ha="center", va="center")
                ax.axis("off")
                return fig

            for temporada, g in daily.groupby("Temporada"):
                ax.plot(g["x_season"], g[var], label=str(temporada))

            nice = VAR_LABELS.get(var, var)
            ax.set_title(nice)
            ax.set_xlabel("Día-Mes")

            if var == "Nieve_pct_Andorra":
                ax.set_ylabel("% nieve s/ Andorra")
            else:
                ax.set_ylabel("Cota de nieve (m)")
                ax.set_ylim(800, 3250)

            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")

            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
            fig.autofmt_xdate(rotation=45)

            fig.subplots_adjust(left=0.14, right=0.98, bottom=0.18, top=0.90)
            return fig

        g = df_agreg[df_agreg["Temporada"].isin(ss)].copy()
        if g.empty:
            ax.text(0.5, 0.5, "No hay agregados para esa selección.", ha="center", va="center")
            ax.axis("off")
            return fig

        g["Temporada"] = g["Temporada"].astype(str)
        g = g.set_index("Temporada").loc[ss].reset_index()

        if var == "Dias_nieve_suelo":
            if "Dias_nieve_suelo" not in g.columns:
                ax.text(0.5, 0.5, "No existe la columna Dias_nieve_suelo en agregados.", ha="center", va="center")
                ax.axis("off")
                return fig

            y = pd.to_numeric(g["Dias_nieve_suelo"], errors="coerce").to_numpy(dtype=float)
            x = np.arange(len(ss))
            ax.bar(x, y)
            ax.set_xticks(x)
            ax.set_xticklabels(ss, rotation=45, ha="right")
            ax.set_title(VAR_LABELS.get(var, var))
            ax.set_ylabel("Días")
            ax.grid(axis="y", alpha=0.25)
            fig.subplots_adjust(left=0.10, right=0.98, bottom=0.30, top=0.90)
            return fig

        if var == "Gantt_inicio_fin":
            if "Inicio_robusto_ge15" not in g.columns or "Fin_robusto_ge15" not in g.columns:
                ax.text(0.5, 0.5, "Faltan Inicio_robusto_ge15 o Fin_robusto_ge15 en agregados.", ha="center", va="center")
                ax.axis("off")
                return fig

            start = pd.to_datetime(g["Inicio_robusto_ge15"], errors="coerce")
            end = pd.to_datetime(g["Fin_robusto_ge15"], errors="coerce")

            season_start_month = 8
            base_year = 1999

            def _map_to_dummy(dt: pd.Timestamp) -> pd.Timestamp:
                if pd.isna(dt):
                    return pd.NaT
                y = base_year if dt.month >= season_start_month else base_year + 1
                md_first = pd.Timestamp(year=y, month=dt.month, day=1)
                last_day = (md_first + pd.offsets.MonthEnd(0)).day
                d = min(int(dt.day), int(last_day))
                return pd.Timestamp(year=y, month=dt.month, day=d)

            xs = start.apply(_map_to_dummy)
            xe = end.apply(_map_to_dummy)

            y = np.arange(len(ss))
            for i in range(len(ss)):
                if pd.notna(xs.iloc[i]) and pd.notna(xe.iloc[i]):
                    ax.hlines(y[i], xs.iloc[i], xe.iloc[i], linewidth=6)
                    ax.plot([xs.iloc[i], xe.iloc[i]], [y[i], y[i]], marker="o", linewidth=0)

            ax.set_yticks(y)
            ax.set_yticklabels(ss)
            ax.set_title(VAR_LABELS.get(var, var))
            ax.set_xlabel("Día-Mes")
            ax.grid(axis="x", alpha=0.25)

            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
            fig.autofmt_xdate(rotation=45)

            fig.subplots_adjust(left=0.20, right=0.98, bottom=0.18, top=0.90)
            return fig

        if var == "Heatmap_snowline_mensual":
            month_map = [
                ("octubre", "Snowline_media_mensual_octubre"),
                ("noviembre", "Snowline_media_mensual_noviembre"),
                ("diciembre", "Snowline_media_mensual_diciembre"),
                ("enero", "Snowline_media_mensual_enero"),
                ("febrero", "Snowline_media_mensual_febrero"),
                ("marzo", "Snowline_media_mensual_marzo"),
                ("abril", "Snowline_media_mensual_abril"),
                ("mayo", "Snowline_media_mensual_mayo"),
                ("junio", "Snowline_media_mensual_junio"),
            ]
            cols = [c for _, c in month_map]
            if any(c not in g.columns for c in cols):
                ax.text(0.5, 0.5, "Faltan columnas Snowline_media_mensual_* en agregados.", ha="center", va="center")
                ax.axis("off")
                return fig

            M = g[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

            im = ax.imshow(M, aspect="auto")
            ax.set_title(VAR_LABELS.get(var, var))
            ax.set_yticks(np.arange(len(ss)))
            ax.set_yticklabels(ss)
            ax.set_xticks(np.arange(len(month_map)))
            ax.set_xticklabels([m for m, _ in month_map], rotation=45, ha="right")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig.subplots_adjust(left=0.20, right=0.95, bottom=0.30, top=0.90)
            return fig

        ax.text(0.5, 0.5, "Opción no soportada.", ha="center", va="center")
        ax.axis("off")
        return fig



app = App(app_ui, server)
