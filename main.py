import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Archivos
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
ARCHIVO_DEPTOS = BASE / "yerba-mate-1970-2024.csv"
ARCHIVO_NACIONAL = BASE / "yerba-mate-1970-2024-anual.csv"
# Cache local de los centroides: evita pegarle a la API de georef en cada arranque
ARCHIVO_CENTROIDES = BASE / "centroides_departamentos.csv"

# ---------------------------------------------------------------------------
# Paleta: Argentina (celeste, blanco, sol de mayo) + verdes de la yerba
# ---------------------------------------------------------------------------
CELESTE = "#74ACDF"
AZUL_TEXTO = "#0B2A4A"
FONDO_SUAVE = "#EAF3FB"
SOL = "#F6B40E"

# Colores fijos: color 1 = sembrado / serie principal, color 2 = cosechado / serie secundaria.
COLOR_1 = "#2E7D32"
COLOR_2 = "#9CCC65"
EXTRAS = [CELESTE, SOL, "#1B5E20", "#9CCC65", "#1F5F99", "#8D6E63"]


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def obtener_centroides(ids):
    """Devuelve un DataFrame departamento_id, lat, lon. Usa cache local si existe."""
    if ARCHIVO_CENTROIDES.exists():
        cache = pd.read_csv(ARCHIVO_CENTROIDES, dtype={"departamento_id": str})
        if set(ids).issubset(set(cache["departamento_id"])):
            return cache

    # Pedido multiple de IDs: al repetirse los pedimos de una y mergeamos los centroides.
    url = "https://apis.datos.gob.ar/georef/api/departamentos"
    payload = {"departamentos": [{"id": i, "campos": "id,centroide"} for i in ids]}
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    # Nota: EL LIMITE ES 5000, si eventualmente agregan mas esto se rompe fijo.

    # Dataset ID, LATITUD, LONGITUD
    filas = []
    for r in resp.json()["resultados"]:
        for d in r["departamentos"]:
            filas.append({
                "departamento_id": d["id"],
                "lat": d["centroide"]["lat"],
                "lon": d["centroide"]["lon"],
            })
    coords = pd.DataFrame(filas)
    coords.to_csv(ARCHIVO_CENTROIDES, index=False)
    return coords


def preparar_dataset_deptos(df_yerba, coords):
    """Une centroides y calcula los radios (en metros) a partir de las superficies."""
    df_yerba = df_yerba.merge(coords, on="departamento_id", how="left")

    # Area = pi * r^2  ->  r = sqrt(Area / pi). 1 ha = 10.000 m2, el radio queda en metros.
    df_yerba["radio_sembrado"] = np.sqrt(df_yerba["superficie_sembrada_ha"] * 10000 / np.pi)
    df_yerba["radio_cosechado"] = np.sqrt(df_yerba["superficie_cosechada_ha"] * 10000 / np.pi)
    return df_yerba


@st.cache_data(show_spinner="Cargando datos y centroides...")
def cargar_deptos():
    df_yerba = pd.read_csv(ARCHIVO_DEPTOS, encoding="cp1252")

    # Casteo y separacion de los ID's
    df_yerba["departamento_id"] = df_yerba["departamento_id"].astype(str)
    ids = df_yerba["departamento_id"].unique().tolist()

    coords = obtener_centroides(ids)
    coords["departamento_id"] = coords["departamento_id"].astype(str)
    return preparar_dataset_deptos(df_yerba, coords)


@st.cache_data
def cargar_nacional():
    return pd.read_csv(ARCHIVO_NACIONAL)


def construir_series(df_deptos, df_nacional):
    """Series anuales: pais (dataset anual) y provincias (agrupando departamentos).

    El rendimiento y el % cosechado se recalculan igual para ambas, asi son comparables.
    """
    cols = ["superficie_sembrada_ha", "superficie_cosechada_ha", "produccion_tm"]
    prov = df_deptos.groupby(["anio", "provincia"], as_index=False)[cols].sum()

    pais = df_nacional.rename(columns={
        "indice_tiempo": "anio",
        "superficie_sembrada_yerba_mate_ha": "superficie_sembrada_ha",
        "superficie_cosechada_yerba_mate_ha": "superficie_cosechada_ha",
        "produccion_yerba_mate_t": "produccion_tm",
    })[["anio"] + cols].copy()
    pais["provincia"] = "Argentina"

    for d in (pais, prov):
        d["rendimiento_kgxha"] = (
            d["produccion_tm"] * 1000 / d["superficie_cosechada_ha"].replace(0, np.nan)
        )
        d["pct_cosechada"] = (
            d["superficie_cosechada_ha"] / d["superficie_sembrada_ha"].replace(0, np.nan) * 100
        )
    return pais, prov


def formato(n, decimales=0):
    return f"{n:,.{decimales}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def hex_a_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def colores_series(columnas, c1, c2):
    """Colores para las columnas de un grafico, respetando la paleta elegida."""
    fijos = {"Argentina": c1, "Total pais": c1, "Misiones": c1, "Corrientes": c2}
    colores, extra = [], iter(EXTRAS)
    for c in columnas:
        colores.append(fijos[c] if c in fijos else next(extra, SOL))
    return colores


# ---------------------------------------------------------------------------
# Mapa de extension (st.map)
# ---------------------------------------------------------------------------
def datos_mapa_extension(df, anio, metrica, escala, color_hex):
    """Un punto por departamento, SIEMPRE los mismos puntos todos los Años.

    st.map recalcula centro y zoom a partir de los datos: si los puntos cambian de un
    Año a otro, el mapa "salta" y pierde el zoom del usuario. Por eso los departamentos
    sin yerba en el Año quedan como puntos transparentes en lugar de eliminarse.
    """
    sembrado = metrica == "Sembrado"
    col_area = "superficie_sembrada_ha" if sembrado else "superficie_cosechada_ha"
    col_radio = "radio_sembrado" if sembrado else "radio_cosechado"

    base = df.dropna(subset=["lat", "lon"])
    todos = base[["departamento_id", "lat", "lon"]].drop_duplicates("departamento_id")
    actual = base[base["anio"] == anio][["departamento_id", "departamento", "provincia", col_area, col_radio]]
    m = todos.merge(actual, on="departamento_id", how="left")

    visible = m[col_area].fillna(0) > 0
    # st.map interpreta la columna `size` como radio en metros.
    m["radio_m"] = (m[col_radio].fillna(0) * escala).where(visible, 1.0)
    rgb = hex_a_rgb(color_hex)
    m["color"] = [rgb + [210] if v else [0, 0, 0, 0] for v in visible]
    return m, visible, col_area, base


def dibujar_mapa_extension(df, anio, metrica, escala, color_hex, zoom, ph):
    m, visible, col_area, base = datos_mapa_extension(df, anio, metrica, escala, color_hex)
    ph["titulo"].markdown(f"**Año {anio}**")
    ph["mapa"].map(m, latitude="lat", longitude="lon", size="radio_m", color="color", zoom=zoom)

    total = m.loc[visible, col_area].sum()
    previo_df = base[base["anio"] == anio - 1]
    delta = None if previo_df.empty else total - previo_df[col_area].sum()
    with ph["metricas"].container():
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Superficie {metrica.lower()} {anio}", f"{formato(total)} ha",
                  None if delta is None else f"{formato(delta)} ha vs {anio - 1}")
        c2.metric("Departamentos con yerba", int(visible.sum()))
        if visible.any():
            mayor = m.loc[m[col_area].idxmax()]
            c3.metric("Mayor extension", f"{mayor['departamento']}",
                      f"{formato(mayor[col_area])} ha", delta_color="off")


# ---------------------------------------------------------------------------
# Exploracion de produccion y rendimiento
# ---------------------------------------------------------------------------
VARIABLES = {
    "Produccion (t)": "produccion_tm",
    "Rendimiento (kg/ha)": "rendimiento_kgxha",
}


def serie_exploracion(df, variable, agrupar, departamentos=None):
    """Serie ancha (Año x grupo) de produccion o rendimiento."""
    if agrupar == "Por departamento":
        d = df if not departamentos else df[df["departamento"].isin(departamentos)]
        return d.pivot_table(index="anio", columns="departamento", values=variable, aggfunc="sum")

    claves = {"Total": None, "Por provincia": "provincia"}
    clave = claves[agrupar]
    cols = ["produccion_tm", "superficie_cosechada_ha"]
    if clave is None:
        g = df.groupby("anio")[cols].sum()
        out = pd.DataFrame(index=g.index)
        if variable == "produccion_tm":
            out["Total pais"] = g["produccion_tm"]
        else:  # rendimiento ponderado por superficie cosechada
            out["Total pais"] = g["produccion_tm"] * 1000 / g["superficie_cosechada_ha"].replace(0, np.nan)
        return out

    g = df.groupby(["anio", clave])[cols].sum().reset_index()
    if variable == "produccion_tm":
        g["valor"] = g["produccion_tm"]
    else:
        g["valor"] = g["produccion_tm"] * 1000 / g["superficie_cosechada_ha"].replace(0, np.nan)
    return g.pivot(index="anio", columns=clave, values="valor")


def resumen_series(ancha):
    filas = []
    for col in ancha.columns:
        s = ancha[col].dropna()
        if s.empty:
            continue
        filas.append({
            "Serie": col,
            "Maximo": s.max(), "Año del maximo": int(s.idxmax()),
            "Minimo": s.min(), "Año del minimo": int(s.idxmin()),
            "Promedio": s.mean(), "Ultimo": s.iloc[-1],
            "Variacion primero-ultimo (%)": (s.iloc[-1] / s.iloc[0] - 1) * 100 if s.iloc[0] else np.nan,
        })
    return pd.DataFrame(filas).set_index("Serie").round(1)


def seccion_exploracion(df, c1, c2):
    anio_min, anio_max = int(df["anio"].min()), int(df["anio"].max())

    f1, f2, f3 = st.columns(3)
    nombre_var = f1.radio("Variable", list(VARIABLES), horizontal=True, key="exp_var")
    agrupar = f2.radio("Agrupar", ["Total", "Por provincia", "Por departamento"], horizontal=True,
                       key="exp_agr")
    rango = f3.slider("Rango de años", anio_min, anio_max, (anio_min, anio_max), key="exp_rango")
    variable = VARIABLES[nombre_var]

    departamentos = None
    if agrupar == "Por departamento":
        todos = sorted(df["departamento"].unique())
        top = (df[df["anio"] == anio_max].nlargest(4, "produccion_tm")["departamento"].tolist())
        departamentos = st.multiselect("Departamentos", todos, default=top, key="exp_deptos")
        if not departamentos:
            st.info("Elegi al menos un departamento.")
            return

    d = df[(df["anio"] >= rango[0]) & (df["anio"] <= rango[1])]
    ancha = serie_exploracion(d, variable, agrupar, departamentos)
    st.subheader(f"{nombre_var} a lo largo del tiempo")
    st.line_chart(ancha, color=colores_series(ancha.columns, c1, c2) if agrupar != "Por departamento"
                  else (colores_departamentos(c1, c2, len(ancha.columns))))
    st.dataframe(resumen_series(ancha), use_container_width=True)

    st.subheader("Un Año en detalle")
    anio = st.slider("Año", rango[0], rango[1], rango[1], key="exp_anio")
    dy = df[df["anio"] == anio].copy()
    izq, der = st.columns(2)
    with izq:
        st.markdown(f"**Ranking de departamentos por {nombre_var.lower()}, {anio}**")
        ranking = dy.set_index("departamento")[variable].sort_values(ascending=False)
        st.bar_chart(ranking, color=c1)
    with der:
        st.markdown(f"**Produccion vs rendimiento, {anio}** (tamano = superficie cosechada)")
        dy["color_prov"] = np.where(dy["provincia"] == "Misiones", c1, c2)
        dy["rendimiento_kgxha"] = dy["rendimiento_kgxha"].astype(float)
        st.scatter_chart(dy, x="produccion_tm", y="rendimiento_kgxha",
                         size="superficie_cosechada_ha", color="color_prov")
        st.caption(f"{c1} = Misiones, {c2} = Corrientes.")


def colores_departamentos(c1, c2, n):
    ciclo = [c1, c2] + EXTRAS
    return [ciclo[i % len(ciclo)] for i in range(n)]


# ---------------------------------------------------------------------------
# Insights con el dataset anual (pais) y los departamentos agrupados (provincias)
# ---------------------------------------------------------------------------
def serie_ancha(pais, prov, vista, variable):
    origen = pais if vista == "Pais total" else prov
    return origen.pivot(index="anio", columns="provincia", values=variable).sort_index()


def saltos_mas_fuertes(ancha):
    """Mayor variacion interanual (en %) de cada columna."""
    filas = []
    for col in ancha.columns:
        var = ancha[col].pct_change() * 100
        anio = var.abs().idxmax()
        filas.append((col, int(anio), float(var.loc[anio])))
    return filas


def seccion_insights(df_deptos, df_nacional, c1, c2):
    pais, prov = construir_series(df_deptos, df_nacional)

    vista = st.radio("Nivel de detalle", ["Pais total", "Por provincia"], horizontal=True,
                     key="vista_insights")

    # ---- Insight 1: produccion y rendimiento a lo largo de las decadas
    st.subheader("1. Produccion y rendimiento 1970-2024")
    variable = st.radio("Variable", ["Produccion (t)", "Rendimiento (kg/ha)"], horizontal=True,
                        key="var_insight1")
    col = "produccion_tm" if variable.startswith("Produccion") else "rendimiento_kgxha"
    ancha = serie_ancha(pais, prov, vista, col)
    st.line_chart(ancha, color=colores_series(ancha.columns, c1, c2))
    for nombre, anio, var in saltos_mas_fuertes(ancha):
        st.caption(
            f"{nombre}: la mayor variacion interanual fue en {anio} ({var:+.0f}%). "
            "Un salto de esta magnitud puede deberse a un cambio de metodologia de relevamiento "
            "mas que a un cambio real del cultivo; conviene confirmarlo con la fuente."
        )

    # ---- Insight 2: cuanta superficie plantada se cosecha
    st.subheader("2. Que parte de lo sembrado se cosecha")
    ancha2 = serie_ancha(pais, prov, vista, "pct_cosechada")
    st.line_chart(ancha2, color=colores_series(ancha2.columns, c1, c2))
    for nombre in ancha2.columns:
        serie = ancha2[nombre].dropna()
        anios_pleno = serie[serie >= 99.9].index.tolist()
        texto = (f"{nombre}: paso de {serie.iloc[0]:.0f}% en {serie.index[0]} "
                 f"a {serie.iloc[-1]:.0f}% en {serie.index[-1]}.")
        if anios_pleno:
            texto += (f" Figura 100% cosechado entre {min(anios_pleno)} y {max(anios_pleno)}"
                      " (en algunos Años, lo que sugiere que sembrado y cosechado se informaron igual).")
        st.caption(texto)

    # ---- Insight 3: promedio por decada y crecimiento
    st.subheader("3. Promedio por decada y crecimiento anual compuesto")
    ancha3 = serie_ancha(pais, prov, vista, "produccion_tm")
    por_decada = ancha3.groupby((ancha3.index // 10) * 10).mean()
    por_decada.index = [f"{d}s" for d in por_decada.index]
    st.bar_chart(por_decada, color=colores_series(por_decada.columns, c1, c2), stack=False)

    cols = st.columns(len(ancha3.columns))
    for c, nombre in zip(cols, ancha3.columns):
        serie = ancha3[nombre].dropna()
        n = serie.index[-1] - serie.index[0]
        cagr = (serie.iloc[-1] / serie.iloc[0]) ** (1 / n) - 1
        c.metric(f"{nombre}: crecimiento anual {serie.index[0]}-{serie.index[-1]}",
                 f"{cagr * 100:.1f}%", f"{formato(serie.iloc[-1])} t en {serie.index[-1]}",
                 delta_color="off")
    if vista == "Por provincia":
        ultimo = ancha3.iloc[-1]
        st.caption("Participacion en la produccion del ultimo Año: " + ", ".join(
            f"{p} {v / ultimo.sum() * 100:.0f}%" for p, v in ultimo.items()) + ".")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def estilo():
    st.markdown(
        f"""
        <style>
        .banda-ar {{
            height: 22px; border-radius: 4px; margin-bottom: 0.6rem;
            background: linear-gradient(180deg, {CELESTE} 0 33%, #ffffff 33% 66%, {CELESTE} 66% 100%);
            border: 1px solid {CELESTE};
        }}
        h1 span.sol {{ color: {SOL}; }}
        div[data-testid="stMetric"] {{
            background: rgba(116, 172, 223, 0.18); border-left: 5px solid {SOL};
            padding: 0.5rem 0.8rem; border-radius: 6px;
        }}
        </style>
        <div class="banda-ar"></div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Yerba mate argentina", layout="wide")
    estilo()
    c1, c2 = COLOR_1, COLOR_2
    st.markdown("# Yerba mate en Argentina <span class='sol'>&#9728;</span>", unsafe_allow_html=True)
    st.markdown("**Dashboard realizado por Francisco Rizzi**")
    st.caption("Superficie, produccion y rendimiento por departamento, 1970-2024. "
               "Fuente: datasets de yerba mate incluidos y centroides de la API Georef.")

    try:
        df = cargar_deptos()
    except Exception as e:  # sin red y sin cache de centroides
        st.error(f"No pude obtener los centroides de los departamentos: {e}")
        st.stop()
    nacional = cargar_nacional()

    anio_min, anio_max = int(df["anio"].min()), int(df["anio"].max())

    tab_ext, tab_exp, tab_ins = st.tabs(["Extension", "Produccion y rendimiento", "Insights"])

    # El pie va antes de llenar las pestanas: con la animacion en bucle el script
    # no pasa de la pestania Extension, y asi igual se dibuja al final de la pagina.
    st.divider()
    st.markdown(
        "Fuente de los datos: "
        "[Yerba mate: siembra, cosecha, produccion y rendimiento (datos.gob.ar)]"
        "(https://datos.gob.ar/dataset/yerba-mate-siembra-cosecha-produccion-rendimiento)"
    )

    # ---------------- Produccion y rendimiento
    with tab_exp:
        seccion_exploracion(df, c1, c2)

    # ---------------- Insights
    with tab_ins:
        seccion_insights(df, nacional, c1, c2)

    # Esta pestania va al final: con la animacion en bucle el script no avanza mas alla de ella.
    # ---------------- Extension
    with tab_ext:
        k1, k2, k3 = st.columns(3)
        metrica = k1.radio("Radio segun superficie", ["Sembrado", "Cosechado"], horizontal=True,
                           key="ext_metrica")
        escala = k2.select_slider("Multiplo del radio (1 = tamano real)", [1, 2, 3, 5, 10], value=1,
                                  key="ext_escala")
        pausa = k3.slider("Segundos por Año (animacion)", 0.2, 2.0, 0.6, 0.1, key="ext_pausa")
        anio = st.slider("Año", anio_min, anio_max, anio_max, key="ext_anio")
        play = st.toggle("Reproducir en bucle", key="ext_play")

        color = c1 if metrica == "Sembrado" else c2
        ph = {"titulo": st.empty(), "mapa": st.empty(), "metricas": st.empty()}

        # El grafico de totales va antes del bucle: si no, nunca llegaria a dibujarse.
        totales = df.groupby("anio")[["superficie_sembrada_ha", "superficie_cosechada_ha"]].sum()
        st.line_chart(totales, color=[c1, c2])

        zoom = 7
        if play:
            anios = list(range(anio_min, anio_max + 1))
            i = anios.index(anio)
            while True:
                dibujar_mapa_extension(df, anios[i], metrica, escala, color, zoom, ph)
                time.sleep(pausa)
                i = (i + 1) % len(anios)
        else:
            dibujar_mapa_extension(df, anio, metrica, escala, color, zoom, ph)


if __name__ == "__main__":
    main()