import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1. Parseo de la ficha de catálogo
# --------------------------------------------------------------------------

def extraer_campo(serie: pd.Series, campo: str) -> pd.Series:
    """Extrae el valor de una línea 'Campo: valor' del texto de catálogo."""
    return serie.str.extract(rf"(?m)^{campo}:\s*(.*)$")[0].str.strip()


def parsear_catalogo(texto: pd.Series) -> pd.DataFrame:
    """Separa catalog_content en sus campos internos."""
    texto = texto.fillna("").astype(str)
    out = pd.DataFrame(index=texto.index)
    out["item_name"] = extraer_campo(texto, "Item Name").fillna("")
    out["bullets"] = texto.str.findall(r"(?m)^Bullet Point\s*\d*:\s*(.*)$").map(" ".join)
    out["n_bullets"] = texto.str.count(r"(?m)^Bullet Point")
    out["description"] = (texto.str.extract(r"(?ms)^Product Description:\s*(.*?)(?=^Value:|\Z)")[0]
                          .fillna("").str.strip())
    out["value"] = pd.to_numeric(extraer_campo(texto, "Value"), errors="coerce")
    out["unit_raw"] = extraer_campo(texto, "Unit")
    return out


# --------------------------------------------------------------------------
# 2. Normalización de unidades
# --------------------------------------------------------------------------
# Familias y factor para pasar a la unidad estándar:
#   Peso -> onzas (oz) | Volumen -> onzas líquidas (fl oz) | Conteo -> unidades
UNIDADES_PESO = {"ounce": 1.0, "ounces": 1.0, "oz": 1.0, "onzas": 1.0,
                 "pound": 16.0, "pounds": 16.0, "lb": 16.0, "lbs": 16.0,
                 "gram": 0.035274, "grams": 0.035274, "gramm": 0.035274, "gr": 0.035274,
                 "g": 0.035274, "gm": 0.035274, "grams(gm)": 0.035274,
                 "kg": 35.274, "kilogram": 35.274, "kilograms": 35.274}
UNIDADES_VOLUMEN = {"ml": 0.033814, "millilitre": 0.033814, "milliliter": 0.033814, "mililitro": 0.033814,
                    "l": 33.814, "liter": 33.814, "liters": 33.814, "litre": 33.814, "ltr": 33.814}
PALABRAS_CONTEO = ["count", "ct", "each", "piece", "unit", "pack", "bottle", "can", "jar", "bag", "box",
                   "pouch", "carton", "k-cup", "capsule", "tea bag", "bucket", "liner", "unità", "per "]

QTY_MAXIMA = 10_000  # por encima se considera error de captura (palés, unidades mal declaradas)


def normalizar_unidad(u):
    """Devuelve (familia, factor) para una unidad escrita de cualquier forma."""
    if not isinstance(u, str):
        return "Desconocido", np.nan
    u = re.sub(r"\s+", " ", u.strip().lower().replace("(s)", "").replace(".", " ")).strip()
    if u in {"", "none", "-", "---"} or re.fullmatch(r"[\d ,]+", u):
        return "Desconocido", np.nan
    if "fl" in u.split()[0] or "fluid" in u:
        return "Volumen", 1.0
    if u in UNIDADES_VOLUMEN:
        return "Volumen", UNIDADES_VOLUMEN[u]
    if u in UNIDADES_PESO:
        return "Peso", UNIDADES_PESO[u]
    if re.fullmatch(r"[\d,]+ oz", u):
        return "Peso", 1.0
    if any(p in u for p in PALABRAS_CONTEO):
        return "Conteo", 1.0
    return "Otro", np.nan


def cantidad_estandar(value: pd.Series, unit_raw: pd.Series) -> pd.DataFrame:
    """Calcula familia de unidad, cantidad estándar y si la cantidad es inválida."""
    familia_factor = unit_raw.map(normalizar_unidad)
    familia = familia_factor.str[0]
    qty = value * familia_factor.str[1].astype(float)
    invalida = qty.isna() | (qty <= 0) | (qty > QTY_MAXIMA)
    return pd.DataFrame({
        "unit_family": familia.replace({"Otro": "Desconocido"}),
        "qty_std": qty.where(~invalida),
        "qty_missing": invalida.astype(int),
    })


# --------------------------------------------------------------------------
# 3. Packs
# --------------------------------------------------------------------------
PATRONES_PACK = [
    r"(?:pack|case|box|set|bundle) of (\d{1,3})",
    r"(\d{1,3})\s*-?\s*(?:pack|pk|packs)\b",
    r"(\d{1,3})\s*(?:per|/)\s*(?:case|pack|box|carton)",
    r"\(\s*(\d{1,3})\s*(?:count|ct|pack|pk|bottles|cans|bags|boxes)\s*\)",
]


def extraer_pack(nombre) -> int:
    """Número de unidades del pack leído del nombre (1 si no es pack)."""
    if not isinstance(nombre, str):
        return 1
    n = nombre.lower()
    for patron in PATRONES_PACK:
        m = re.search(patron, n)
        if m and 1 < int(m.group(1)) <= 200:
            return int(m.group(1))
    return 1


# --------------------------------------------------------------------------
# 4. Limpieza del dataset de entrenamiento
# --------------------------------------------------------------------------

def limpiar_dataset(df_raw: pd.DataFrame):
    """
    Limpia el dataset crudo (sample_id, catalog_content, image_link, price).

    Devuelve (df_limpio, reporte) donde `reporte` es una tabla con lo que hizo cada paso.
    """
    reporte = []
    df = df_raw.copy()
    reporte.append(("0. Dataset original", len(df), "-"))

    # Paso 1: filas sin texto o sin precio válido (no se pueden usar para entrenar)
    malas = df["catalog_content"].isna() | df["price"].isna() | (df["price"] <= 0)
    df = df.loc[~malas]
    reporte.append(("1. Eliminar filas sin texto o con precio <= 0", len(df), f"{int(malas.sum())} eliminadas"))

    # Paso 2: duplicados exactos (mismo texto y mismo precio)
    dup = df.duplicated(subset=["catalog_content", "price"])
    df = df.loc[~dup].reset_index(drop=True)
    reporte.append(("2. Eliminar duplicados exactos (texto + precio)", len(df), f"{int(dup.sum())} eliminadas"))

    # Paso 3: parsear la ficha de catálogo en campos
    df = pd.concat([df, parsear_catalogo(df["catalog_content"])], axis=1)
    reporte.append(("3. Parsear nombre, bullets, descripción, Value, Unit", len(df), "5 campos nuevos"))

    # Paso 4: normalizar unidades y cantidad
    n_grafias = df["unit_raw"].str.lower().nunique()
    df = pd.concat([df, cantidad_estandar(df["value"], df["unit_raw"])], axis=1)
    reporte.append(("4. Normalizar unidades → Peso / Volumen / Conteo", len(df),
                    f"{n_grafias} grafías → {df['unit_family'].nunique()} familias"))
    reporte.append(("5. Marcar cantidades inválidas (vacías, 0 o > 10.000)", len(df),
                    f"{int(df['qty_missing'].sum())} marcadas (no se borran)"))

    # Paso 6: packs y variables de texto
    df["pack_count"] = df["item_name"].map(extraer_pack)
    df["has_description"] = (df["description"].str.len() > 0).astype(int)
    df["n_words"] = df["catalog_content"].str.split().str.len()
    reporte.append(("6. Extraer packs y variables de texto", len(df), "pack_count, has_description, n_words"))

    # Paso 7: target en escala log y grupo de imagen (para validar sin fuga)
    df["log_price"] = np.log1p(df["price"])
    df["image_group"] = df.groupby("image_link", sort=False).ngroup()
    reporte.append(("7. Crear log_price e image_group", len(df), "target y grupos para validación"))

    reporte = pd.DataFrame(reporte, columns=["paso", "filas", "detalle"])
    return df, reporte


# --------------------------------------------------------------------------
# 5. Features para el modelo (se aplica igual en entrenamiento y en predicción)
# --------------------------------------------------------------------------
COLUMNAS_NUMERICAS = ["log_qty", "qty_missing", "log_pack", "n_bullets", "has_description", "log_words"]
COLUMNAS_CATEGORICAS = ["unit_family"]


def construir_features(X) -> pd.DataFrame:
    """
    Recibe un DataFrame con la columna `catalog_content` (texto crudo)
    y devuelve las columnas que usa el modelo.
    """
    if isinstance(X, pd.Series):
        X = X.to_frame("catalog_content")
    texto = X["catalog_content"].fillna("").astype(str).reset_index(drop=True)
    p = parsear_catalogo(texto)
    q = cantidad_estandar(p["value"], p["unit_raw"])
    return pd.DataFrame({
        "nombre": p["item_name"].str.lower(),
        "log_qty": np.log1p(q["qty_std"]).fillna(0.0),
        "qty_missing": q["qty_missing"],
        "unit_family": q["unit_family"],
        "log_pack": np.log(p["item_name"].map(extraer_pack)),
        "n_bullets": p["n_bullets"],
        "has_description": (p["description"].str.len() > 0).astype(int),
        "log_words": np.log1p(texto.str.split().str.len()),
    })


def smape(y_true, y_pred) -> float:
    """SMAPE en % (métrica del Amazon ML Challenge). 0 = perfecto, 200 = peor."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    den = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(np.where(den == 0, 0, 2 * np.abs(y_pred - y_true) / den)) * 100)
