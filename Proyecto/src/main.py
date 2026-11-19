import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

CARPETA = Path(__file__).resolve().parent          # .../Proyecto/src
sys.path.insert(0, str(CARPETA))
from artifacts import features as sf, limpieza  # noqa: E402

# El modelo se guardó cuando los módulos se llamaban stb_features / stb_limpieza:
# se registran con ese nombre para que joblib encuentre las clases al cargarlo.
sys.modules["stb_features"], sys.modules["stb_limpieza"] = sf, limpieza

RUTA_MODELO = CARPETA.parent / "models" / "stb_modelo_v2.joblib"

# unidad -> (unidad que entiende el modelo, factor para pasar a g / ml / unidades, unidad base)
UNIDADES = {"mg": ("grams", 0.001, "g"), "g": ("grams", 1, "g"), "kg": ("grams", 1000, "g"),
            "ml": ("ml", 1, "ml"), "l": ("ml", 1000, "ml"), "unidades": ("count", 1, "unidades")}
# Rango de cantidades del entrenamiento (~percentiles 0,5 y 99,5): fuera de él la predicción es poco fiable
RANGO = {"g": (10, 15_000), "ml": (10, 20_000), "unidades": (1, 600)}
modelo, terminos_cluster = None, None


@asynccontextmanager
async def lifespan(app):
    global modelo, terminos_cluster
    modelo = joblib.load(RUTA_MODELO)
    svd = modelo.cod_.svd_
    svd.components_ = svd.components_.astype(np.float32)  # se guarda en float16 para pesar la mitad
    terminos_cluster = modelo.cod_.terminos_cluster(5)
    yield


app = FastAPI(title="Estimación de precios", version="1.0", lifespan=lifespan, redoc_url=None,
              description="Estima el precio en dólares (USD) de un producto de alimentación a partir de su nombre y cantidad.")


class Producto(BaseModel):
    nombre: str = Field(description="Nombre del producto en inglés, como aparece en Amazon (marca + tipo de producto).")
    cantidad: float = Field(gt=0, description="Cantidad total del envase (si es un pack, la suma de todas las unidades).")
    unidad: Literal["mg", "g", "kg", "ml", "l", "unidades"] = Field(description="Unidad de la cantidad.")

    model_config = {"json_schema_extra": {"examples": [
        {"nombre": "Lavazza Super Crema Whole Bean Coffee Blend, Medium Espresso Roast",
         "cantidad": 1, "unidad": "kg"}]}}


class Explicacion(BaseModel):
    cantidad_interpretada: str = Field(description="Cantidad que usó el modelo, pasada a g, ml o unidades.")
    tipo_producto: str = Field(description="Grupo de productos parecidos (cluster) al que asignó el nombre.")
    precio_tipico_tipo_producto: float = Field(description="Precio mediano de ese grupo en el entrenamiento.")
    precio_tipico_marca: float = Field(description="Precio mediano de la marca (si es desconocida, el global).")
    marca_conocida: bool = Field(description="Si la marca aparecía en los datos de entrenamiento.")
    precio_esperado_por_cantidad: float = Field(description="Precio por unidad típico del grupo x la cantidad.")
    precio_por_kg_l_o_unidad: float = Field(description="Precio estimado dividido por la cantidad (por kg, litro o unidad).")


class Prediccion(BaseModel):
    precio_estimado: float = Field(description="Precio estimado en dólares (USD).")
    categoria: str
    marca: str
    explicacion: Explicacion
    aviso: Optional[str] = Field(None, description="Advertencia si la cantidad está fuera del rango de entrenamiento.")


@app.post("/predecir", response_model=Prediccion, summary="Estimar el precio de un producto")
def predecir(p: Producto):
    """
    Cómo predice el modelo:
    1. Lee el **nombre** (TF-IDF + Ridge): las palabras indican qué es el producto y si es de marca cara o barata.
    2. Asigna el producto a un **tipo de producto** (cluster) y busca su precio típico y el de la **marca**.
    3. Calcula un **precio esperado** = precio por unidad típico del tipo de producto x la **cantidad**.
    4. Un **LightGBM** combina todo lo anterior y da el precio final.

    La cantidad influye, pero no de forma proporcional: el doble de cantidad no cuesta el doble
    (descuento por volumen), y el nombre del producto pesa más que la cantidad.
    """
    unidad_modelo, factor, base = UNIDADES[p.unidad]
    cantidad_base = p.cantidad * factor
    texto = pd.Series([f"Item Name: {p.nombre}\nValue: {cantidad_base}\nUnit: {unidad_modelo}\n"])

    f = sf.features_basicas(texto)
    enc = modelo.cod_.transform(f).iloc[0]
    precio = float(modelo.predict(texto)[0])
    divisor = cantidad_base / 1000 if base in ("g", "ml") else cantidad_base
    minimo, maximo = RANGO[base]
    aviso = None
    if not minimo <= cantidad_base <= maximo:
        aviso = (f"La cantidad ({cantidad_base:g} {base}) está fuera del rango de los productos de entrenamiento "
                 f"({minimo:,} - {maximo:,} {base}); el modelo casi no la tiene en cuenta y la predicción es poco fiable.")

    return Prediccion(
        precio_estimado=round(precio, 2), categoria=f["categoria"].iloc[0], marca=f["marca"].iloc[0],
        explicacion=Explicacion(
            cantidad_interpretada=f"{cantidad_base:g} {base}",
            tipo_producto=terminos_cluster[int(enc["cluster_producto"])],
            precio_tipico_tipo_producto=round(float(np.expm1(enc["precio_tipico_cluster"])), 2),
            precio_tipico_marca=round(float(np.expm1(enc["precio_tipico_marca"])), 2),
            marca_conocida=bool(enc["frecuencia_marca"] > 0),
            precio_esperado_por_cantidad=round(max(float(np.expm1(enc["log_precio_esperado"])), 0.0), 2),
            precio_por_kg_l_o_unidad=round(precio / divisor, 2)),
        aviso=aviso)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
