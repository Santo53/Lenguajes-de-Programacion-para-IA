import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler

from artifacts import limpieza as stb

SEMILLA = 42

# --------------------------------------------------------------------------
# 1. Split train / validation / test
# --------------------------------------------------------------------------

def crear_split(grupos, frac_val=0.15, frac_test=0.15, semilla=SEMILLA) -> np.ndarray:
    """Devuelve un array con 'train' / 'validation' / 'test'. Variantes de la misma imagen van juntas."""
    grupos = np.asarray(grupos)
    idx = np.arange(len(grupos))
    resto, test = next(GroupShuffleSplit(1, test_size=frac_test, random_state=semilla).split(idx, groups=grupos))
    frac_val_rel = frac_val / (1 - frac_test)
    tr, va = next(GroupShuffleSplit(1, test_size=frac_val_rel, random_state=semilla).split(resto, groups=grupos[resto]))
    split = np.empty(len(grupos), dtype=object)
    split[resto[tr]], split[resto[va]], split[test] = "train", "validation", "test"
    return split


# --------------------------------------------------------------------------
# 2. Variables claras (solo del texto, sin aprender nada)
# --------------------------------------------------------------------------
PALABRAS_CLAVE = {
    "es_organico": r"\borganic",
    "es_sin_gluten": r"gluten[\s-]*free",
    "es_kosher": r"\bkosher\b",
    "es_vegano": r"\bvegan\b",
    "es_sin_azucar": r"sugar[\s-]*free|no sugar|zero sugar|unsweetened",
    "es_regalo": r"\bgift|\bbasket\b|hamper|holiday|christmas",
    "es_granel": r"\bbulk\b|\bcase\b|wholesale|pallet|foodservice|food service",
    "es_premium": r"premium|gourmet|artisan|luxury|imported",
    "es_variedad": r"variety|assort|sampler",
    "es_individual": r"single[\s-]*serve|individual|snack[\s-]*size|travel size|on[\s-]*the[\s-]*go",
}

CATEGORIAS = [
    ("Café y té", r"coffee|tea\b|teas\b|k-cup|kcup|espresso|matcha|chai|cappuccino"),
    ("Bebidas", r"juice|soda|\bwater\b|drink|beverage|wine|beer|kombucha|lemonade|sparkling|cola|smoothie"),
    ("Dulces y chocolate", r"candy|chocolate|gummy|gummies|gummi|caramel|lollipop|marshmallow|licorice|mint"),
    ("Snacks", r"chip|cracker|snack|popcorn|pretzel|jerky|bar\b|bars\b|cookie|nuts?\b|almond|cashew|pistachio|trail mix"),
    ("Salsas y condimentos", r"sauce|seasoning|spice|salt|pepper|dressing|ketchup|mustard|vinegar|marinade|salsa|rub\b|extract|syrup|honey|oil\b"),
    ("Panadería, cereales y pasta", r"bread|cereal|oat|flour|granola|pasta|rice|noodle|baking|cake|mix\b|tortilla|quinoa|muffin|pancake"),
    ("Lácteos y proteína", r"cheese|milk|protein|butter|yogurt|creamer|meat|tuna|chicken|beef|salmon|egg"),
    ("Conservas y sopas", r"soup|broth|canned|beans|bean\b|tomato|pickle|olive|stock\b"),
    ("Frutas, verduras y semillas", r"fruit|berry|berries|apple|mango|vegetable|seed|raisin|date\b|fig\b|coconut"),
]
_PALABRAS_CORTAS = {"la", "le", "el", "the", "mrs", "mr", "dr", "st", "de", "old", "big", "little", "san",
                    "member's", "trader", "great", "good", "happy", "nature's"}


def categorizar(nombre) -> str:
    n = nombre.lower() if isinstance(nombre, str) else ""
    for cat, patron in CATEGORIAS:
        if re.search(patron, n):
            return cat
    return "Otros"


def marca_aproximada(nombre) -> str:
    if not isinstance(nombre, str):
        return "desconocida"
    palabras = re.sub(r"[^\w'&\- ]", " ", nombre).split()
    if not palabras:
        return "desconocida"
    n = 2 if palabras[0].lower().strip(".") in _PALABRAS_CORTAS and len(palabras) > 1 else 1
    return " ".join(palabras[:n]).lower()


def features_basicas(catalog_content) -> pd.DataFrame:
    """Variables legibles derivadas solo del texto crudo de la ficha."""
    texto = pd.Series(catalog_content).fillna("").astype(str).reset_index(drop=True)
    p = stb.parsear_catalogo(texto)
    q = stb.cantidad_estandar(p["value"], p["unit_raw"])
    pack = p["item_name"].map(stb.extraer_pack)
    texto_min = texto.str.lower()
    n_car = texto.str.len()

    f = pd.DataFrame({
        "nombre_producto": p["item_name"],
        "categoria": p["item_name"].map(categorizar),
        "marca": p["item_name"].map(marca_aproximada),
        # Cantidad
        "familia_unidad": q["unit_family"],
        "cantidad_estandar": q["qty_std"],
        "log_cantidad": np.log1p(q["qty_std"]),
        "cantidad_invalida": q["qty_missing"],
        # Pack
        "unidades_pack": pack,
        "es_pack": (pack > 1).astype(int),
        "log_unidades_pack": np.log(pack),
        "cantidad_por_unidad": q["qty_std"] / pack,
        "log_cantidad_por_unidad": np.log1p(q["qty_std"] / pack),
        # Ficha
        "n_caracteres": n_car,
        "log_palabras": np.log1p(texto.str.split().str.len()),
        "n_bullets": p["n_bullets"],
        "tiene_descripcion": (p["description"].str.len() > 0).astype(int),
        "palabras_nombre": p["item_name"].str.split().str.len().fillna(0).astype(int),
        "ratio_mayusculas": texto.str.count(r"[A-Z]") / n_car.clip(lower=1),
        "ratio_digitos": texto.str.count(r"\d") / n_car.clip(lower=1),
    })
    for col, patron in PALABRAS_CLAVE.items():
        f[col] = texto_min.str.contains(patron, regex=True).astype(int)
    return f


# --------------------------------------------------------------------------
# 3. Variables que aprenden de train (cluster + codificaciones de precio)
# --------------------------------------------------------------------------

class CodificadorAprendido:
    """
    - cluster_producto: KMeans sobre TF-IDF+SVD del nombre (no usa el precio).
    - precio_tipico_cluster / precio_por_unidad_cluster / precio_tipico_marca: medianas de
      log-precio (solo train, fuera de fold) -> "cuánto suele costar este tipo de producto".
    - log_precio_esperado: precio por unidad típico del cluster x cantidad del producto.
    """

    def __init__(self, n_clusters=60, n_svd=150, suavizado_marca=20, semilla=SEMILLA):
        self.n_clusters, self.n_svd, self.m, self.semilla = n_clusters, n_svd, suavizado_marca, semilla

    # ---- cluster ----
    def _ajustar_cluster(self, nombres_train, nombres_extra=None):
        corpus = pd.concat([nombres_train, nombres_extra]) if nombres_extra is not None else nombres_train
        corpus = corpus.fillna("").str.lower()
        self.tfidf_ = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True).fit(corpus)
        M = self.tfidf_.transform(corpus)
        self.svd_ = TruncatedSVD(self.n_svd, random_state=self.semilla).fit(M)
        self.norm_ = Normalizer()
        Z = self.norm_.fit_transform(self.svd_.transform(M))
        self.kmeans_ = MiniBatchKMeans(self.n_clusters, random_state=self.semilla, n_init=3, batch_size=4096).fit(Z)
        return self

    def embedding_texto(self, nombres) -> np.ndarray:
        """Componentes SVD (LSA) del nombre: un resumen denso del texto."""
        return self.svd_.transform(self.tfidf_.transform(pd.Series(nombres).fillna("").str.lower()))

    def cluster(self, nombres) -> np.ndarray:
        return self.kmeans_.predict(self.norm_.transform(self.embedding_texto(nombres)))

    # ---- codificaciones de precio ----
    @staticmethod
    def _tabla(clave, valor):
        return pd.DataFrame({"k": clave, "v": valor}).groupby("k")["v"].agg(["median", "size"])

    def _mapear(self, f, tablas):
        out = pd.DataFrame(index=f.index)
        out["precio_tipico_cluster"] = f["cluster_producto"].map(tablas["cl"]["median"]).fillna(self.global_)
        out["precio_por_unidad_cluster"] = f["cluster_producto"].map(tablas["ppu"]["median"]).fillna(self.global_ppu_)
        t = tablas["marca"]
        suav = (t["median"] * t["size"] + self.global_ * self.m) / (t["size"] + self.m)
        out["precio_tipico_marca"] = f["marca"].map(suav).fillna(self.global_)
        out["frecuencia_marca"] = f["marca"].map(t["size"]).fillna(0)
        out["log_precio_esperado"] = out["precio_por_unidad_cluster"] + np.log(f["cantidad_estandar"])
        return out

    def _tablas(self, f, y):
        log_ppu = y - np.log(f["cantidad_estandar"])            # log(precio / cantidad) aprox
        ok = f["cantidad_estandar"].notna()
        return {"cl": self._tabla(f["cluster_producto"], y),
                "ppu": self._tabla(f["cluster_producto"][ok], log_ppu[ok]),
                "marca": self._tabla(f["marca"], y)}

    def fit_transform(self, f_train, y_train, grupos, nombres_extra=None, n_folds=5):
        """Ajusta con train y devuelve las variables de train calculadas fuera de fold."""
        f = f_train.reset_index(drop=True).copy()
        y = pd.Series(np.asarray(y_train), index=f.index)
        self._ajustar_cluster(f["nombre_producto"], nombres_extra)
        f["cluster_producto"] = self.cluster(f["nombre_producto"])
        self.global_ = float(y.median())
        ok = f["cantidad_estandar"].notna()
        self.global_ppu_ = float((y - np.log(f["cantidad_estandar"]))[ok].median())

        oof = []
        for a, b in GroupKFold(n_folds).split(f, groups=grupos):
            oof.append(self._mapear(f.iloc[b], self._tablas(f.iloc[a], y.iloc[a])))
        enc = pd.concat(oof).sort_index()
        self.tablas_ = self._tablas(f, y)                        # tablas finales con todo train
        return pd.concat([f[["cluster_producto"]], enc], axis=1)

    def transform(self, f_otro):
        f = f_otro.reset_index(drop=True).copy()
        f["cluster_producto"] = self.cluster(f["nombre_producto"])
        return pd.concat([f[["cluster_producto"]], self._mapear(f, self.tablas_)], axis=1)

    def terminos_cluster(self, n=6) -> pd.Series:
        """Palabras más representativas de cada cluster (para interpretarlos)."""
        centros = self.svd_.inverse_transform(self.kmeans_.cluster_centers_)
        vocab = np.array(self.tfidf_.get_feature_names_out())
        return pd.Series({c: ", ".join(vocab[np.argsort(-centros[c])[:n]]) for c in range(self.n_clusters)})


# --------------------------------------------------------------------------
# 4. Modelo v2: Ridge (texto) + LightGBM (tabular) + mezcla
# --------------------------------------------------------------------------
PARAMS_LGBM = dict(objective="regression_l1", learning_rate=0.04, n_estimators=3000, num_leaves=63,
                   min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                   verbose=-1)

NUM_RIDGE = ["log_cantidad", "cantidad_invalida", "log_unidades_pack", "log_cantidad_por_unidad", "log_palabras",
             "n_bullets", "tiene_descripcion", "ratio_mayusculas", "ratio_digitos"] + list(PALABRAS_CLAVE)
CAT_RIDGE = ["familia_unidad", "categoria"]


def _aligerar(modelo):
    """Borra atributos que solo sirven de diagnóstico (scikit-learn recomienda eliminar `stop_words_`,
    que guarda todo el vocabulario descartado y multiplica el tamaño del .joblib)."""
    for vec in [getattr(modelo, "tf_pal_", None), getattr(modelo, "tf_car_", None),
                getattr(getattr(modelo, "cod_", None), "tfidf_", None)]:
        if vec is not None and hasattr(vec, "stop_words_"):
            vec.stop_words_ = None
    modelo.oof_ridge_ = None
    svd = getattr(getattr(modelo, "cod_", None), "svd_", None)
    if svd is not None:                       # 150 x vocabulario: float32 reduce el archivo a la mitad
        svd.components_ = svd.components_.astype(np.float32)


class ModeloPreciosV2:
    """
    Pipeline completo que recibe la columna `catalog_content` cruda y predice el precio ($).

    Etapa 1 - Ridge de texto: TF-IDF de palabras + de caracteres del nombre, variables claras,
              cluster de producto y la interacción cluster x cantidad.
    Etapa 2 - LightGBM tabular: variables seleccionadas + codificaciones aprendidas + componentes
              del texto + la predicción "fuera de fold" del Ridge (stacking).
    Etapa 3 - Mezcla: w * Ridge + (1 - w) * LightGBM, con w elegido en validación.
    """

    def __init__(self, features_lgbm, n_clusters=60, alpha=1.0, n_svd_lgbm=30, params_lgbm=None, peso_ridge=0.0,
                 usar_char=False, semilla=SEMILLA):
        self.features_lgbm = list(features_lgbm)
        self.n_clusters, self.alpha, self.n_svd_lgbm = n_clusters, alpha, n_svd_lgbm
        self.peso_ridge, self.usar_char, self.semilla = peso_ridge, usar_char, semilla
        self.params_lgbm = dict(params_lgbm or PARAMS_LGBM)

    # ---- Ridge ----
    def _matriz_ridge(self, f, ajustar=False):
        nombres = f["nombre_producto"].fillna("").str.lower()
        if ajustar:
            self.tf_pal_ = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True).fit(nombres)
            if self.usar_char:
                self.tf_car_ = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, sublinear_tf=True,
                                               max_features=200_000).fit(nombres)
            self.esc_ = StandardScaler().fit(f[NUM_RIDGE].fillna(0))
            self.oh_ = OneHotEncoder(handle_unknown="ignore").fit(f[CAT_RIDGE])
            self.media_lq_ = float(f["log_cantidad"].fillna(0).mean())
        partes = [self.tf_pal_.transform(nombres)]
        if self.usar_char:
            partes.append(self.tf_car_.transform(nombres))
        c = f["cluster_producto"].to_numpy()
        C = sp.csr_matrix((np.ones(len(c)), (np.arange(len(c)), c)), shape=(len(c), self.n_clusters))
        lq = f["log_cantidad"].fillna(0).to_numpy()[:, None] - self.media_lq_
        partes += [sp.csr_matrix(self.esc_.transform(f[NUM_RIDGE].fillna(0))), self.oh_.transform(f[CAT_RIDGE]),
                   C, C.multiply(lq).tocsr()]
        return sp.hstack(partes).tocsr()

    def _tabla_lgbm(self, f, score_ridge):
        X = f[self.features_lgbm].copy()
        for col in X.columns:
            if X[col].dtype == object or str(X[col].dtype).startswith("str"):
                X[col] = X[col].astype("category")
        if "cluster_producto" in X:
            X["cluster_producto"] = X["cluster_producto"].astype("category")
        svd = self.cod_.embedding_texto(f["nombre_producto"])[:, :self.n_svd_lgbm]
        for i in range(svd.shape[1]):
            X[f"texto_svd_{i}"] = svd[:, i]
        X["score_ridge_texto"] = score_ridge
        return X

    def preparar(self, catalog_content, ajustar=False, y=None, grupos=None, nombres_extra=None):
        """Variables claras + aprendidas. Con ajustar=True aprende de este conjunto (train)."""
        f = features_basicas(catalog_content)
        if ajustar:
            self.cod_ = CodificadorAprendido(n_clusters=self.n_clusters, semilla=self.semilla)
            enc = self.cod_.fit_transform(f, y, grupos, nombres_extra)
        else:
            enc = self.cod_.transform(f)
        return pd.concat([f, enc], axis=1)

    def fit(self, catalog_content, y_log, grupos, nombres_extra=None, n_folds=5, eval_set=None):
        import lightgbm as lgb
        y = np.asarray(y_log)
        grupos = np.asarray(grupos)
        f = self.preparar(catalog_content, ajustar=True, y=y, grupos=grupos, nombres_extra=nombres_extra)

        # Etapa 1: Ridge final + predicciones fuera de fold para el stacking
        self.ridge_ = Ridge(alpha=self.alpha).fit(self._matriz_ridge(f, ajustar=True), y)
        oof = np.zeros(len(y))
        for a, b in GroupKFold(n_folds).split(f, groups=grupos):
            sub = ModeloPreciosV2.__new__(ModeloPreciosV2)
            sub.__dict__.update({k: v for k, v in self.__dict__.items() if not k.endswith("_")})
            Xa = sub._matriz_ridge(f.iloc[a], ajustar=True)
            oof[b] = Ridge(alpha=self.alpha).fit(Xa, y[a]).predict(sub._matriz_ridge(f.iloc[b]))
        self.oof_ridge_ = oof

        # Etapa 2: LightGBM
        X = self._tabla_lgbm(f, oof)
        self.lgbm_ = lgb.LGBMRegressor(random_state=self.semilla, **self.params_lgbm)
        if eval_set is not None:   # parada temprana con un conjunto de validación (texto crudo, y_log)
            fv = self.preparar(eval_set[0])
            Xv = self._tabla_lgbm(fv, self.ridge_.predict(self._matriz_ridge(fv)))
            self.lgbm_.fit(X, y, eval_X=(Xv,), eval_y=(np.asarray(eval_set[1]),),
                           callbacks=[lgb.early_stopping(100, verbose=False)])
        else:
            self.lgbm_.fit(X, y)
        _aligerar(self)
        return self

    def predecir_partes(self, catalog_content) -> pd.DataFrame:
        """Predicciones en escala log de cada etapa (útil para comparar)."""
        f = self.preparar(catalog_content)
        r = self.ridge_.predict(self._matriz_ridge(f))
        g = self.lgbm_.predict(self._tabla_lgbm(f, r))
        return pd.DataFrame({"ridge": r, "lgbm": g, "mezcla": self.peso_ridge * r + (1 - self.peso_ridge) * g})

    def predict(self, catalog_content) -> np.ndarray:
        """Precio estimado en dólares."""
        return np.expm1(self.predecir_partes(catalog_content)["mezcla"].to_numpy())
