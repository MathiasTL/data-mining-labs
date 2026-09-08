"""01 — Descarga de los World Development Indicators desde la API del Banco Mundial.

Genera un CSV por indicador en data/raw/ más paises.csv con los metadatos
(región y grupo de ingreso) necesarios para segmentar el análisis.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import requests

from config import ANIO_MAX, ANIO_MIN, DIR_RAW, INDICADORES
from utils_resultados import guardar

BASE = "https://api.worldbank.org/v2"
TIMEOUT = 60
REINTENTOS = 3


def _pedir(url: str, params: dict) -> list:
    """GET con reintentos; devuelve el segundo elemento de la respuesta JSON."""
    for intento in range(1, REINTENTOS + 1):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            cuerpo = r.json()
            if not isinstance(cuerpo, list) or len(cuerpo) < 2 or cuerpo[1] is None:
                return []
            return cuerpo[1]
        except (requests.RequestException, ValueError) as e:
            if intento == REINTENTOS:
                raise RuntimeError(f"Falló {url} tras {REINTENTOS} intentos: {e}") from e
            time.sleep(2 * intento)
    return []


def descargar_indicador(codigo: str) -> pd.DataFrame:
    filas = _pedir(
        f"{BASE}/country/all/indicator/{codigo}",
        {"format": "json", "per_page": 20000, "date": f"{ANIO_MIN}:{ANIO_MAX}"},
    )
    datos = [
        {
            "iso3": f.get("countryiso3code"),
            "pais": (f.get("country") or {}).get("value"),
            "anio": int(f["date"]),
            "valor": f["value"],
        }
        for f in filas
        if f.get("countryiso3code")
    ]
    return pd.DataFrame(datos)


def descargar_paises() -> pd.DataFrame:
    filas = _pedir(f"{BASE}/country", {"format": "json", "per_page": 400})
    datos = [
        {
            "iso3": f["id"],
            "pais": f["name"],
            "region": (f.get("region") or {}).get("value"),
            "grupo_ingreso": (f.get("incomeLevel") or {}).get("value"),
            "capital": f.get("capitalCity"),
        }
        for f in filas
    ]
    df = pd.DataFrame(datos)
    # El Banco Mundial marca sus agregados regionales con region == "Aggregates"
    df["es_agregado"] = df["region"].eq("Aggregates")
    return df


def main() -> None:
    print(f"Descargando {len(INDICADORES)} indicadores ({ANIO_MIN}-{ANIO_MAX})...\n")

    paises = descargar_paises()
    paises.to_csv(DIR_RAW / "paises.csv", index=False)
    n_paises = int((~paises["es_agregado"]).sum())
    print(f"paises.csv: {len(paises)} entidades ({n_paises} países, "
          f"{len(paises) - n_paises} agregados)\n")

    resumen = {}
    for codigo, (alias, etiqueta, _) in INDICADORES.items():
        df = descargar_indicador(codigo)
        df.to_csv(DIR_RAW / f"{alias}.csv", index=False)
        n_ok = int(df["valor"].notna().sum())
        resumen[alias] = n_ok
        print(f"  {alias:22s} {len(df):6d} filas | {n_ok:6d} con valor | {etiqueta}")

    guardar("datos", "n_indicadores", len(INDICADORES))
    guardar("datos", "n_paises_api", n_paises)
    guardar("datos", "obs_por_indicador", resumen)
    guardar("datos", "periodo", f"{ANIO_MIN}-{ANIO_MAX}")

    # Control de sanidad contra un valor publicado conocido
    le = pd.read_csv(DIR_RAW / "esperanza_vida.csv")
    peru_2019 = le.query("iso3 == 'PER' and anio == 2019")["valor"].iloc[0]
    assert 75.0 < peru_2019 < 77.5, f"Valor inesperado para Perú 2019: {peru_2019}"
    print(f"\nControl de sanidad OK — Perú 2019: {peru_2019:.2f} años")


if __name__ == "__main__":
    main()
