"""Configuración central del Laboratorio 1 — World Development Indicators.

Todas las rutas y constantes del pipeline viven aquí para que los scripts
01..06 compartan exactamente las mismas definiciones.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_RAW = RAIZ / "data" / "raw"
DIR_PROC = RAIZ / "data" / "processed"
DIR_FIG = RAIZ / "figures"
DIR_INFORME = RAIZ / "informe"
RESULTS_JSON = RAIZ / "results.json"

for _d in (DIR_RAW, DIR_PROC, DIR_FIG, DIR_INFORME):
    _d.mkdir(parents=True, exist_ok=True)

ANIO_MIN, ANIO_MAX = 2000, 2023

# Variable objetivo del análisis predictivo
OBJETIVO = "esperanza_vida"

# code -> (alias, etiqueta legible, unidad)
INDICADORES = {
    "SP.DYN.LE00.IN":    ("esperanza_vida",        "Esperanza de vida al nacer",            "años"),
    "NY.GDP.PCAP.PP.KD": ("pib_pc",                "PIB per cápita (PPA, USD 2021)",        "USD"),
    "SH.XPD.CHEX.PC.CD": ("gasto_salud_pc",        "Gasto en salud per cápita",             "USD"),
    "SH.XPD.CHEX.GD.ZS": ("gasto_salud_pib",       "Gasto en salud (% del PIB)",            "%"),
    "SH.MED.PHYS.ZS":    ("medicos_1000",          "Médicos por cada 1.000 habitantes",     "por 1.000"),
    "SP.DYN.IMRT.IN":    ("mort_infantil",         "Mortalidad infantil (<1 año)",          "por 1.000 n.v."),
    "SP.DYN.TFRT.IN":    ("fertilidad",            "Tasa de fertilidad total",              "hijos/mujer"),
    "SE.SEC.ENRR":       ("matricula_secundaria",  "Matrícula secundaria (bruta)",          "%"),
    "SH.IMM.MEAS":       ("inmunizacion",          "Inmunización contra sarampión (12-23m)", "%"),
    "SH.H2O.BASW.ZS":    ("agua_potable",          "Acceso a agua potable básica",          "% pob."),
    "SH.STA.BASS.ZS":    ("saneamiento",           "Acceso a saneamiento básico",           "% pob."),
    "SP.URB.TOTL.IN.ZS": ("urbanizacion",          "Población urbana",                      "% pob."),
    "SP.POP.65UP.TO.ZS": ("poblacion_65",          "Población de 65 años o más",            "% pob."),
    "SP.POP.TOTL":       ("poblacion",             "Población total",                       "hab."),
}

ALIAS = {c: v[0] for c, v in INDICADORES.items()}
ETIQUETAS = {v[0]: v[1] for v in INDICADORES.values()}
UNIDADES = {v[0]: v[2] for v in INDICADORES.values()}

# Predictores candidatos del modelo (población es contexto, no predictor).
# mort_infantil se trata aparte: es un componente mecánico de la esperanza de
# vida, por lo que se reporta en el diagnóstico pero se excluye del modelo
# socioeconómico principal para evitar circularidad.
PREDICTORES = [
    "pib_pc", "gasto_salud_pc", "gasto_salud_pib", "medicos_1000",
    "fertilidad", "matricula_secundaria", "inmunizacion", "agua_potable",
    "saneamiento", "urbanizacion", "poblacion_65",
]
PREDICTOR_CIRCULAR = "mort_infantil"

# Cobertura mínima exigida a un país en la variable objetivo para entrar al panel
COBERTURA_MINIMA = 0.60

PAIS_FOCO = "PER"
REGION_FOCO = "Latin America & Caribbean"

# Paleta categórica de referencia, validada con el validador de la skill dataviz
# (banda de luminosidad, piso de croma, separación para daltonismo y visión normal).
# Orden fijo: los colores se asignan por entidad, nunca ciclados por posición.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

COLORES = {
    "principal": SLOTS[0],   # azul  — serie por defecto / mundo
    "acento": SLOTS[1],      # naranja — Perú y elementos destacados
    "apoyo": SLOTS[2],       # aqua — LAC
    "tinta": "#0b0b0b",
    "tinta_sec": "#52514e",
    "tinta_tenue": "#8a8985",
    "superficie": "#fcfcfb",
    "reja": "#e6e5e1",
}

# 7 regiones en gráficos de líneas/barras: pasa las puertas de pares adyacentes.
PALETA_REGIONES = SLOTS[:7]
# Scatter y coropléticos usan la lista de "todos los pares": máximo 4 slots validados.
PALETA_SCATTER = [SLOTS[0], SLOTS[1], SLOTS[2], SLOTS[6]]

# Rampa secuencial azul (magnitud continua: mapa coroplético, heatmap)
RAMPA_SECUENCIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
                    "#256abf", "#1c5cab", "#184f95", "#0d366b"]
# Rampa divergente azul <-> rojo con gris neutro al medio (polaridad: caída COVID)
RAMPA_DIVERGENTE = ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f5a3a2", "#e34948", "#8f1f1f"]

DPI = 150


# Nombres en español de los países que aparecen en figuras y tablas del informe.
# Si un país no está aquí se usa el nombre original de la API.
PAISES_ES = {
    "Monaco": "Mónaco", "San Marino": "San Marino", "Hong Kong SAR, China": "Hong Kong (China)",
    "Liechtenstein": "Liechtenstein", "Switzerland": "Suiza", "French Polynesia": "Polinesia Francesa",
    "Japan": "Japón", "Andorra": "Andorra", "Spain": "España", "Gibraltar": "Gibraltar",
    "Italy": "Italia", "Singapore": "Singapur", "Australia": "Australia", "Sweden": "Suecia",
    "Norway": "Noruega", "France": "Francia", "Iceland": "Islandia", "Malta": "Malta",
    "Korea, Rep.": "Corea del Sur", "Israel": "Israel", "Luxembourg": "Luxemburgo",
    "Burkina Faso": "Burkina Faso", "Benin": "Benín", "Guinea": "Guinea", "Mali": "Malí",
    "Somalia, Fed. Rep.": "Somalia", "Somalia": "Somalia", "South Sudan": "Sudán del Sur",
    "Central African Republic": "Rep. Centroafricana", "Lesotho": "Lesoto", "Chad": "Chad",
    "Nigeria": "Nigeria", "Niger": "Níger", "Sierra Leone": "Sierra Leona",
    "Cote d'Ivoire": "Costa de Marfil", "Côte d'Ivoire": "Costa de Marfil",
    "Eswatini": "Esuatini", "Mozambique": "Mozambique", "Zimbabwe": "Zimbabue",
    "Equatorial Guinea": "Guinea Ecuatorial", "Cameroon": "Camerún",
    "Guinea-Bissau": "Guinea-Bisáu", "Afghanistan": "Afganistán", "Haiti": "Haití",
    "Peru": "Perú", "Chile": "Chile", "Colombia": "Colombia", "Brazil": "Brasil",
    "Argentina": "Argentina", "Mexico": "México", "Ecuador": "Ecuador", "Bolivia": "Bolivia",
    "Uruguay": "Uruguay", "Paraguay": "Paraguay", "Venezuela, RB": "Venezuela",
    "Costa Rica": "Costa Rica", "Panama": "Panamá", "Cuba": "Cuba",
    "Dominican Republic": "Rep. Dominicana", "Guatemala": "Guatemala", "Honduras": "Honduras",
    "Nicaragua": "Nicaragua", "El Salvador": "El Salvador", "Jamaica": "Jamaica",
    "United States": "Estados Unidos", "Canada": "Canadá", "China": "China", "India": "India",
    "Russian Federation": "Rusia", "United Kingdom": "Reino Unido", "Germany": "Alemania",
    "Bulgaria": "Bulgaria", "Ukraine": "Ucrania", "Belarus": "Bielorrusia",
    "North Macedonia": "Macedonia del Norte", "Serbia": "Serbia", "Bosnia and Herzegovina": "Bosnia y Herzegovina",
    "Slovak Republic": "Eslovaquia", "Czechia": "Chequia", "Poland": "Polonia",
    "Romania": "Rumanía", "Hungary": "Hungría", "Lithuania": "Lituania", "Latvia": "Letonia",
    "Estonia": "Estonia", "Armenia": "Armenia", "Georgia": "Georgia", "Azerbaijan": "Azerbaiyán",
    "Kazakhstan": "Kazajistán", "Egypt, Arab Rep.": "Egipto", "Turkiye": "Turquía",
    "Iran, Islamic Rep.": "Irán", "Iraq": "Irak", "Syrian Arab Republic": "Siria",
    "Yemen, Rep.": "Yemen", "Pakistan": "Pakistán", "Bangladesh": "Bangladés",
    "Indonesia": "Indonesia", "Philippines": "Filipinas", "Vietnam": "Vietnam",
    "Thailand": "Tailandia", "Myanmar": "Myanmar", "South Africa": "Sudáfrica",
    "Kenya": "Kenia", "Ethiopia": "Etiopía", "Tanzania": "Tanzania", "Uganda": "Uganda",
    "Ghana": "Ghana", "Angola": "Angola", "Congo, Dem. Rep.": "R.D. del Congo",
    "Papua New Guinea": "Papúa Nueva Guinea", "Solomon Islands": "Islas Salomón",
    "New Zealand": "Nueva Zelanda", "Ireland": "Irlanda", "Netherlands": "Países Bajos",
    "Belgium": "Bélgica", "Austria": "Austria", "Portugal": "Portugal", "Greece": "Grecia",
    "Finland": "Finlandia", "Denmark": "Dinamarca", "Cyprus": "Chipre", "Croatia": "Croacia",
    "Slovenia": "Eslovenia", "Qatar": "Catar", "United Arab Emirates": "Emiratos Árabes Unidos",
    "Saudi Arabia": "Arabia Saudita", "Kuwait": "Kuwait", "Bahrain": "Baréin", "Oman": "Omán",
    "Morocco": "Marruecos", "Tunisia": "Túnez", "Algeria": "Argelia", "Libya": "Libia",
    "Jordan": "Jordania", "Lebanon": "Líbano", "Malaysia": "Malasia", "Sri Lanka": "Sri Lanka",
    "Nepal": "Nepal", "Bhutan": "Bután", "Maldives": "Maldivas", "Mauritius": "Mauricio",
    "Seychelles": "Seychelles", "Cabo Verde": "Cabo Verde", "Botswana": "Botsuana",
    "Namibia": "Namibia", "Zambia": "Zambia", "Malawi": "Malaui", "Rwanda": "Ruanda",
    "Burundi": "Burundi", "Senegal": "Senegal", "Gambia, The": "Gambia", "Togo": "Togo",
    "Liberia": "Liberia", "Mauritania": "Mauritania", "Sudan": "Sudán", "Madagascar": "Madagascar",
    "Bahamas, The": "Bahamas", "Barbados": "Barbados", "Trinidad and Tobago": "Trinidad y Tobago",
    "Puerto Rico": "Puerto Rico", "Guyana": "Guyana", "Suriname": "Surinam", "Belize": "Belice",
}


def es(nombre: str) -> str:
    """Traduce el nombre de un país al español si está en el diccionario."""
    return PAISES_ES.get(nombre, nombre)
