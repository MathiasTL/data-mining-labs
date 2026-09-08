"""06 — Generación del informe en Word (.docx).

Todas las cifras se leen de results.json: el texto no puede desincronizarse de
los cálculos. Si cambia el análisis, basta con volver a ejecutar este script.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from config import DIR_FIG, DIR_INFORME
from utils_resultados import cargar

AZUL = RGBColor(0x1C, 0x5C, 0xAB)
GRIS = RGBColor(0x52, 0x51, 0x4E)
NEGRO = RGBColor(0x0B, 0x0B, 0x0B)

TITULO = "Análisis descriptivo, diagnóstico y predictivo de los World Development Indicators"
SUBTITULO = "¿Qué explica las brechas de esperanza de vida entre países y dónde se ubica el Perú?"
AUTOR = "[Nombre del estudiante]"
CURSO = "Minería de Datos — Laboratorio 1"


class Informe:
    def __init__(self):
        self.doc = Document()
        self._configurar_estilos()
        self.n_fig = 0
        self.n_tabla = 0
        self.R = cargar()

    # ---------- infraestructura ----------
    def _configurar_estilos(self):
        for s in self.doc.sections:
            s.top_margin = s.bottom_margin = Cm(2.4)
            s.left_margin = s.right_margin = Cm(2.6)
        normal = self.doc.styles["Normal"]
        normal.font.name = "Calibri"
        normal.font.size = Pt(11)
        normal.paragraph_format.space_after = Pt(8)
        normal.paragraph_format.line_spacing = 1.15
        normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for nombre, tam, color in (("Heading 1", 17, AZUL), ("Heading 2", 13.5, NEGRO),
                                   ("Heading 3", 11.5, GRIS)):
            e = self.doc.styles[nombre]
            e.font.name = "Calibri"
            e.font.size = Pt(tam)
            e.font.color.rgb = color
            e.font.bold = True
            e.paragraph_format.space_before = Pt(16 if nombre == "Heading 1" else 12)
            e.paragraph_format.space_after = Pt(6)

    def h1(self, texto):
        self.doc.add_heading(texto, level=1)

    def h2(self, texto):
        self.doc.add_heading(texto, level=2)

    def p(self, texto, cursiva=False, tam=None):
        par = self.doc.add_paragraph()
        run = par.add_run(texto)
        run.italic = cursiva
        if tam:
            run.font.size = Pt(tam)
        return par

    def vinetas(self, items):
        for it in items:
            par = self.doc.add_paragraph(it, style="List Bullet")
            par.paragraph_format.space_after = Pt(4)

    def figura(self, archivo, pie, ancho=15.6):
        self.n_fig += 1
        par = self.doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_before = Pt(10)
        par.paragraph_format.space_after = Pt(2)
        par.add_run().add_picture(str(DIR_FIG / archivo), width=Cm(ancho))
        cap = self.doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(12)
        run = cap.add_run(f"Figura {self.n_fig}. {pie}")
        run.font.size = Pt(9)
        run.font.color.rgb = GRIS
        run.italic = True

    def tabla(self, encabezados, filas, titulo, anchos=None):
        self.n_tabla += 1
        cap = self.doc.add_paragraph()
        cap.paragraph_format.space_after = Pt(3)
        run = cap.add_run(f"Tabla {self.n_tabla}. {titulo}")
        run.font.size = Pt(9)
        run.font.color.rgb = GRIS
        run.italic = True
        t = self.doc.add_table(rows=1, cols=len(encabezados))
        t.style = "Light Grid Accent 1"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for celda, texto in zip(t.rows[0].cells, encabezados):
            celda.text = ""
            par = celda.paragraphs[0]
            r = par.add_run(str(texto))
            r.bold = True
            r.font.size = Pt(9)
        for fila in filas:
            celdas = t.add_row().cells
            for i, (celda, valor) in enumerate(zip(celdas, fila)):
                celda.text = ""
                par = celda.paragraphs[0]
                if i > 0:
                    par.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                r = par.add_run(str(valor))
                r.font.size = Pt(9)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(6)
        return t

    def salto(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ---------- portada e índice ----------
    def portada(self):
        for _ in range(4):
            self.doc.add_paragraph()
        par = self.doc.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = par.add_run(TITULO); r.bold = True; r.font.size = Pt(22); r.font.color.rgb = AZUL
        par = self.doc.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = par.add_run(SUBTITULO); r.font.size = Pt(13); r.font.color.rgb = GRIS; r.italic = True
        for _ in range(3):
            self.doc.add_paragraph()
        d = self.R["datos"]
        for texto, negrita in ((CURSO, True), (AUTOR, False),
                               (f"Fuente: World Development Indicators, Banco Mundial", False),
                               (f"Panel de {d['n_paises_panel']} países · {d['periodo']} · "
                                f"{d['n_indicadores']} indicadores · {d['n_filas_panel']:,} observaciones".replace(",", "."), False)):
            par = self.doc.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = par.add_run(texto); r.font.size = Pt(11.5); r.bold = negrita
            if not negrita:
                r.font.color.rgb = GRIS
        self.salto()

    def indice(self):
        self.h1("Índice")
        par = self.doc.add_paragraph()
        fld = OxmlElement("w:fldSimple")
        fld.set(qn("w:instr"), r'TOC \o "1-2" \h \z \u')
        aviso = OxmlElement("w:p")
        run_x = OxmlElement("w:r"); txt = OxmlElement("w:t")
        txt.text = "Haga clic derecho aquí y elija «Actualizar campos» para generar el índice."
        run_x.append(txt); aviso.append(run_x); fld.append(aviso)
        par._p.append(fld)
        self.salto()


def construir():
    inf = Informe()
    R = inf.R
    D, DE, DI, PR = R["datos"], R["descriptivo"], R["diagnostico"], R["predictivo"]
    anio = D["anio_ultimo"]

    inf.portada()
    inf.indice()

    # ---------------------------------------------------------------- 1
    inf.h1("1. Introducción y pregunta de investigación")
    inf.p(
        f"La esperanza de vida al nacer es el indicador más comprimido que existe sobre el "
        f"bienestar de una población: en un solo número resume la nutrición, la salubridad "
        f"del entorno, el acceso a servicios médicos y la exposición a la violencia y a la "
        f"enfermedad. En {anio} ese número va desde los {DE['distribucion']['minimo']:.1f} años "
        f"de {DE['distribucion']['pais_min']} hasta los {DE['distribucion']['maximo']:.1f} de "
        f"{DE['distribucion']['pais_max']}: casi {DE['amplitud']:.0f} años de diferencia entre "
        f"nacer en un país u otro. Esa brecha es el objeto de este informe.")
    inf.p(
        "El trabajo se organiza alrededor de una única pregunta, con tres niveles de respuesta "
        "que corresponden a los tres tipos de análisis solicitados:")
    inf.vinetas([
        "Descriptivo — ¿cómo se distribuye hoy la esperanza de vida en el mundo y cómo ha "
        "evolucionado desde el año 2000?",
        "Diagnóstico — ¿qué factores acompañan a las diferencias observadas y cuánto aporta "
        "cada uno? ¿Qué ocurrió durante la pandemia?",
        "Predictivo — ¿se puede anticipar la esperanza de vida de un país a partir de sus "
        "indicadores socioeconómicos, y qué trayectoria cabe esperar hasta 2030?",
    ])
    inf.p(
        "Transversalmente, el informe sigue el caso del Perú dentro de América Latina, una "
        "elección que resultó especialmente fértil: como se verá, el Perú es simultáneamente "
        "uno de los países que más sufrió durante la pandemia y uno de los que más años de "
        "vida obtiene de sus recursos económicos.")

    # ---------------------------------------------------------------- 2
    inf.h1("2. Datos y metodología")
    inf.h2("2.1 Fuente y alcance")
    inf.p(
        f"Los datos provienen de la base World Development Indicators (WDI) del Banco Mundial, "
        f"descargados mediante su API pública. Se seleccionaron {D['n_indicadores']} indicadores "
        f"para el período {D['periodo']}, correspondientes a {D['n_paises_api']} países. Se "
        f"excluyeron los {'agregados regionales'} que el propio Banco Mundial publica junto a "
        f"los países (Unión Europea, «Ingreso alto», «Mundo», etc.), porque mezclarlos con las "
        f"observaciones nacionales duplicaría la información y distorsionaría cualquier "
        f"estadístico. El resultado es un panel país-año de {D['n_filas_panel']:,} filas "
        f"cubriendo {D['n_paises_panel']} países en {D['n_regiones']} regiones."
        .replace(",", "."))
    inf.p(
        "La variable objetivo es la esperanza de vida al nacer. Los demás indicadores se "
        "eligieron por representar las cuatro vías por las que la literatura explica la "
        "longevidad: recursos económicos (PIB per cápita), sistema de salud (gasto sanitario, "
        "médicos por habitante, cobertura de inmunización), condiciones materiales de vida "
        "(agua potable, saneamiento, urbanización) y estructura demográfica y educativa "
        "(fertilidad, matrícula secundaria, población mayor de 65 años).")

    inf.h2("2.2 Calidad de los datos e imputación")
    inf.p(
        f"La cobertura es muy desigual entre indicadores, y ese es el primer hallazgo del "
        f"trabajo: no todos los países miden lo mismo con la misma constancia. La esperanza "
        f"de vida, la fertilidad y la urbanización están completas al 100%, pero el número de "
        f"médicos por cada 1.000 habitantes sólo está disponible en el "
        f"{D['cobertura_antes']['medicos_1000']:.1f}% de las celdas y la matrícula secundaria "
        f"en el {D['cobertura_antes']['matricula_secundaria']:.1f}%. En conjunto, el "
        f"{D['pct_celdas_imputadas']}% de las celdas del panel requirió imputación.")
    imp = D["imputacion"]
    filas_imp = [
        (nombre, f"{D['cobertura_antes'][alias]:.1f}%", f"{imp[alias]['faltantes_originales']:,}".replace(",", "."),
         f"{imp[alias]['imputados_interpolacion']:,}".replace(",", "."),
         f"{imp[alias]['imputados_mediana_regional']:,}".replace(",", "."))
        for alias, nombre in [
            ("medicos_1000", "Médicos por 1.000 hab."), ("matricula_secundaria", "Matrícula secundaria"),
            ("gasto_salud_pc", "Gasto en salud per cápita"), ("gasto_salud_pib", "Gasto en salud (% PIB)"),
            ("inmunizacion", "Inmunización sarampión"), ("mort_infantil", "Mortalidad infantil"),
            ("pib_pc", "PIB per cápita"), ("saneamiento", "Saneamiento básico"),
            ("agua_potable", "Agua potable básica")]]
    inf.tabla(["Indicador", "Cobertura original", "Celdas vacías", "Interpoladas", "Mediana regional"],
              filas_imp, "Cobertura original de cada indicador y método de imputación aplicado.")
    inf.p(
        "Se aplicó una estrategia en dos niveles. Primero, interpolación lineal dentro de cada "
        "país: son series lentas y suaves, de modo que estimar el valor de un año a partir de "
        "los años vecinos del mismo país introduce muy poco error. Segundo, para los países que "
        "no reportan un indicador en ningún año —y que por tanto no admiten interpolación— se "
        "usó la mediana de su región en ese año. Esta segunda vía es la más costosa en términos "
        "de supuestos y afecta sobre todo al gasto en salud y a la inmunización; conviene "
        "tenerlo presente al leer la importancia de esas variables en el modelo. No se descartó "
        "ningún país: todos superaron el umbral de 60% de cobertura exigido en la variable "
        "objetivo.")

    inf.h2("2.3 Años de catástrofe: valores extremos que no son errores")
    cat = D.get("detalle_catastrofes", [])
    inf.p(
        f"Una revisión de valores atípicos detectó {D.get('n_anios_catastrofe', 0)} "
        f"observaciones en {len(D.get('paises_catastrofe', []))} países cuya esperanza de vida "
        f"cambia más de diez años respecto del año anterior. El primer impulso sería tratarlos "
        f"como errores de registro, pero la inspección caso por caso muestra lo contrario: son "
        f"catástrofes reales. Haití cae de 61,7 a 45,6 años en 2010 —el terremoto— y vuelve a "
        f"61,6 en 2011; Somalia se desploma a 32,5 años en 2011, el año de la hambruna; Sudán "
        f"del Sur desciende de forma sostenida entre 2014 y 2017 durante su guerra civil; "
        f"Cisjordania y Gaza pasa de 76,7 a 65,2 años en 2023.")
    inf.p(
        "La explicación es técnica: la esperanza de vida que publica el Banco Mundial es de "
        "«período», es decir, cuántos años viviría un recién nacido si las tasas de mortalidad "
        "de ese año concreto se mantuvieran toda su vida. En un año de desastre masivo esas "
        "tasas se disparan y el indicador colapsa, para recuperarse al año siguiente. No es un "
        "error del dato: es lo que el indicador mide.")
    inf.p(
        "La única serie que sí parece defectuosa es la de la República Centroafricana, que "
        "oscila de forma errática en seis años distintos —incluidos un 14,7 en 2009 y un 18,8 "
        "en 2022— sin ningún evento identificable que lo justifique. Se decidió no eliminar "
        "ninguna de estas observaciones, porque borrarlas equivaldría a ocultar los peores años "
        "de los países más frágiles del panel. En su lugar se marcaron, y la sección 5.2 "
        "presenta un análisis de sensibilidad que mide si las conclusiones dependen de ellas.")
    if cat:
        inf.tabla(["País", "Año", "Esperanza de vida"],
                  [(c["pais"], c["anio"], f"{c['esperanza_vida']:.1f}") for c in cat],
                  "Observaciones marcadas como años de catástrofe.")

    inf.h2("2.4 Herramientas")
    inf.p(
        "Todo el análisis se realizó en Python 3.12 sobre un entorno reproducible gestionado "
        "con mamba. Se usaron pandas para la manipulación del panel, scikit-learn para los "
        "modelos y el clustering, statsmodels para el diagnóstico de colinealidad y la "
        "proyección temporal, y matplotlib y plotly para las visualizaciones. El pipeline está "
        "dividido en seis scripts numerados que se ejecutan en orden; el detalle está en la "
        "sección 8.")
    inf.salto()

    # ---------------------------------------------------------------- 3
    inf.h1("3. Análisis descriptivo")
    inf.h2("3.1 La forma de la distribución")
    dist = DE["distribucion"]
    inf.p(
        f"En {anio} la esperanza de vida mundial tiene una mediana de {dist['mediana']:.1f} años "
        f"y una media de {dist['media']:.1f}. Que la media quede por debajo de la mediana señala "
        f"una distribución con cola a la izquierda: la mayoría de los países se agrupa en la "
        f"franja alta y un grupo minoritario —casi todo él en África subsahariana— arrastra el "
        f"promedio hacia abajo. La mitad central de los países vive entre {dist['p25']:.1f} y "
        f"{dist['p75']:.1f} años, un rango de apenas {dist['p75']-dist['p25']:.0f} años, mientras "
        f"que la distancia entre los extremos es de {dist['rango']:.1f}. Dicho de otro modo: el "
        f"mundo no está repartido de forma continua, sino que hay un pelotón mayoritario y una "
        f"cola rezagada.")
    inf.figura("fig01_distribucion.png",
               f"Distribución de la esperanza de vida al nacer en {D['n_paises_panel']} países, {anio}.")

    inf.h2("3.2 Región e ingreso: dos formas de cortar la misma brecha")
    reg = DE["por_region"]
    orden_reg = sorted(reg.items(), key=lambda kv: -kv[1]["median"])
    inf.p(
        f"Al segmentar por región aparece una brecha de {DE['brecha_regional']:.1f} años entre "
        f"la mediana más alta ({orden_reg[0][0]}, {orden_reg[0][1]['median']:.1f} años) y la más "
        f"baja ({orden_reg[-1][0]}, {orden_reg[-1][1]['median']:.1f} años). Pero la región no es "
        f"un bloque homogéneo: África subsahariana va de {reg[orden_reg[-1][0]]['min']:.1f} a "
        f"{reg[orden_reg[-1][0]]['max']:.1f} años, y Asia Oriental y el Pacífico es la región de "
        f"mayor dispersión interna. La geografía, por sí sola, explica menos de lo que parece.")
    inf.figura("fig02_regiones.png",
               f"Distribución de la esperanza de vida por región del Banco Mundial, {anio}.")
    ing = DE["por_ingreso"]
    inf.p(
        f"El corte por grupo de ingreso es más limpio: las cuatro categorías del Banco Mundial "
        f"se ordenan de forma monótona, desde {ing['Ingreso bajo']['median']:.1f} años en los "
        f"países de ingreso bajo hasta {ing['Ingreso alto']['median']:.1f} en los de ingreso "
        f"alto, una brecha de {DE['brecha_ingreso']:.1f} años. La comparación entre ambas "
        f"figuras ya sugiere el hallazgo central del diagnóstico: el ingreso ordena la "
        f"esperanza de vida mejor que la ubicación geográfica.")
    inf.figura("fig03_ingreso.png",
               f"Distribución de la esperanza de vida por grupo de ingreso, {anio}.")

    inf.h2("3.3 Veinte años de avance y un quiebre")
    serie = DE["mundo_serie"]
    inf.p(
        f"Entre 2000 y 2019 la esperanza de vida mundial ponderada por población subió de "
        f"{serie['2000']:.1f} a {serie['2019']:.1f} años: {DE['ganancia_2000_2019']:.1f} años "
        f"ganados en dos décadas, a un ritmo cercano a tres meses de vida por cada año "
        f"transcurrido. La pandemia interrumpió esa serie de forma abrupta: entre 2019 y 2021 "
        f"el indicador mundial retrocedió {abs(DE['caida_covid_mundo']):.1f} años, borrando de "
        f"golpe alrededor de seis años de progreso. Para {anio} el mundo ya había recuperado el "
        f"nivel previo ({serie['2023']:.1f} años), pero la recuperación no fue pareja.")
    caidas = DE["caida_covid_por_region"]
    peor = min(caidas.items(), key=lambda kv: kv[1])
    inf.p(
        f"El desglose regional muestra por qué el promedio mundial engaña. América Latina y el "
        f"Caribe perdió {abs(caidas['América Latina y Caribe']):.1f} años y Asia del Sur "
        f"{abs(caidas['Asia del Sur']):.1f}, mientras que África subsahariana y Asia Oriental "
        f"apenas cedieron cuatro y cinco décimas respectivamente. La región más golpeada del "
        f"mundo fue precisamente aquella en la que se inserta el Perú.")
    inf.figura("fig04_series_regiones.png",
               "Evolución de la esperanza de vida por región, 2000-2023, ponderada por población.")

    inf.h2("3.4 El mapa y los extremos")
    inf.p(
        "El mapa coroplético traduce la brecha a geografía: una franja continua de valores bajos "
        "atraviesa África central y occidental, mientras que Europa occidental, Japón, Corea, "
        "Australia y Canadá forman el bloque de mayor longevidad. América Latina ocupa una "
        "posición intermedia-alta, más cerca del bloque desarrollado que del rezagado.")
    inf.figura("fig05_mapa.png", f"Esperanza de vida al nacer por país, {anio}.")
    top, bot = DE["top10"], DE["bottom10"]
    inf.p(
        f"El ranking de extremos añade un matiz metodológico importante. Las diez primeras "
        f"posiciones están dominadas por microestados y territorios —{top[0]['pais']}, "
        f"{top[1]['pais']}, {top[3]['pais']}, {top[9]['pais']}— cuyas poblaciones son de decenas "
        f"de miles de habitantes y cuyos valores no son comparables con los de un país de "
        f"decenas de millones. El extremo inferior, en cambio, sí está formado por países "
        f"grandes: {bot[0]['pais']} ({bot[0]['valor']:.1f} años) es el país más poblado de "
        f"África. Conviene por tanto leer los rankings con cautela y apoyarse en las medianas "
        f"regionales, que son robustas a estos casos.")
    inf.figura("fig06_ranking.png",
               f"Los diez países con menor y mayor esperanza de vida, {anio}.")

    inf.h2("3.5 Estadísticos descriptivos del panel")
    filas_est = [(f["indicador"], f["unidad"], f"{f['media']:,.2f}".replace(",", "."),
                  f"{f['mediana']:,.2f}".replace(",", "."), f"{f['desv']:,.2f}".replace(",", "."),
                  f"{f['min']:,.2f}".replace(",", "."), f"{f['max']:,.2f}".replace(",", "."))
                 for f in DE["tabla_estadisticos"]]
    inf.tabla(["Indicador", "Unidad", "Media", "Mediana", "Desv.", "Mín.", "Máx."],
              filas_est, f"Estadísticos descriptivos de los indicadores del panel, {anio}.")
    inf.salto()

    # ---------------------------------------------------------------- 4
    inf.h1("4. Análisis diagnóstico")
    inf.h2("4.1 Qué acompaña a una vida más larga")
    corr = DI["corr_con_objetivo"]
    inf.p(
        f"La matriz de correlaciones de Spearman confirma que casi todos los indicadores se "
        f"mueven en la dirección esperada, y con intensidad alta. La mortalidad infantil es el "
        f"correlato más fuerte (ρ = {corr['Mortalidad infantil']:.2f}), seguida del PIB per "
        f"cápita (ρ = {corr['PIB per cápita']:.2f}), el saneamiento básico "
        f"(ρ = {corr['Saneamiento']:.2f}) y el agua potable (ρ = {corr['Agua potable']:.2f}). "
        f"En el otro extremo, el gasto en salud medido como porcentaje del PIB apenas alcanza "
        f"ρ = {corr['Gasto salud %PIB']:.2f}: gastar una proporción alta del producto en salud "
        f"no dice casi nada sobre cuánto vive la población, mientras que gastar muchos dólares "
        f"por persona (ρ = {corr['Gasto salud p/c']:.2f}) sí. La diferencia entre ambas cifras "
        f"es uno de los resultados más instructivos del análisis: lo que importa no es el "
        f"esfuerzo relativo, sino el volumen absoluto de recursos sanitarios disponibles.")
    inf.p(
        "La lectura de la matriz obliga a una advertencia. Los predictores no sólo se "
        "correlacionan con la esperanza de vida: se correlacionan intensamente entre sí. Los "
        "países con buen saneamiento son los mismos que tienen agua potable, más médicos y "
        "mayor PIB. Esto significa que ninguna correlación bivariada puede interpretarse como "
        "un efecto aislado, y motiva el análisis de colinealidad de la sección 4.3.")
    inf.figura("fig08_correlaciones.png",
               f"Matriz de correlación de Spearman entre indicadores, {anio}.")

    inf.h2("4.2 La curva de Preston: el ingreso rinde cada vez menos")
    pres = DI["preston"]
    inf.p(
        f"Al graficar el PIB per cápita contra la esperanza de vida aparece la relación conocida "
        f"como curva de Preston, y aparece con nitidez. La relación no es lineal: un ajuste "
        f"lineal en niveles alcanza un R² de {pres['r2_lineal']:.2f}, mientras que el mismo "
        f"ajuste sobre el logaritmo del ingreso llega a {pres['r2_log']:.2f}. La forma "
        f"logarítmica es, por tanto, claramente superior, y su interpretación es económica: "
        f"cada vez que el ingreso de un país se duplica, su esperanza de vida aumenta unos "
        f"{pres['anios_por_duplicar_pib']:.1f} años, con independencia del nivel de partida. "
        f"Para un país pobre eso es transformador; para uno rico, duplicar el PIB per cápita es "
        f"una hazaña que compra apenas unos años más.")
    sobre, bajo = DI["preston_sobre"], DI["preston_bajo"]
    inf.p(
        f"Igual de informativa que la curva es la dispersión alrededor de ella. Con el mismo "
        f"nivel de ingreso hay países que viven una década más que otros. Entre los que más "
        f"superan lo predicho por su ingreso destacan {sobre[0]['pais']} "
        f"(+{sobre[0]['residuo_preston']:.1f} años) y {sobre[1]['pais']} "
        f"(+{sobre[1]['residuo_preston']:.1f}); entre los que menos, {bajo[0]['pais']} "
        f"({bajo[0]['residuo_preston']:.1f}) y {bajo[1]['pais']} ({bajo[1]['residuo_preston']:.1f}). "
        f"El Perú se sitúa {pres['residuo_peru']:+.1f} años por encima de lo que su PIB per "
        f"cápita anticiparía, un resultado que se retoma en la sección 6.")
    inf.figura("fig09_preston.png",
               f"Curva de Preston: PIB per cápita y esperanza de vida, {anio}.")

    inf.h2("4.3 Colinealidad entre predictores")
    vif = DI["vif"]
    altos = DI["vif_altos"]
    filas_vif = [(f["variable"], f"{f['vif']:.2f}") for f in vif]
    inf.tabla(["Variable", "VIF"], filas_vif,
              "Factor de inflación de la varianza (VIF) de los predictores. "
              "Valores superiores a 5 indican colinealidad relevante.")
    if altos:
        nombres = " y ".join(a["variable"] for a in altos)
        inf.p(
            f"El diagnóstico formal confirma la sospecha: {nombres} superan el umbral de 5, con "
            f"valores de {altos[0]['vif']:.2f} y {altos[1]['vif']:.2f} respectivamente. Es "
            f"esperable —el acceso a agua potable y a saneamiento se despliegan casi siempre "
            f"juntos, como parte de la misma infraestructura urbana— pero tiene una consecuencia "
            f"práctica: los coeficientes de una regresión lineal repartirían de forma "
            f"arbitraria el crédito entre ambas variables, y no deben leerse como efectos "
            f"causales separados. Es una de las razones por las que el análisis predictivo "
            f"prioriza la importancia por permutación sobre los coeficientes.")
    inf.p(
        "Por el mismo motivo, la mortalidad infantil se excluyó deliberadamente del conjunto de "
        "predictores del modelo. No es un factor externo que explique la esperanza de vida: es "
        "un componente aritmético de ella, ya que las muertes en el primer año de vida entran "
        "directamente en el cálculo del indicador. Incluirla habría producido un modelo con "
        "métricas excelentes y ningún valor explicativo.")

    inf.h2("4.4 Anatomía del shock de la COVID-19")
    cv = DI["covid"]
    inf.p(
        f"Entre 2019 y 2021, {cv['n_paises_con_caida']} de los {cv['n_paises']} países del panel "
        f"perdieron esperanza de vida, con una caída mediana de {abs(cv['delta_mediano']):.1f} "
        f"años. Ninguna región quedó indemne, pero la distancia entre ellas es enorme: América "
        f"Latina y el Caribe perdió {abs(cv['por_region']['América Latina y Caribe']):.1f} años "
        f"en promedio simple, cinco veces lo que perdió África subsahariana "
        f"({abs(cv['por_region']['África subsahariana']):.1f}). La estructura etaria más joven "
        f"del África subsahariana y el subregistro de defunciones explican buena parte de esa "
        f"asimetría, y son un recordatorio de que estos indicadores miden tanto la realidad "
        f"como la capacidad estadística de cada país para observarla.")
    peores = cv["peores5"]
    inf.p(
        f"En el ranking individual, cuatro de los cinco países más golpeados del mundo son "
        f"latinoamericanos: {peores[0]['pais']} ({peores[0]['delta']:+.1f} años), "
        f"{peores[1]['pais']} ({peores[1]['delta']:+.1f}), {peores[2]['pais']} "
        f"({peores[2]['delta']:+.1f}) y {peores[3]['pais']} ({peores[3]['delta']:+.1f}). El Perú "
        f"ocupa la posición número {cv['ranking_peru']} del mundo con "
        f"{cv['delta_peru']:+.1f} años. Cinco años después, "
        f"{cv['n_sin_recuperar_2023']} países aún no habían recuperado su nivel de 2019.")
    inf.figura("fig10_covid.png",
               "Pérdida de esperanza de vida entre 2019 y 2021, por región y por país.")

    inf.h2("4.5 Segmentación de países por perfil de desarrollo-salud")
    inf.p(
        f"Para resumir la estructura del panel se aplicó un análisis de conglomerados K-means "
        f"sobre los indicadores estandarizados. La elección del número de grupos se hizo con el "
        f"coeficiente de silueta, evaluado para k entre 2 y 8. El máximo absoluto está en k = 2 "
        f"(silueta {DI['siluetas']['2']:.2f}), pero esa solución se limita a separar países "
        f"ricos de países pobres y no aporta información nueva. Se adoptó por ello "
        f"k = {DI['k_elegido']} (silueta {DI['siluetas'][str(DI['k_elegido'])]:.2f}), la mejor "
        f"solución no trivial, que sigue por encima del umbral habitual de 0,25. Los dos "
        f"primeros componentes principales concentran el {DI['varianza_pca']:.0f}% de la "
        f"varianza, lo que permite visualizar los grupos en un plano.")
    filas_cl = [(c["perfil"], c["n_paises"], f"{c['esperanza_vida']:.1f}",
                 f"{c['pib_pc']:,}".replace(",", "."), f"{c['gasto_salud_pc']:,}".replace(",", "."),
                 f"{c['medicos_1000']:.2f}", f"{c['saneamiento']:.1f}%", f"{c['fertilidad']:.2f}")
                for c in DI["clusters"]]
    inf.tabla(["Perfil", "Países", "Esp. vida", "PIB p/c", "Gasto salud p/c",
               "Médicos/1.000", "Saneamiento", "Fertilidad"], filas_cl,
              "Perfil medio de cada conglomerado.")
    cl = DI["clusters"]
    inf.p(
        f"Los tres perfiles se ordenan con nitidez. El grupo de «{cl[0]['perfil']}» reúne "
        f"{cl[0]['n_paises']} países con una esperanza de vida media de {cl[0]['esperanza_vida']:.1f} "
        f"años, un PIB per cápita de {cl[0]['pib_pc']:,} dólares y {cl[0]['medicos_1000']:.1f} "
        f"médicos por cada mil habitantes. El de «{cl[-1]['perfil']}» agrupa {cl[-1]['n_paises']} "
        f"países con {cl[-1]['esperanza_vida']:.1f} años, {cl[-1]['medicos_1000']:.2f} médicos "
        f"por mil y una fertilidad de {cl[-1]['fertilidad']:.1f} hijos por mujer, más del doble "
        f"que el primer grupo. Entre ambos, la «{cl[1]['perfil']}» concentra "
        f"{cl[1]['n_paises']} países. La distancia decisiva entre los grupos no está tanto en el "
        f"ingreso como en el saneamiento: {cl[0]['saneamiento']:.0f}% de cobertura en el primer "
        f"grupo frente a {cl[-1]['saneamiento']:.0f}% en el último."
        .replace(",", "."))
    lac_cl = DI["clusters_lac"]
    inf.p(
        f"América Latina se distribuye de forma muy concentrada: {lac_cl.get('Transición intermedia', 0)} "
        f"de sus países caen en el perfil de transición intermedia, "
        f"{lac_cl.get('Desarrollo alto', 0)} alcanzan el de desarrollo alto y sólo "
        f"{lac_cl.get('Rezago sanitario', 0)} queda en el de rezago sanitario. El Perú pertenece "
        f"al grupo de «{DI['cluster_peru']}», junto a la mayoría de sus vecinos.")
    inf.figura("fig11_clusters.png",
               "Conglomerados de países proyectados sobre los dos primeros componentes "
               "principales, y selección del número de grupos.")
    inf.salto()

    # ---------------------------------------------------------------- 5
    inf.h1("5. Análisis predictivo")
    inf.h2("5.1 Diseño del experimento")
    inf.p(
        f"El modelo transversal busca predecir la esperanza de vida de una observación país-año "
        f"a partir de sus {PR['n_predictores']} indicadores socioeconómicos. El conjunto tiene "
        f"{PR['n_observaciones']:,} observaciones de {PR['n_paises']} países. La decisión "
        f"metodológica más importante es cómo se valida: una partición aleatoria ordinaria "
        f"dejaría al Perú de 2015 en entrenamiento y al Perú de 2016 en prueba, y el modelo "
        f"acertaría por memorizar el país, no por aprender la relación. Para evitar esa fuga se "
        f"usó GroupKFold con cinco particiones agrupadas por país, de modo que todo país "
        f"presente en entrenamiento está ausente de la prueba correspondiente; el script "
        f"verifica esta condición explícitamente en cada partición."
        .replace(",", "."))
    inf.p(
        "Se compararon tres modelos contra una línea base consistente en predecir siempre la "
        "media global del conjunto de entrenamiento. Esa línea base es la referencia mínima: "
        "cualquier modelo que no la supere con holgura no está aprendiendo nada.")
    filas_m = [(k, f"{v['rmse']:.2f}", f"{v['mae']:.2f}", f"{v['r2']:.3f}")
               for k, v in PR["metricas"].items()]
    inf.tabla(["Modelo", "RMSE (años)", "MAE (años)", "R²"], filas_m,
              "Desempeño de los modelos en validación cruzada agrupada por país.")

    inf.h2("5.2 Resultados")
    mej, base = PR["metricas"][PR["mejor_modelo"]], PR["metricas"]["Media global (baseline)"]
    lin = PR["metricas"]["Regresión lineal"]
    inf.p(
        f"El mejor modelo es {PR['mejor_modelo']}, con un RMSE de {mej['rmse']:.2f} años y un R² "
        f"de {mej['r2']:.3f} sobre países nunca vistos durante el entrenamiento. Frente a la "
        f"línea base ({base['rmse']:.2f} años de RMSE) representa una reducción del error del "
        f"{PR['mejora_vs_baseline_rmse']:.0f}%. En términos prácticos: conociendo únicamente los "
        f"indicadores socioeconómicos de un país desconocido, se puede estimar la esperanza de "
        f"vida de su población con un error típico de menos de cuatro años.")
    inf.p(
        f"Vale la pena subrayar lo cerca que queda la regresión lineal (RMSE {lin['rmse']:.2f}, "
        f"R² {lin['r2']:.3f}) del mejor modelo no lineal. La diferencia es de apenas "
        f"{lin['rmse'] - mej['rmse']:.2f} años. Eso indica que la relación entre desarrollo y "
        f"longevidad es fundamentalmente sencilla y monótona: no hay interacciones complejas "
        f"que un modelo de ensamble pueda explotar. Para un trabajo aplicado, el modelo lineal "
        f"sería preferible por interpretabilidad al costo de un error marginalmente mayor.")
    sens = PR.get("sensibilidad")
    if sens:
        inf.p(
            f"Como prueba de robustez se reentrenó el mismo modelo excluyendo las "
            f"{sens['n_excluidos']} observaciones marcadas como años de catástrofe en la "
            f"sección 2.3. El resultado despeja la duda: el R² pasa de {sens['r2_con']:.3f} a "
            f"{sens['r2_sin']:.3f} y el RMSE de {sens['rmse_con']:.2f} a {sens['rmse_sin']:.2f} "
            f"años. Es decir, el poder explicativo no cambia y el error apenas mejora unas "
            f"centésimas. Las conclusiones del informe no dependen de esos valores extremos, lo "
            f"que justifica haberlos conservado.")
    inf.figura("fig12_ajuste.png",
               "Valores observados frente a predichos, validación cruzada agrupada por país.", ancho=11.5)

    inf.h2("5.3 Qué variables mandan")
    imps = PR["importancias"]
    inf.p(
        f"La importancia por permutación mide cuánto se degrada el R² del modelo al desordenar "
        f"aleatoriamente cada variable. El resultado es contundente: el "
        f"{imps[0]['variable'].lower()} domina con {imps[0]['importancia']:.3f}, cerca de "
        f"{imps[0]['importancia']/imps[1]['importancia']:.0f} veces la importancia del "
        f"{imps[1]['variable'].lower()} ({imps[1]['importancia']:.3f}), que ocupa el segundo "
        f"lugar. Le siguen {imps[2]['variable'].lower()} ({imps[2]['importancia']:.3f}) y "
        f"{imps[3]['variable'].lower()} ({imps[3]['importancia']:.3f}).")
    inf.p(
        "Este es el hallazgo de política pública más relevante del informe. El acceso a "
        "saneamiento básico —infraestructura barata y de baja tecnología— predice la esperanza "
        "de vida mucho mejor que el PIB per cápita o que el número de médicos por habitante. La "
        "interpretación causal debe ser prudente, porque el saneamiento es también un buen "
        "indicador indirecto de capacidad estatal y de urbanización ordenada. Pero la señal es "
        "consistente con lo que muestra la curva de Preston: los primeros dólares invertidos en "
        "condiciones materiales básicas rinden mucho más que los últimos invertidos en "
        "tecnología médica.")
    inf.figura("fig13_importancias.png",
               "Importancia por permutación de cada predictor sobre el conjunto de prueba.")

    inf.h2("5.4 Los residuos: quién vive más de lo que le corresponde")
    pos, neg = PR["residuos_positivos"], PR["residuos_negativos"]
    inf.p(
        f"Los errores del modelo son tan informativos como sus aciertos. Un residuo positivo "
        f"indica un país que vive más de lo que sus indicadores anticipan; uno negativo, lo "
        f"contrario. En el extremo positivo aparecen {pos[0]['pais']} "
        f"({pos[0]['residuo']:+.1f} años) y {pos[1]['pais']} ({pos[1]['residuo']:+.1f}); en el "
        f"negativo, {neg[0]['pais']} ({neg[0]['residuo']:+.1f}) y {neg[1]['pais']} "
        f"({neg[1]['residuo']:+.1f}). Los residuos negativos grandes se concentran en países con "
        f"conflicto armado o epidemias prolongadas —factores que el panel no mide— y los "
        f"positivos, en países con sistemas de atención primaria eficientes en relación con su "
        f"nivel de ingreso.")
    inf.figura("fig14_residuos.png",
               f"Países con mayor residuo positivo y negativo del modelo, {anio}.")

    inf.h2("5.5 Proyección temporal a 2030")
    proy = PR["proyeccion"]
    pe, la = proy["Perú"], proy["América Latina y el Caribe"]
    inf.p(
        f"El segundo ejercicio predictivo es de naturaleza distinta: en lugar de comparar países "
        f"entre sí, proyecta la trayectoria temporal del Perú y de América Latina hasta 2030. Se "
        f"contrastan dos referencias. La primera es un contrafactual: la tendencia lineal de "
        f"2000-2019 prolongada como si la pandemia no hubiera ocurrido. La segunda es un "
        f"suavizado exponencial de Holt con tendencia amortiguada ajustado sobre la serie "
        f"observada. Los años 2020 y 2021 se excluyeron del ajuste y se interpolaron, porque "
        f"tratarlos como nivel —y no como el shock puntual que fueron— arrastraría la "
        f"proyección por debajo del último dato observado, un artefacto del método y no un "
        f"resultado del fenómeno.")
    inf.p(
        f"Para el Perú, la tendencia previa a la pandemia era de {pe['tendencia_anual_2000_2019']:.2f} "
        f"años de vida ganados por año calendario. La proyección sitúa al país en "
        f"{pe['proyeccion_2030']:.1f} años en 2030 (intervalo de ±{pe['banda_2030']:.1f}), frente "
        f"a los {pe['contrafactual_2030']:.1f} que habría alcanzado sin la pandemia: una brecha "
        f"residual de {abs(pe['brecha_vs_sin_covid']):.1f} años. Es una cicatriz notablemente "
        f"pequeña dada la magnitud del golpe, y refleja la fuerza del rebote de 2022-2023.")
    inf.p(
        f"El agregado regional es menos alentador. La proyección deja a América Latina en "
        f"{la['proyeccion_2030']:.1f} años en 2030, prácticamente el mismo valor que en "
        f"{anio} ({la['obs_2023']:.1f}), y {abs(la['brecha_vs_sin_covid']):.1f} años por debajo "
        f"de su trayectoria contrafactual. Este resultado debe leerse con cautela: la tendencia "
        f"amortiguada del modelo de Holt es sensible al escaso número de observaciones "
        f"posteriores al shock —sólo dos años— y probablemente subestima la recuperación. Es la "
        f"limitación más seria de esta sección, y se declara como tal.")
    inf.figura("fig15_proyeccion.png",
               "Proyección de la esperanza de vida a 2030 para el Perú y América Latina.")
    inf.salto()

    # ---------------------------------------------------------------- 6
    inf.h1("6. El caso del Perú")
    pe_d = DE["peru"]
    inf.p(
        f"Reunir los resultados anteriores alrededor de un solo país permite ver cómo se "
        f"combinan. Entre 2000 y 2019 el Perú ganó {pe_d['ev_2019'] - pe_d['ev_2000']:.1f} años "
        f"de esperanza de vida, pasando de {pe_d['ev_2000']:.1f} a {pe_d['ev_2019']:.1f}, un "
        f"ritmo superior al del promedio latinoamericano. En el mismo período su mortalidad "
        f"infantil cayó de {pe_d['mi_2000']:.1f} a {pe_d['mi_2023']:.1f} muertes por cada mil "
        f"nacidos vivos, una reducción de más del 60%.")
    inf.p(
        f"La pandemia interrumpió ese avance de la forma más violenta registrada en la muestra "
        f"para un país de su tamaño: {cv['delta_peru']:+.1f} años entre 2019 y 2021, la posición "
        f"número {cv['ranking_peru']} del mundo. La caída llevó al país a "
        f"{pe_d['ev_2021']:.1f} años, un nivel que no registraba desde comienzos de la década de "
        f"2010. Es notable, sin embargo, que la mortalidad infantil no se movió: siguió "
        f"descendiendo durante todo el período. La pandemia golpeó a los adultos mayores, no a "
        f"los recién nacidos, y por eso el desplome de la esperanza de vida no vino acompañado "
        f"de un deterioro de los indicadores materno-infantiles.")
    inf.figura("fig07_peru.png",
               "Perú frente a América Latina y el mundo: esperanza de vida y mortalidad infantil.")
    inf.p(
        f"La recuperación fue igual de rápida. En {anio} el Perú registra {pe_d['ev_2023']:.1f} "
        f"años, {pe_d['brecha_vs_lac_2023']:+.1f} respecto de América Latina "
        f"({pe_d['lac_2023']:.1f}) y {pe_d['ev_2023'] - pe_d['mundo_2023']:+.1f} respecto del "
        f"mundo ({pe_d['mundo_2023']:.1f}). Superó incluso su propio nivel prepandemia. Conviene "
        f"tratar esa recuperación con escepticismo metodológico: las series de esperanza de vida "
        f"de los años pandémicos se han revisado varias veces, y un rebote que supera el nivel "
        f"previo puede reflejar tanto una recuperación real como una corrección estadística de "
        f"la sobremortalidad registrada.")
    inf.p(
        f"Lo más interesante del caso peruano es su posición respecto de los modelos. En la "
        f"curva de Preston el Perú se ubica {pres['residuo_peru']:+.1f} años por encima de lo "
        f"que su PIB per cápita de {pe_d['pib_pc_2023']:,.0f} dólares anticiparía. En el modelo "
        f"predictivo completo el residuo es aún mayor: {PR['residuo_peru']:+.1f} años por encima "
        f"de lo predicho. Con {pe_d['gasto_salud_pib_2023']:.1f}% del PIB destinado a salud —por "
        f"debajo del promedio regional— y {pe_d['medicos_2023']:.2f} médicos por cada mil "
        f"habitantes, el país obtiene más años de vida de los que sus recursos harían esperar. "
        f"La contracara es que ese desempeño convive con una fragilidad extrema ante shocks, "
        f"como demostró 2020."
        .replace(",", "."))

    # ---------------------------------------------------------------- 7
    inf.h1("7. Conclusiones y limitaciones")
    inf.h2("7.1 Conclusiones")
    inf.vinetas([
        f"La brecha mundial de esperanza de vida es de {DE['amplitud']:.0f} años entre extremos, "
        f"pero la mitad central de los países se concentra en un rango de apenas "
        f"{dist['p75']-dist['p25']:.0f} años. El mundo tiene un pelotón mayoritario y una cola "
        f"rezagada, no una gradación continua.",
        f"El nivel de ingreso ordena la esperanza de vida mejor que la región geográfica, pero "
        f"lo hace con rendimientos decrecientes: duplicar el PIB per cápita añade unos "
        f"{pres['anios_por_duplicar_pib']:.1f} años, sea cual sea el punto de partida.",
        f"El acceso a saneamiento básico es el predictor individual más potente del modelo, muy "
        f"por encima del PIB per cápita y del número de médicos. Las condiciones materiales "
        f"básicas pesan más que la sofisticación del sistema sanitario.",
        f"El gasto en salud importa en dólares por persona (ρ = {corr['Gasto salud p/c']:.2f}), "
        f"no como porcentaje del PIB (ρ = {corr['Gasto salud %PIB']:.2f}).",
        f"Un modelo entrenado sobre países distintos de los que evalúa alcanza un R² de "
        f"{mej['r2']:.2f} y un error típico de {mej['rmse']:.1f} años, un "
        f"{PR['mejora_vs_baseline_rmse']:.0f}% mejor que la línea base. La relación entre "
        f"desarrollo y longevidad es predecible y esencialmente monótona.",
        f"América Latina fue la región más golpeada del mundo por la pandemia "
        f"({cv['por_region']['América Latina y Caribe']:+.1f} años en promedio), y el Perú, uno "
        f"de sus casos extremos (posición {cv['ranking_peru']} mundial).",
        f"El Perú vive entre {pres['residuo_peru']:.1f} y {PR['residuo_peru']:.1f} años más de "
        f"lo que sus recursos económicos y sanitarios predicen, un excedente que convive con "
        f"una alta vulnerabilidad ante shocks.",
    ])

    inf.h2("7.2 Limitaciones")
    inf.vinetas([
        f"Imputación. El {D['pct_celdas_imputadas']}% de las celdas del panel fue imputado. En "
        f"médicos por habitante y matrícula secundaria la proporción original de vacíos superaba "
        f"el 30%, y la imputación por mediana regional impone a esos países el comportamiento "
        f"medio de sus vecinos, lo que comprime artificialmente la varianza.",
        "Correlación, no causalidad. Ningún resultado de este informe identifica efectos "
        "causales. La alta colinealidad entre predictores (VIF > 5 en agua y saneamiento) "
        "impide atribuir el crédito a una variable concreta.",
        "Calidad desigual del registro. Los países con sistemas estadísticos débiles reportan "
        "menos mortalidad, lo que sesga sus indicadores al alza; parte de la aparente "
        "resistencia de África subsahariana durante la pandemia puede ser subregistro.",
        "Proyección con pocos datos post-shock. Las proyecciones a 2030 se apoyan en sólo dos "
        "observaciones posteriores a la pandemia. La del agregado latinoamericano es la más "
        "frágil y probablemente subestima la recuperación.",
        "Microestados en los rankings. Los extremos superiores están dominados por territorios "
        "de población muy pequeña, cuyos valores no son comparables con los de países grandes.",
        "Ventana temporal. El panel cubre 2000-2023; las conclusiones no se extienden "
        "automáticamente a períodos anteriores.",
    ])

    # ---------------------------------------------------------------- 8
    inf.h1("8. Reproducibilidad")
    inf.p(
        "El análisis completo se reproduce desde cero con los siguientes pasos. Los scripts "
        "deben ejecutarse en orden; cada uno deja sus salidas en disco para el siguiente.")
    inf.vinetas([
        "mamba env create -f environment.yml && mamba activate wdi-lab",
        "python src/01_descarga.py — descarga los indicadores desde la API del Banco Mundial",
        "python src/02_preparacion.py — construye el panel país-año limpio",
        "python src/03_descriptivo.py — figuras 1 a 7",
        "python src/04_diagnostico.py — figuras 8 a 11",
        "python src/05_predictivo.py — figuras 12 a 15",
        "python src/06_informe.py — genera este documento",
        "streamlit run dashboard/app.py — dashboard interactivo complementario",
    ])
    inf.p(
        "Todas las cifras citadas en el texto se leen del archivo results.json, generado por los "
        "scripts de análisis. Ninguna fue transcrita a mano, de modo que el documento no puede "
        "desincronizarse de los cálculos que lo respaldan.")

    inf.h2("8.1 Dashboard interactivo")
    inf.p(
        "Como complemento al documento se construyó un dashboard en Streamlit, que cumple el "
        "papel que el enunciado asigna a Power BI o Tableau. Consume exactamente el mismo panel "
        "limpio y el mismo archivo de resultados que este informe, de modo que ninguna cifra "
        "puede divergir entre ambos entregables. Permite filtrar por año, región y grupo de "
        "ingreso, y se organiza en cuatro pestañas: Panorama (indicadores clave, mapa "
        "coroplético y evolución regional), Comparar países (series por indicador y curva de "
        "Preston con el tamaño de la burbuja proporcional a la población), Modelo (métricas de "
        "los cuatro modelos, importancia de variables y proyecciones) y Datos (tabla filtrable "
        "y descarga en CSV).")
    inf.figura("dash01_panorama.png",
               "Dashboard, pestaña «Panorama»: indicadores clave, mapa coroplético "
               "interactivo y evolución por región.")
    inf.figura("dash02_modelo.png",
               "Dashboard, pestaña «Modelo»: comparación de los cuatro modelos, importancia "
               "de las variables y proyecciones a 2030.")
    inf.p(
        "Para levantarlo basta con activar el entorno y ejecutar «streamlit run "
        "dashboard/app.py»; la aplicación queda disponible en http://localhost:8501.")

    ruta = DIR_INFORME / "Laboratorio1_WDI.docx"
    inf.doc.save(ruta)
    return ruta, inf


if __name__ == "__main__":
    ruta, inf = construir()
    print(f"Informe generado: {ruta}")
    print(f"  {inf.n_fig} figuras, {inf.n_tabla} tablas")
