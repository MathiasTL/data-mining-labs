# Segunda evaluación: reconciliación de vistas y pruebas

Fecha: 13/11/2026. Complementa [linaje y reglas temporales](02-linaje-y-reglas-temporales.md).

## Resultado ejecutivo

Se ejecutaron **39 comprobaciones satisfactorias** y se registraron **tres no ejecutadas** por ausencia de matrices de entrenamiento. No se detectaron discrepancias en los controles ejecutados. Además, pasaron **159 pruebas** focalizadas del código existente. Esto no equivale a certificar ausencia de fuga ni a repetir F3 completo.

## Evidencia nueva

- [Controles detallados](controles-vistas.csv).
- [Resumen y huellas de catálogo y especificación](resumen-vistas.json).
- [Salida de pruebas](pruebas-segunda-evaluacion.txt).
- [Script reproducible](auditar_vistas.py).

## Checklist ejecutado

- [x] Comprobar unicidad de columnas activas por vista.
- [x] Relacionar características de origen con las 124 entradas del catálogo, incluyendo descartes explícitos.
- [x] Comprobar ausencia de columnas reservadas entre los predictores declarados.
- [x] Verificar controles declarados y parámetros JSON interpretables.
- [x] Leer las seis matrices locales OOT/test y comprobar columnas, dimensiones y unicidad/no nulos de `case_id`.
- [x] Verificar `WEEK_NUM >= 74` y etiqueta binaria no nula en las tres matrices OOT.
- [x] Comparar casos, orden, etiquetas y controles de las vistas OOT con la matriz maestra OOT.
- [x] Verificar ausencia de nulos en predictores lineales de OOT y test.
- [x] Ejecutar pruebas focalizadas de vistas, solicitudes/personas, bureau, productos as-of, contratos y profundidad cero.
- [ ] Verificar contenido de las tres matrices de entrenamiento ausentes.
- [ ] Recalcular parámetros de transformación contra entrenamiento y comprobar ajuste real por partición.
- [ ] Validar semántica y disponibilidad temporal de cada fuente.
- [ ] Aprobar diseño temporal de selección, calibración, umbrales y evaluación final.

## Reconciliación de descartes

La especificación contiene 411 filas: 348 predictores activos, 10 controles y 53 descartes. Los descartes tienen `view=none`; las vistas afectadas figuran en `parameters.views`.

Al considerar esa convención, ninguna de las 124 características queda sin representación en cada vista. Hay diez características base descartadas de ambas vistas y 43 salidas adicionales descartadas de la vista lineal. La regla de descarte constante está implementada y probada, pero no se recalcularon las constantes sobre entrenamiento ausente.

La primera ejecución del script de auditoría señaló diez orígenes ausentes porque no interpretaba `parameters.views`. Se corrigió el script de auditoría, no los datos ni el código productivo. La ejecución final no presenta esas discrepancias.

## Lectura del código de vistas

En `src/dm_credit/preparation/views.py`:

- Los controles se separan de predictores y se rechazan columnas no declaradas.
- La vista lineal transforma con un estado ajustado; utiliza imputación, escalado y codificación congelados.
- La vista de árboles conserva nulos y aplica vocabulario categórico congelado.
- Los descartes de constantes se deciden en el ajuste y se registran.

Las vistas publicadas fueron ajustadas con el train completo anterior a la semana 74. **No deben reutilizarse como si fueran preprocesamiento libre de contaminación para validaciones internas dentro de ese mismo train.** En cada partición interna deben reajustarse imputación, vocabulario, escalado y descarte de constantes con su subconjunto de entrenamiento. Para ello conviene partir de las características sin esas transformaciones ajustadas.

## Pruebas ejecutadas

Desde la raíz:

```sh
POLARS_MAX_THREADS=1 PYTHONPATH=src python3 -m pytest -q tests/test_preparation_views.py tests/test_preparation_applications_people.py tests/test_preparation_bureau.py tests/test_preparation_products_as_of.py tests/test_preparation_contracts.py tests/test_preparation_depth0.py
```

Resultado registrado: **159 passed in 6.83s**. La advertencia de termcap no impidió la ejecución. Las pruebas son del código existente, no una prueba de correspondencia de todos los artefactos históricos con ese código.

Para reproducir los controles de matrices:

```sh
POLARS_MAX_THREADS=1 python3 context/auditoria-variables-predictoras/auditar_vistas.py
```

Requiere Polars en el entorno. Escribe únicamente los controles y el resumen de esta carpeta; no regenera matrices ni entrena modelos.

## Prioridades siguientes

1. Documentar las dependencias reales de los conteos y precisar textos ambiguos del catálogo, sin declarar disponibilidad temporal no demostrada.
2. Buscar evidencia de origen sobre snapshots de personas, contactos, hijos de solicitudes y cobertura estática.
3. Clasificar lo que permanezca sin evidencia como supuesto o bloqueo explícito, no como dato verificado.
4. Fijar el protocolo temporal antes de reconstruir y ejecutar nuevos experimentos.
5. Verificar disponibilidad de fuentes y recursos para regenerar train en una ejecución controlada.

## Consulta al equipo

¿Existe documentación de Kaggle/Home Credit, un README original o una comunicación del equipo que respalde que `person_1/2`, los registros hijos de `applprev_2` y la presencia de fuentes estáticas representan información disponible al decidir cada solicitud?

Si no existe, no debe sustituirse por una afirmación: habrá que acordar y registrar un supuesto de trabajo o un análisis conservador de exclusión/ablación. Esta consulta no bloquea continuar la revisión técnica restante.
