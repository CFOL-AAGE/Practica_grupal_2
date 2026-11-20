# Reto de aprendizaje federado

El reto plantea un escenario de aprendizaje federado más exigente que el de la práctica: 100
clientes con datos muy heterogéneos, poca comunicación por ronda y una parte de los clientes que
actúa de forma maliciosa. El objetivo es obtener el modelo global con mayor accuracy sobre un
conjunto de test oculto.

La base es el código de la práctica: los mismos clientes, agregadores, codecs de comunicación y
configuración YAML. Cambian los datos y el escenario.

## 1. Instalación

El kit se descomprime en la raíz del repositorio de la práctica. Solo añade archivos nuevos; no
sobrescribe ninguno existente:

```
reto.py               simula el reto y prepara el envío (como run.py)
configs/reto.yaml     configuración del reto (como los ejemplos de configs/examples/)
fl/reto_models.py     modelos del reto (como build_model en fl/models.py)
data/reto/            datos
```

No requiere instalar nada más.

## 2. Probar

```bash
python reto.py                                        # una simulación completa (segundos)
python reto.py --set client.lr=0.1 client.local_epochs=5
python reto.py --seeds 0 1 2 --curve                  # varias semillas y accuracy por ronda
```

`reto.py` separa un 10 % de los datos como validación y reparte el resto entre los clientes. Al
final muestra la accuracy en validación. Con el FedAvg de la práctica sale en torno a un 70 %.

## 3. El escenario

| | |
|---|---|
| Datos | cada ejemplo es un vector de **384 números** (el *embedding* de una imagen, extraído con una red preentrenada sin etiquetas); **100 clases** |
| Clientes | **100**, con un reparto muy no-IID (pocas clases por cliente y en proporciones desiguales) |
| Rondas | **20**; en cada una se eligen **10** clientes al azar |
| Abandonos | en torno a uno de cada cinco elegidos no responde en esa ronda |
| Presupuesto | lo que sube un cliente en una ronda (la salida de `codec.encode`) no puede pasar de **160 KiB** |
| Atacantes | hasta un **20 %** de los clientes (ver abajo) |

El escenario lo fija `reto.py`. En `configs/reto.yaml` se ignoran `partition.*`, `server.rounds` y
`server.fraction`.

Sobre el presupuesto:
- el modelo lineal en float32 cabe justo;
- un MLP no cabe sin comprimir. `python reto.py --set model.name=mlp server.codec=int8` usa el codec
  de la práctica.

**Clientes maliciosos.** En la evaluación, una parte de los clientes es maliciosa durante toda la
simulación, y no se sabe cuáles son. Pueden:
- entrenar con etiquetas manipuladas;
- enviar actualizaciones manipuladas: invertidas, exageradas o con ruido.

Los ataques actúan sobre lo que se sube, esté comprimido o no. `reto.py` no los simula, así que
probar las defensas forma parte del reto.

## 4. Qué se puede cambiar

Lo mismo que en la práctica, con los mismos mecanismos:

- **Cliente** (`fl/clients/`, registrado en `build_clients`): la pérdida o el entrenamiento local.
  Cada cliente se crea una vez y vive toda la simulación, así que puede guardar estado entre rondas.
- **Agregador** (`fl/aggregators/`, registrado en `build_aggregator`). En el reto, el agregador
  tiene además en `self.global_state` el modelo global de la ronda, útil para calcular deltas, usar
  momento, etc.
- **Codec** (`fl/codecs/`, registrado en `build_codec`): qué viaja y cómo se comprime.
- **Modelo** (`fl/reto_models.py`). Se elige con `model.name`.
- **Hiperparámetros** (`configs/reto.yaml`). Las claves nuevas (p. ej. `server.trim`) se añaden
  también a `configs/default.yaml`, como en la práctica.

## 5. Enviar

```bash
python reto.py --pack       # comprueba las reglas y crea envio.zip
```

`envio.zip` se sube en la web del reto, junto con una descripción de 3-4 líneas de lo que hace.

- El zip lleva la carpeta `fl/` y los dos YAML de `configs/`.
- `fl/utils.py` y `fl/data.py` se sustituyen por los originales, así que cualquier cambio en ellos
  se pierde.
- Como mucho, **1 envío por hora y 6 al día** por grupo. Los que fallan no cuentan.
- La web muestra la puntuación de cada envío o, si falla, el error.

**Puntuación:** accuracy sobre el test oculto, en media de varias ejecuciones con repartos y
atacantes distintos. En la clasificación cuenta el mejor envío de cada grupo.

**Cierre: 11 de diciembre de 2026 a las 23:55.**

## 6. Reglas

1. **No se suben datos:** ni ejemplos, ni sus etiquetas, ni nada calculado a partir de un ejemplo
   individual. Los estadísticos agregados de un cliente (medias por clase, recuentos…) sí.
2. **Sin datos ni modelos externos.**
3. **Aprendizaje federado de verdad.** Cada cliente usa solo sus datos y el servidor solo lo que
   recibe. No se permite intentar averiguar nada del evaluador ni de otros clientes por otros caminos.
4. **El código no accede a archivos, red ni procesos.** `reto.py --pack` comprueba los imports
   permitidos (`torch`, `numpy`, `math`, `copy`, `collections`, `random`, `fl`…).
5. **Tiempo limitado.** Si `python reto.py` tarda más de un minuto por semilla en un portátil,
   probablemente se pase del tiempo de la evaluación.
6. **Revisión del código.** Antes de cerrar la clasificación se revisa el código de los primeros
   puestos, y un envío que incumpla las reglas queda descalificado.
