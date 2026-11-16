# 03 · Ronda federada, FedAvg y referencia "solo local"

**Estado:** confirmada para `sample_clients` y FedAvg (tests en verde). `Server.run_round` y
`LocalOnly.run_round` están validados con una implementación provisional de `local_train`; falta
repetir los tests con la definitiva.
**Afecta a:** `fl/server.py` (`sample_clients`, `Server.run_round`, `LocalOnly.run_round`) y
`fl/aggregators/fedavg.py`.

## 1. Selección de clientes (`sample_clients`)

- Se eligen `max(1, round(C · K))` clientes sin reemplazo, como dice el docstring. Con K=10, C=0,3,
  0,5 y 1,0 dan 3, 5 y 10 clientes.
- Se usa el generador del servidor (`self.rng`, sembrado con la semilla de la configuración) y no el
  global de NumPy, para que la secuencia de clientes sea reproducible e independiente del resto.
- Se devuelve una lista ordenada de `int` de Python, no de `np.int64`, que es lo que promete la firma
  (`list[int]`) y lo que se puede serializar a JSON.
- Con C > 1 se limita a K, para no pedir más clientes de los que hay.

## 2. Una ronda (`Server.run_round`)

Sigue paso a paso la ronda del seminario (diapositiva 3):

| Paso | Implementación |
|---|---|
| 1. El servidor elige una fracción C de clientes | `sample_clients(...)` |
| 2. Les envía el modelo global wₜ (bajada) | Cada cliente recibe `self.model` |
| 3. Cada uno entrena E épocas con sus datos | `client.local_train(self.model)` |
| 4. Devuelven sus pesos (subida) | `codec.encode` en el cliente y `codec.decode` en el servidor |
| 5. El servidor los combina | `aggregator.aggregate(updates)` y `load_state_dict` |
| 6. Se evalúa el modelo global | `evaluate` sobre el test completo (10.000 imágenes) |

**Todos los clientes parten del mismo wₜ.** Entrenan uno detrás de otro, pero el modelo global no se
toca hasta después del bucle y `local_train` trabaja sobre una copia. Por eso es equivalente a que
entrenen en paralelo, como en FL real. Si cada actualización se cargase en el modelo global al
recibirla, el cliente 2 partiría del modelo del cliente 1: eso sería entrenamiento secuencial, no
FedAvg.

**Copia de la referencia.** Antes del bucle se clona el estado global (`detach().clone()`). Es la
referencia que comparten cliente y servidor en el codec (int8 cuantiza la diferencia respecto a ella).
`load_state_dict` sobrescribe los tensores en su sitio; con la copia, la referencia no cambia aunque
se use después de agregar.

**Bytes.** Siguen la definición del seminario (diapositiva 8): nº de elementos × bytes por elemento,
con `state_dict_nbytes`.

- `bytes_down` = nº de clientes × tamaño del modelo global. La bajada nunca se comprime.
- `bytes_up` = suma del tamaño de lo que cada cliente envía **ya codificado** (el *payload*). Así, al
  activar int8 baja solo la subida, sin tocar esta función.
- No se cuenta la cabecera de serialización ni el protocolo de red, igual que en el seminario y en el
  test público.

Valores de control con la configuración por defecto (C=0,5, 30 rondas, sin codec):

| Modelo | Bytes por modelo | Bytes por ronda (5 clientes, bajada + subida) | Total en 30 rondas |
|---|---:|---:|---:|
| MLP (199.210 parámetros) | 796.840 | 7.968.400 | 239.052.000 (≈ 239 MB) |
| CNN (83.466 parámetros) | 333.864 | 3.338.640 | 100.159.200 (≈ 100 MB) |

**`train_loss`** es la media simple, no ponderada, de la pérdida que cada cliente midió *mientras*
entrenaba su copia local, como dice el docstring. No es la pérdida del modelo global sobre los datos de
entrenamiento. En no-IID puede ser baja aunque el modelo global vaya mal, porque cada cliente aprende
bien su propia distribución, y esa diferencia es en sí misma un argumento para la pregunta 1.

**Solo se devuelven las cinco métricas** del docstring (`test_acc`, `test_loss`, `train_loss`,
`bytes_up`, `bytes_down`). `RunLogger` escribe con un `csv.DictWriter` de columnas fijas, y una clave
de más haría fallar la ejecución.

## 3. Agregación (`FedAvg.aggregate`)

w_{t+1} = Σ_k (n_k / n) · w_k, donde n es el total de ejemplos **de los clientes de esa ronda**, como
en el seminario (diapositiva 4) y en McMahan et al. (2017).

- **Se acumula en float64** y al final se vuelve al dtype original. Sumar en float32 añade un error de
  redondeo pequeño pero innecesario.
- **Se conservan las claves, su orden, la forma y el dtype** de cada tensor. Los tensores enteros (como
  el `num_batches_tracked` de BatchNorm, aunque nuestros modelos no lo tienen) se redondean y vuelven
  a su tipo, como pide el docstring.
- **Se crean tensores nuevos**: no se modifica ningún `state_dict` de los clientes, y tocar el
  resultado tampoco los cambia.
- **Con 0 ejemplos en total se lanza un `ValueError`.** Con el reparto de la decisión 02 no debería
  ocurrir, pero si ocurre el error dice qué falla, en vez de propagar un NaN durante varias rondas.

## 4. Referencia "solo local" (`LocalOnly.run_round`)

- **Entrenan todos los clientes en cada ronda**, como dice el docstring. Sin servidor no hay muestreo,
  así que C no se usa.
- **Cada cliente conserva su modelo entre rondas.** `local_train` devuelve los pesos entrenados sin
  tocar el modelo que recibe, así que hay que cargarlos con `load_state_dict`. Sin esa línea los
  modelos locales nunca aprenderían. 30 rondas equivalen a 30 épocas de entrenamiento local por cliente.
- **`test_acc` es la media simple de los 10 modelos** evaluados sobre el test global, no un modelo
  único. Mide cuánto acierta, de media, un cliente que no colabora. Con α=0,1 cada modelo solo conoce
  una o dos clases, y por eso el seminario lo sitúa en torno a 0,35.
- Los bytes son 0, porque no hay comunicación.
- Es la ejecución más lenta: evalúa 10 modelos por ronda sobre las 10.000 imágenes de test.

## 5. Lo que estas funciones suponen de `local_train`

`run_round` y `LocalOnly` dan por hecho tres cosas que la implementación definitiva de
`Client.local_train` debe cumplir (las dice su docstring):

1. no modifica `global_model` (trabaja sobre `copy.deepcopy`);
2. devuelve un `state_dict` con las mismas claves que el modelo;
3. pone la pérdida media de entrenamiento en `metrics["train_loss"]`.

## 6. Verificación

- Tests públicos `test_sample_clients` y `test_fedavg`, en verde.
- Tests propios en `tests/extra/test_extra_aggregators.py` y `tests/extra/test_extra_server.py`:
  ponderación con tres clientes, un solo cliente, dtypes y enteros, inmutabilidad, error con 0
  ejemplos; `sample_clients` (tipo, orden, reproducibilidad, casos límite); `run_round` con C=1, solo
  las 5 columnas, paso por el codec con un codec de prueba a float16 (comprueba que `bytes_up` se
  mide sobre el *payload*), reproducibilidad; y `LocalOnly` (conserva el modelo entre rondas y no
  toca el global).
- `test_run_round`, `test_local_only` y los tests propios del servidor pasan con una implementación
  provisional de `local_train`. Hay que repetirlos con la definitiva.

## 7. Fuentes

1. McMahan, B., Moore, E., Ramage, D., Hampson, S. y Agüera y Arcas, B. (2017). *Communication-Efficient
   Learning of Deep Networks from Decentralized Data*. AISTATS 2017. https://arxiv.org/abs/1602.05629
2. Seminario 3 de la asignatura (diapositivas 3, 4, 7 y 8).
