# 02 · Reparto de los datos entre clientes

**Estado:** confirmada (tests públicos y propios en verde).
**Afecta a:** `fl/data.py` (`iid`, `dirichlet`, `dirichlet_shards`), los histogramas `partition.png` y
la pregunta 1 de la memoria.

## 1. Contexto

El enunciado pide tres repartos de Fashion-MNIST (60.000 imágenes, 6.000 por clase) entre 10
clientes: IID, Dirichlet(α) y k clases por cliente con proporciones Dirichlet entre ellas. Los tres
deben ser particiones: cada ejemplo va a exactamente un cliente, sin huecos ni solapes. Además, la
misma semilla debe dar el mismo reparto.

## 2. IID

Se barajan los índices 0..N−1 con un generador propio (`np.random.default_rng(seed)`) y se cortan en
`n_clients` trozos consecutivos. Si N no es divisible entre los clientes, los `N mod n_clients`
primeros reciben un ejemplo más, así que los tamaños difieren como mucho en 1.

Con Fashion-MNIST y 10 clientes, cada cliente recibe exactamente 6.000 ejemplos y la clase más
frecuente de cada cliente ronda el 11 % (lo esperado es el 10 %).

## 3. Dirichlet(α)

Sigue a Hsu et al. (2019), como indica el docstring. Para cada clase:

1. se barajan sus índices,
2. se muestrean unas proporciones p ~ Dir(α, …, α), una por cliente,
3. se reparten con `split_by_proportions`.

Si algún cliente termina con menos de `min_size` (10) ejemplos, **se repite el muestreo entero**, no
solo el de ese cliente. El generador no se reinicia entre intentos, así que cada intento es distinto
y el resultado sigue siendo determinista para una semilla dada.

Con nuestra configuración los reintentos casi nunca hacen falta (medido sobre 20 semillas):

| α | Intentos medios | Máximo |
|---|---:|---:|
| 0,1 | 1,00 | 1 |
| 0,05 | 1,15 | 2 |
| 0,01 | 9,05 | 24 |

El bucle no tiene un límite de intentos. Con α ≥ 0,05 no es un problema, pero con α muy pequeño
podría tardar mucho; no lo usamos en ningún experimento.

## 4. k clases por cliente (`dirichlet_shards`)

### 4.1. Por qué se reescribió

La primera versión pasaba el test público, pero al medirla con 50 semillas aparecieron tres
problemas:

| Problema | Medida |
|---|---|
| Clientes vacíos o casi vacíos | Con k=2 y α=0,1 (el caso del experimento 1), 28 de 50 semillas dejaban algún cliente con menos de 10 ejemplos, y en algunas con 0 |
| Clases repetidas | Las clases sobrantes se sorteaban con reemplazo y se troceaban en grupos de k: un cliente podía recibir dos veces la misma clase y quedarse con menos de k distintas |
| No seguía el docstring | Muestreaba un Dirichlet por clase entre sus dueños, cuando el docstring pide que cada cliente tenga pesos Dir(α) sobre sus k clases. Además, no barajaba los índices de cada clase |

Un cliente vacío hace fallar el entrenamiento local (no hay mini-lotes) y FedAvg (peso 0/0). Como la
corrección usa tests adicionales, también era probable que alguno comprobase las k clases distintas.

### 4.2. Cómo funciona ahora

1. **Asignación de clases.** Se baraja la lista de clases y se recorre en círculo: el cliente i recibe
   las posiciones i·k, i·k+1, …, i·k+k−1 (módulo 10). Esto garantiza que:
   - las k clases de cada cliente son distintas (porque k ≤ 10),
   - todas las clases tienen dueño (porque n_clients·k ≥ 10),
   - cada clase tiene el mismo número de dueños, ±1 (con 10 clientes y k=2, exactamente 2).
2. **Pesos de cada cliente.** Cada cliente muestrea w_i ~ Dir(α, …, α) sobre sus k clases.
3. **Reparto de cada clase.** Los ejemplos de cada clase (barajados) se reparten entre sus dueños en
   proporción a lo que pesa esa clase para cada uno.

Si la configuración es imposible (n_clients·k < 10, o k fuera de 1..10) se lanza un `ValueError`
que explica por qué. Hay una guarda por si un muestreo de Dirichlet con α diminuto devuelve 0/0.

### 4.3. Por qué nunca queda un cliente vacío

Los pesos de cada cliente suman 1, así que su clase con más peso tiene al menos 1/k. Con 10 clientes y
k=2, cada clase tiene 2 dueños, y el peso del otro dueño en esa clase es como mucho 1. Por tanto, el
cliente se lleva al menos (1/2)/(1/2 + 1) = 1/3 de esa clase, es decir, **al menos 2.000 ejemplos**.
Es una cota garantizada; en la práctica, el mínimo observado en 50 semillas fue de 3.000.

## 5. Cómo quedan los repartos

Medido con las semillas 0 a 4. Los tamaños solo dependen de los recuentos por clase y de la semilla,
así que coinciden con los de Fashion-MNIST real.

| Reparto | Tamaño mín. | Tamaño máx. | Peso medio de la clase dominante | Clases con > 5 % de los datos (media) |
|---|---:|---:|---:|---:|
| IID | 6.000 | 6.000 | 11 % | 10,0 |
| Dirichlet α=1 | 2.959 | 9.700 | 28 % | 6,3 |
| Dirichlet α=0,1 | 387 | 16.169 | 58 % | 3,1 |
| k=2, α=0,1 | 3.000 | 9.000 | 90 % | 1,4 |

Con k=2 y α=0,1, el 58 % de los clientes tiene más del 95 % de sus datos en una sola clase. El peso
Dirichlet de cada cliente suele concentrarse en una de sus dos clases, así que muchos clientes son en
la práctica de una sola clase. **Es lo que pide el enunciado**, y coincide con el histograma de k=2
del seminario: debe presentarse como el escenario más heterogéneo, no como un error del reparto.

## 6. Consecuencia para los experimentos

La semilla de la configuración controla a la vez el reparto, la inicialización del modelo, el
muestreo de clientes y el orden de los mini-lotes. Al repetir con otra semilla cambia también el
reparto, así que la desviación típica que damos mide la variabilidad total del método, incluida la
del reparto, y no solo la del entrenamiento. Es lo que queremos, pero hay que decirlo en la memoria.

## 7. Verificación

- Tests públicos de `tests/public/test_data.py`.
- Tests propios en `tests/extra/test_extra_data.py`: tamaños no divisibles en IID, reproducibilidad,
  `min_size` en Dirichlet con varios α y semillas, α grande ≈ IID, shards sin clientes vacíos para
  k ∈ {1, 2, 3, 5, 10}, exactamente k clases distintas, k=1 y configuraciones imposibles.

## 8. Fuentes

1. Hsu, T.-M. H., Qi, H. y Brown, M. (2019). *Measuring the Effects of Non-Identical Data
   Distribution for Federated Visual Classification*. arXiv:1909.06335.
   https://arxiv.org/abs/1909.06335
2. Seminario 3 de la asignatura (diapositiva 5: repartos IID, Dirichlet α=1, α=0,1 y k=2).
