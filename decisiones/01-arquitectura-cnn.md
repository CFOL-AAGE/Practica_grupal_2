# 01 · Arquitectura de la CNN

**Estado:** propuesta, pendiente de la comparación de la sección 6.
**Afecta a:** `fl/models.py` (clase `CNN`), pregunta 6 de la memoria y, de forma indirecta, la
pregunta 5 (bytes transmitidos).

## 1. Contexto

El enunciado pide una CNN de diseño libre con un máximo de 100.000 parámetros y que se justifique el
diseño. La CNN se usa para repetir el experimento 1 (heterogeneidad: IID, Dirichlet α=1, α=0,1, k=2
clases por cliente, centralizado y solo local) y compararla con el MLP 784→200→200→10.

Las condiciones de entrenamiento son las de `configs/default.yaml` y no se tocan para la CNN, porque
la pregunta 6 compara arquitecturas y cambiar el optimizador mezclaría dos efectos:

- 10 clientes, fracción C=0,5, 30 rondas, E=1 época local.
- SGD **sin momentum**, lr=0,01, mini-lote de 32.
- Agregación FedAvg; evaluación del modelo global sobre el test completo de Fashion-MNIST.
- Repartos fuertemente no-IID (α=0,1 y k=2), donde cada cliente ve una o dos clases dominantes.

Por tanto, la pregunta no es "qué CNN da más accuracy en Fashion-MNIST", sino **qué CNN de ≤100.000
parámetros funciona mejor promediada con FedAvg sobre datos no-IID, con SGD plano y 30 rondas**.

## 2. Decisión

La CNN del artículo original de FedAvg (McMahan et al., 2017) **sin su capa oculta densa**:

```python
nn.Conv2d(1, 32, kernel_size=5, padding=2),   # 832 parámetros      -> (32, 28, 28)
nn.ReLU(),
nn.MaxPool2d(2),                              #                      -> (32, 14, 14)
nn.Conv2d(32, 64, kernel_size=5, padding=2),  # 51.264 parámetros   -> (64, 14, 14)
nn.ReLU(),
nn.MaxPool2d(2),                              #                      -> (64, 7, 7)
nn.Flatten(),                                 #                      -> 3.136
nn.Linear(64 * 7 * 7, 10),                    # 31.370 parámetros   -> logits
```

- **83.466 parámetros** (83 % del límite), 333,9 KB por envío en float32.
- **10,7 M multiplicaciones-acumulaciones (MAC) por imagen** en la pasada hacia delante.
- Sin BatchNorm, sin dropout, sin weight decay y sin global average pooling.
- Mismos hiperparámetros de entrenamiento que el MLP.

## 3. Justificación, decisión a decisión

### 3.1. Partir de la CNN de referencia del aprendizaje federado

McMahan et al. (2017), el artículo que introduce FedAvg, usa para MNIST una CNN con dos convoluciones
5×5 (32 y 64 canales, cada una seguida de max pooling 2×2), una capa densa de 512 unidades con ReLU y
una salida softmax, con 1.663.370 parámetros [1, 2]. Es la arquitectura de referencia en la
literatura de FL sobre MNIST y Fashion-MNIST.

La capa densa de 512 unidades tiene 3.136 × 512 + 512 = 1.606.144 parámetros, el **96,6 %** del
total. Quitándola, las convoluciones quedan exactamente como en el artículo y el modelo cabe en el
límite. No proponemos una arquitectura nueva: adaptamos la de referencia eliminando precisamente la
parte que no cabe, que además es la menos eficiente en parámetros.

`padding=2` reproduce el *same padding* del original (28→14→7). Es también lo que cuadra el recuento
de 1.663.370 parámetros, que exige una entrada de 7×7×64 = 3.136 a la capa densa.

### 3.2. Sin BatchNorm

Es la decisión con más evidencia a favor.

- Hsieh et al. (ICML 2020) muestran que el problema de los datos no-IID es especialmente grave en
  modelos con batch normalization, y que group normalization recupera gran parte de esa pérdida [3].
- El mecanismo: BN normaliza con medias y varianzas de los mini-lotes locales, que con datos no-IID
  no se parecen a las del conjunto global con el que se evalúa. Ese desajuste entre las estadísticas
  de entrenamiento y las de test degrada FedAvg [4].
- En un estudio sobre CIFAR-10 con 10 clientes de 2 clases cada uno, añadir BN baja la accuracy
  no-IID de 74,11 % a 50,46 %, mientras que en IID la sube a 83,54 % [5]. Es decir, BN ayuda
  justo en el caso que no es el nuestro.
- Hay además un motivo práctico: BN añade al `state_dict` un tensor entero (`num_batches_tracked`) y
  estadísticas que no son parámetros entrenables. Eso complica FedAvg (redondear enteros) y el codec
  int8, y no aporta nada a una red de dos capas convolucionales, que se entrena bien sin normalizar.

FedBN (Li et al., 2021) propone mantener BN local en cada cliente [6], pero está pensado para
diferencias en las características de entrada (no en las etiquetas) y deja un modelo distinto por
cliente. Aquí se evalúa un único modelo global, así que no aplica.

### 3.3. Poca profundidad y mucha anchura

- En el mismo estudio sobre CIFAR-10, las redes de 3, 6 y 9 capas convolucionales obtienen en no-IID
  74,11 %, 73,67 % y 68,98 %, mientras que en IID mejoran con la profundidad. En cambio, ensanchar
  las capas mejora el resultado no-IID [5].
- Un trabajo de ICML 2025 lo explica en teoría: el efecto de la heterogeneidad de los datos sobre la
  convergencia de FedAvg disminuye al aumentar la anchura de la red y desaparece en el límite de
  anchura infinita, donde FedAvg se comporta como el entrenamiento centralizado [7].

Con un presupuesto fijo de parámetros, eso significa gastarlos en canales (32 y 64) y no en capas.
Además, con SGD sin momentum y lr=0,01 una red profunda sin normalización aprende despacio, y solo
tenemos 30 rondas. Dos capas convolucionales son el punto de equilibrio.

### 3.4. Flatten + capa lineal, no global average pooling

Es la decisión contraintuitiva y la que más fácilmente se puede cuestionar.

Global average pooling (GAP), propuesto en *Network in Network*, sustituye las capas densas por la
media espacial de cada mapa de características. No tiene parámetros, actúa como regularizador y
reduce mucho el tamaño del modelo [8]. Con el límite de 100.000 parámetros parecería la opción
natural.

Sin embargo, en FL no-IID da el peor resultado: en el estudio sobre CIFAR-10, la red con GAP obtiene
62,82 % frente a 74,35 % de la equivalente con max pooling y aplanado, aunque en entrenamiento
centralizado GAP gana (85,05 % frente a 82,74 %) [5]. Los autores lo atribuyen a que GAP deja una
entrada mucho más pequeña a la capa de clasificación. Mantenemos por tanto el aplanado (3.136
características) y una capa lineal.

### 3.5. Sin dropout ni weight decay

- **Weight decay:** en el estudio sobre CIFAR-10, la accuracy no-IID se mantiene estable hasta un
  factor de 0,0001 pero cae a 54,11 % con 0,0005, mientras que en IID apenas cambia [5]. No hay nada
  que ganar en nuestro caso y sí un hiperparámetro peligroso.
- **Dropout:** en ese estudio da una mejora pequeña en no-IID (+1,69 puntos, frente a +2,85 en IID),
  pero aumenta la divergencia de la última capa entre clientes [5]. Con 30 rondas y E=1 el modelo
  apenas llega a sobreajustar, así que dropout solo añadiría ruido y otro hiperparámetro.
- El MLP no lleva ninguna de las dos. Dejarlas fuera mantiene limpia la comparación de la pregunta 6:
  la única diferencia es la arquitectura.

### 3.6. Mismos hiperparámetros de entrenamiento que el MLP

La pregunta 6 pide repetir el experimento 1 con la CNN. Si cambiáramos lr, optimizador o tamaño de
lote, no podríamos atribuir las diferencias a la arquitectura. Si la CNN aprende más despacio con
SGD plano, eso es un resultado que se discute, no un motivo para cambiar la configuración.

### 3.7. Coherencia con el resto de la práctica

- **Comunicación (pregunta 5):** A envía 333,9 KB por modelo frente a 796,8 KB del MLP, 2,4 veces
  menos. Con la configuración por defecto, 30 rondas suman unos 100,2 MB (83.466 × 4 B × 2 sentidos
  × 5 clientes × 30) frente a 239,1 MB del MLP. Si la CNN iguala o supera al MLP en accuracy, lo hace
  con menos de la mitad de tráfico.
- **Cuantización int8:** las cinco matrices de pesos y sesgos son tensores float32 normales, sin
  buffers enteros, así que el codec int8 se aplica a todos sin casos especiales.
- **Accuracy esperable:** en el benchmark oficial de Fashion-MNIST, redes convolucionales de dos
  capas con menos de 100.000 parámetros rondan el 92 % en centralizado [9]. Es una referencia, no una
  predicción: aquellas se entrenan más épocas y con otros optimizadores.

## 4. Alternativas consideradas

Recuentos calculados exactamente (MAC por imagen en la pasada hacia delante; KB por envío en
float32):

| Candidata | Parámetros | KB/envío | MAC/imagen | Motivo |
|---|---:|---:|---:|---|
| **A · 5×5, 32-64, lineal (elegida)** | 83.466 | 333,9 | 10,7 M | CNN de FedAvg sin la capa densa; ancha y poco profunda |
| B · 3×3, 32-64, lineal | 50.186 | 200,7 | 3,9 M | Mismo diseño con menor campo receptivo; 2,8× más barata. Candidata seria (sección 6) |
| A + GroupNorm | 83.658 | 334,6 | ≈10,7 M | La normalización segura en FL [3]; probablemente innecesaria con dos capas (sección 6) |
| C · LeNet-5 (6-16, densas 120-84) | 61.706 | 246,8 | 0,4 M | Demasiado estrecha (6 y 16 canales), contra la evidencia de 3.3 |
| D · 3×3, 32-64-128, GAP | 93.962 | 375,8 | 7,5 M | Más profunda y con GAP: las dos cosas que peor funcionan en no-IID (3.3 y 3.4) |
| E · 5×5, 16-32, densa de 32, lineal | 63.786 | 255,1 | 2,9 M | Sacrifica anchura convolucional para meter una capa oculta |
| CNN de McMahan et al. completa | 1.663.370 | 6.653,5 | 12,3 M | 16,6 veces el límite |
| MLP (referencia) | 199.210 | 796,8 | 0,2 M | Fijado por el enunciado |

## 5. Riesgos y límites de esta justificación

1. **La evidencia no es de nuestra configuración.** Casi todos los números de las secciones 3.2 a
   3.5 vienen de CIFAR-10 con SGD con momentum [5], y sus autores los presentan como tendencias, no
   como reglas universales. Además, [5] es un envío a ICLR 2020 publicado en OpenReview, no un
   artículo de actas; lo usamos porque es el único que mide todos estos factores con el mismo
   montaje, y sus conclusiones sobre BN coinciden con las de Hsieh et al. [3], que sí está publicado.
   Justifican el diseño, pero no demuestran que A supere a B en Fashion-MNIST con SGD plano.
2. **Seleccionar sobre el test.** `run.py` evalúa siempre en el conjunto de test. Si se elige la
   arquitectura mirando esa accuracy, el número final queda optimista. Por eso A se fija *a priori*
   por la literatura y la sección 6 se presenta como comprobación, no como búsqueda.
3. **Coste de cómputo.** A cuesta 2,8 veces más que B por imagen. El README da unos 75 s para "la CNN
   de referencia" en un sobremesa, pero no dice cuál es. Antes de lanzar los barridos hay que medir
   el tiempo de una ronda en nuestros portátiles.
4. **Inicialización.** Usamos la de PyTorch por defecto, igual que el MLP, para no introducir otra
   diferencia. Con dos capas no debería ser un problema.

## 6. Comprobación propuesta

Comparar A, B y A + GroupNorm en IID y en Dirichlet α=0,1, con las semillas 0, 1 y 2: **18
ejecuciones**. Las de la candidata que quede se reutilizan en la pregunta 6.

- Las variantes B y A + GroupNorm se prueban en una rama aparte. La clase `CNN` final conserva su
  firma `CNN(n_classes=10)`, que usan los tests y `build_model`.
- Métricas: accuracy final (media de las 5 últimas rondas) y estabilidad (desviación típica de
  `test_acc` en las 10 últimas), con media ± desviación entre semillas, y tiempo por ronda.
- Criterio fijado de antemano: se mantiene A salvo que B o A + GroupNorm la superen en α=0,1 por más
  de una desviación típica. Si quedan dentro del ruido, se mantiene A por la justificación de la
  sección 3 (o B, si el tiempo de cómputo de A resulta inasumible, y se dice así en la memoria).

| Variante | IID: acc. final | IID: estabilidad | α=0,1: acc. final | α=0,1: estabilidad | s/ronda |
|---|---|---|---|---|---|
| A | | | | | |
| B | | | | | |
| A + GroupNorm | | | | | |

## 7. Texto para la memoria

> Como CNN usamos la red convolucional del artículo original de FedAvg (McMahan et al., 2017), dos
> convoluciones 5×5 de 32 y 64 canales con ReLU y max pooling, sin su capa densa de 512 unidades,
> que concentra el 96,6 % de sus 1,66 M de parámetros. El resultado tiene 83.466 parámetros. No
> usamos batch normalization porque sus estadísticas por mini-lote dependen de la distribución local
> de cada cliente y degradan FedAvg con datos no-IID (Hsieh et al., 2020). Preferimos una red ancha y
> poco profunda, y una capa lineal sobre el mapa aplanado en lugar de global average pooling, porque
> con datos no-IID la profundidad y GAP empeoran el resultado mientras que la anchura lo mejora
> [5, 7]. Tampoco usamos dropout ni weight decay, y la entrenamos con los mismos hiperparámetros que
> el MLP para que la comparación solo dependa de la arquitectura. Con ello envía 2,4 veces menos
> bytes por ronda que el MLP.

## 8. Fuentes

1. McMahan, B., Moore, E., Ramage, D., Hampson, S. y Agüera y Arcas, B. (2017). *Communication-Efficient
   Learning of Deep Networks from Decentralized Data*. AISTATS 2017. https://arxiv.org/abs/1602.05629
2. Flower (2023). *FL Starter Pack: FedAvg on MNIST using a CNN* (reproducción de la CNN de [1] y su
   recuento de 1.663.370 parámetros). https://flower.ai/blog/2023-01-12-fl-starter-pack-fedavg-mnist-cnn
3. Hsieh, K., Phanishayee, A., Mutlu, O. y Gibbons, P. (2020). *The Non-IID Data Quagmire of
   Decentralized Machine Learning*. ICML 2020, PMLR 119:4387-4398.
   https://proceedings.mlr.press/v119/hsieh20a.html
4. Zhong, J., Chen, H.-Y. y Chao, W.-L. *Making Batch Normalization Great in Federated Deep
   Learning*. arXiv:2303.06530. https://arxiv.org/abs/2303.06530
5. Anónimo (2019). *On Federated Learning of Deep Networks from Non-IID Data: Parameter Divergence and
   the Effects of Hyperparametric Methods*. Envío a ICLR 2020, OpenReview.
   https://openreview.net/pdf/5e77d730063a01ed6a3cd0b0b51e689011736107.pdf
6. Li, X., Jiang, M., Zhang, X., Kamp, M. y Dou, Q. (2021). *FedBN: Federated Learning on Non-IID
   Features via Local Batch Normalization*. ICLR 2021. https://arxiv.org/abs/2102.07623
7. *Widening the Network Mitigates the Impact of Data Heterogeneity on FedAvg*. ICML 2025.
   https://arxiv.org/abs/2508.12576
8. Lin, M., Chen, Q. y Yan, S. (2014). *Network In Network*. ICLR 2014.
   https://arxiv.org/abs/1312.4400
9. Zalando Research. *Fashion-MNIST*, tabla de benchmarks del README.
   https://github.com/zalandoresearch/fashion-mnist
10. Wu, Y. y He, K. (2018). *Group Normalization*. ECCV 2018. https://arxiv.org/abs/1803.08494
