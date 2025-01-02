# AI Training Prueba 1



## Instalación
```bash
git clone https://github.com/josepuga/oq-falcon-training
cd oq-falcon-training
source setup.sh
```

Esto descarga todas la dependencias, activa el Entorno Virtual de Python y deja todo listo.


## Uso
* Entrenar el modelo con `train.py`.
* (opcional) Realizar inferencias (deduciones) para probarlo `inferencia.py`.
  

**TODO:** Entrenar desde el último checkpoint.

## Ejemplo de sesión
He realizado una sesión con el dataset `mini_dataset.json` y modificando algunos parámetros para adaptarlos a mi hardware. Usando `time`para calcular el tiempo, sin CUDA me ha tardado casi 18 minutos.

```
$ time ./train.py 
Loading checkpoint shards: 100%|████████████████████████| 2/2 [00:01<00:00,  1.46it/s]
Map: 100%|██████████████████████████████████████| 1/1 [00:00<00:00, 301.10 examples/s]
Map: 100%|██████████████████████████████████████| 1/1 [00:00<00:00, 495.31 examples/s]
{'eval_loss': nan, 'eval_runtime': 107.7586, 'eval_samples_per_second': 0.009, 'eval_steps_per_second': 0.009, 'epoch': 1.0}                                                
{'train_runtime': 993.8091, 'train_samples_per_second': 0.001, 'train_steps_per_second': 0.001, 'train_loss': 6.0390625, 'epoch': 1.0}                                      
100%|██████████████████████████████████████████████████| 1/1 [16:33<00:00, 993.81s/it]

real    17m33.392s
user    18m26.171s
sys     2m17.199s
```

El fichero de entrenamiento ha sido `mini_dataset.json`.

```json
[
    {
        "prompt": "Hola",
        "response": "Hola, ¿cómo estás?"
    },
    {
        "prompt": "¿Qué día es hoy?",
        "response": "Hoy es lunes."
    },
    {
        "prompt": "¿Cómo te llamas?",
        "response": "Me llamo Falcon."
    },
    {
        "prompt": "¿Cuál es tu color favorito?",
        "response": "Mi color favorito es el azul."
    }
]
```


Ahora dos prueba de inferencia (deducciones) con el modelo entrenado:

```
(venv) $ time ./inferencia.py hola
Loading checkpoint shards: 100%|████████████████████████| 3/3 [00:01<00:00,  1.85it/s]
Setting `pad_token_id` to `eos_token_id`:11 for open-end generation.
Hola

real    1m17.717s
user    8m31.287s
sys     0m16.092s
(venv) $ time ./inferencia.py color favorito
Loading checkpoint shards: 100%|████████████████████████| 3/3 [00:01<00:00,  1.84it/s]
Setting `pad_token_id` to `eos_token_id`:11 for open-end generation.
Color favorito

real    1m15.437s
user    8m21.092s
sys     0m16.124s
```

El hecho de que el modelo se repita, es los pocos prompt de entrenamiento que tiene. Lógicamente, **habría que ampliar el dataset y comprobar los resultados finales**.

## Posible solución al largo tiempo de entrenamiento
Con el mismo código se puede usar `falcon 1B`, un modelo inferior, pero que es mucho más rápido, se puede trabajar en ese modelo mientras se crear el código y se depura, luego cuando todo esté en un estado estable, con una simple linea, se cambia el modelo y se empieza a trabajar con `falcon 7B`. Un ejemplo del mismo training que me ha tardado casi 8m, esto es un **42% menos de tiempo**.

Mismo training con el modelo 1B.
```
$ time ./train.py 
tokenizer_config.json: 100%|█████████████████████████| 445/445 [00:00<00:00, 6.67MB/s]
vocab.json: 100%|██████████████████████████████████| 798k/798k [00:00<00:00, 2.72MB/s]
merges.txt: 100%|██████████████████████████████████| 456k/456k [00:00<00:00, 4.78MB/s]
tokenizer.json: 100%|████████████████████████████| 2.11M/2.11M [00:00<00:00, 5.07MB/s]
special_tokens_map.json: 100%|█████████████████████| 99.0/99.0 [00:00<00:00, 1.88MB/s]
config.json: 100%|███████████████████████████████████| 660/660 [00:00<00:00, 10.1MB/s]
model.safetensors: 100%|█████████████████████████| 2.62G/2.62G [01:01<00:00, 42.6MB/s]
generation_config.json: 100%|██████████████████████| 89.0/89.0 [00:00<00:00, 1.47MB/s]
Map: 100%|██████████████████████████████████████| 1/1 [00:00<00:00, 306.87 examples/s]
Map: 100%|██████████████████████████████████████| 1/1 [00:00<00:00, 526.59 examples/s]
{'eval_loss': nan, 'eval_runtime': 20.6386, 'eval_samples_per_second': 0.048, 'eval_steps_per_second': 0.048, 'epoch': 1.0}                                                 
{'train_runtime': 369.2747, 'train_samples_per_second': 0.003, 'train_steps_per_second': 0.003, 'train_loss': 6.90234375, 'epoch': 1.0}                                     
100%|██████████████████████████████████████████████████| 1/1 [06:09<00:00, 369.27s/it]

real    7m26.131s
user    8m36.002s
sys     0m27.625s
```

## TODO: En proceso el training a partir de los checkpoints (da error)


## TODO: Guardar el modelo para uso futuro.
