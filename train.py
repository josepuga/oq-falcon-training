#!/usr/bin/env python
import json
import torch  # Para forzar más núcleos
from datasets import Dataset
from transformers import AutoTokenizer
from ai_model import AIModel

### IMPORTANTE: Revisar estas constantes antes de ejecutar el trainer.

# El id del modelo a usar
MODEL_ID = "falcon1b"
# El dataset usado para el entrenamiento
DATASET = "mini_dataset2.json"
# Porcentaje del dataset que se usará para test
TEST_PERCENT = 0.2
# Cores a usar, poner máximo del equipo. FIXME: No termina de funcionar correctamente
CPU_CORES = 24
# Tamaño de los lotes del procesamiento depende de la RAM
BATH_SIZE = 32


def load_dataset_from_json(dataset_path) -> Dataset:
    """
    Carga un archivo JSON como un objeto Dataset de Hugging Face.
    :param dataset_path: Ruta al archivo JSON.
    :return: Dataset cargado.
    """
    try:
        # Cargar datos desde el JSON
        with open(dataset_path, "r") as file:
            data = json.load(file)

        # Convertir a Dataset de Hugging Face
        return Dataset.from_list(data)

    except FileNotFoundError:
        print(f"Error: El archivo {dataset_path} no existe.")
        exit(1)

    except json.JSONDecodeError:
        print(f"Error: El archivo {dataset_path} no tiene un formato JSON válido.")
        exit(1)


def main():

    # Instanciar el modelo
    my_model = AIModel(MODEL_ID)

    #### Mi configuración:  Ryzen 9 7900 - 128G RAM
    # Forzar a torch a usar todos mis núcleos
    torch.set_num_threads(CPU_CORES)
    my_model.training_arguments.per_device_train_batch_size = BATH_SIZE

    # Ruta al dataset
    dataset_path = DATASET

    # Cargar y tokenizar dataset
    dataset = load_dataset_from_json(dataset_path)

    # Se divide un % para entrenamiento y otro para evaluación
    split_datasets = dataset.train_test_split(test_size=TEST_PERCENT)
    train_dataset = split_datasets["train"]
    eval_dataset = split_datasets["test"]

    # tokenized_dataset = tokenize_dataset(dataset, falcon_model.tokenizer, falcon_model.max_tokens)
    tokenized_train_dataset = my_model.tokenize_dataset(train_dataset)
    tokenized_eval_dataset = my_model.tokenize_dataset(eval_dataset)

    # Entrenar el modelo con el dataset tokenizado
    # falcon_model.fine_tune(tokenized_dataset)
    my_model.fine_tune(tokenized_train_dataset, tokenized_eval_dataset)


if __name__ == "__main__":
    main()
