from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os

# Diccionario que contiene los diferentes modelos. Añadir el que se quiera manualmente
# Consiste en un ID y el  nombre completo para usarlo.
MODELS = {
    "falcon1b": "ericzzz/falcon-rw-1b-instruct-openorca",
    "falcon7b": "tiiuae/falcon-7b",
}
# Rutas. BASE + "-" + model_id
BASE_OUTPUT_DIR = "./results"
BASE_LOGS_DIR = "./logs"


class AIModel:
    """
    Clase que representa un modelo General de IA usando transformers.
    """

    def __init__(self, model_id):
        """
        Constructor, usa los parámetros definidos en set_default_parameters()
        """
        # Al definir el model_id, será BASE_xxx "-" + model_id
        self.training_output_dir = BASE_OUTPUT_DIR
        self.training_loggin_dir = BASE_LOGS_DIR
        self.training_last_checkpoint = None
        self.training_eval_strategy = "epoch"

        self._model_name = ""
        self.model_id = model_id
        self.max_tokens = None
        self.top_p = None
        self.temperature = None
        self.repetition_penalty = None
        self.frequency_penalty = None
        self.presence_penalty = None
        self.set_default_parameters()

        self.batched = None
        self.batch_size = None
        self.set_tokenize_params()  # Parámetros por defecto (no veo necesario crear un método default)

        self.training_arguments = self.get_default_training_arguments()
        self._trainer = None

        self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Ahora que conocemos la ruta, asignamos el ultimo checkpoint para continuar
        # la sesión si lo hubiera.
        self._set_last_checkpoint()

        # Si se quisiera un pad_token específicio en vez de eos_token:
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if self.training_last_checkpoint:
            print(f"DEBUG: Cargando modelo desde checkpoint: {self.training_last_checkpoint}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.training_last_checkpoint,
                torch_dtype=torch.float16,
                device_map="auto",
                ignore_mismatched_sizes=True,
            )   
        else:
            print("DEBUG: No se encontró checkpoint, cargando modelo base")
            self.model = AutoModelForCausalLM.from_pretrained(
                self._model_name, 
                torch_dtype=torch.float16, 
                device_map="auto"                
            )

    # Propiedad que definir el model_name y algunos parámetros más a partir del
    # model_id. Como es un dado crucial, se manda una excepción si es erróneo
    @property
    def model_id(self):
        """."""
        return self._model_id

    @model_id.setter
    def model_id(self, model_id):
        """."""
        if not model_id in MODELS:
            raise ValueError("model_id {model_id}, no está registrado.")
        self._model_id = model_id
        self._model_name = MODELS[model_id]

        # Se cambian las rutas, esto permite trabajar con diversos modelos
        self.training_output_dir = f"{BASE_OUTPUT_DIR}-{model_id}"
        self.training_loggin_dir = f"{BASE_LOGS_DIR}-{model_id}"

    def _set_last_checkpoint(self):
        """
        Busca el último checkpoint en el directorio de salida y lo asigna.
        Si no encuentra ningún checkpoint, se asigna una cadena vacía.
        Este método, lógicamente ha de llamarse depueś de asignar
        """
        checkpoints = []
        # Listar todos los subdirectorios en `self.training_output_dir`
        if os.path.exists(self.training_output_dir):
            for entry in os.listdir(self.training_output_dir):
                if entry.startswith("checkpoint-") and entry[len("checkpoint-"):].isdigit():
                    checkpoints.append(int(entry[len("checkpoint-"):]))

        # Si hay checkpoints, selecciona el mayor
        if checkpoints:
            last_checkpoint = max(checkpoints)
            self.training_last_checkpoint = f"{self.training_output_dir}/checkpoint-{last_checkpoint}"
        else:
            # No se encontraron checkpoints
            self.training_last_checkpoint = None
        

    def fine_tune(self, train_dataset, eval_dataset=None):
        """
        Entrena el modelo con los datasets indicados.
        :param train_dataset, dataset tokenizado para entrenamiento.
        :param eval_dataset, dataset tokenizado para evaluación.
        """

        # Configurar el Trainer
        self._trainer = Trainer(
            model=self.model,
            args=self.training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Entrenar el modelo
        print(f"DEBUG: last checkpoint = {self.training_last_checkpoint}")
        self._trainer.train(resume_from_checkpoint=self.training_last_checkpoint)

        # Guardar el modelo y el tokenizador entrenado
        self.model.save_pretrained(self.training_output_dir)
        self.tokenizer.save_pretrained(self.training_output_dir)

    def generate(self, prompt) -> str:
        """
        Genera una respuesta a partir del prompt prompt.
        :param prompt: str, entrada para el modelo.
        :return: str, respuesta generada.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def set_default_parameters(self):
        """
        Establece los parémetros por defecto, que son los indicados normalmente
        para respuestas 'coherentes'. TODO: Afinar estos.
        ----------------------------------------------------------------------------
        Parámetro	        Foco principal	                            Comentarios
        repetition_penalty	Repetición de cualquier palabra	            Penaliza palabras repetidas previamente generadas. Generación de texto largo sin repeticiones monótonas.
        frequency_penalty	Frec. de aparición de palabras específicas	Penaliza palabras según cuántas veces han aparecido. Diversificar palabras comunes en una respuesta.
        presence_penalty	Presencia de palabras en la salida	        Penaliza si una palabra ya apareció en algún punto.	Evitar duplicar palabras clave.
        max_tokens	        Longitud de la respuesta	                Define el número máximo de tokens generado. Limitar la extensión de respuestas largas o divagantes.
        top_p	            Probabilidad acumulativa (nucleus sampling)	Selecciona solo palabras dentro del rango acumulativo de probabilidad top_p. Usar valores más bajos para controlar precisión en la generación.
        temperature	        Aleatoriedad en la generación	            Valores más altos generan texto más creativo, valores bajos generan texto más predecible. Creatividad en historias (1.0+) o respuestas precisas (<0.7).
        ----------------------------------------------------------------------------
        """
        self.set_parameters(
            max_tokens=50,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.2,
            frequency_penalty=None,
            presence_penalty=None,
        )

    def set_parameters(
        self,
        max_tokens=None,
        top_p=None,
        temperature=None,
        repetition_penalty=1.2,
        frequency_penalty=None,
        presence_penalty=None,
    ):
        """
        Setea los parámetros del modelo.
        """
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if top_p is not None:
            self.top_p = top_p
        if repetition_penalty is not None:
            self.repetition_penalty = repetition_penalty
        if presence_penalty is not None:
            self.presence_penalty = presence_penalty
        if frequency_penalty is not None:
            self.frequency_penalty = frequency_penalty

    def set_tokenize_params(self, batched=True, batch_size=4):
        """
        Estable los valores por defecto para la 'tokenización' del entrenamiento
        del modelo.AutoTokenizer
        :param batched, indica si se va a usar procesar por lotes o las entradas una a una.
        :param batch_size, si se elige por lotes, cuando elementos en cada proceso.
        """

    def tokenize_dataset(self, dataset):
        """
        Tokeniza un dataset con el tokenizador de Falcon.
        """
        try:
            return dataset.map(
                lambda sample: self.tokenizer(
                    sample["prompt"],
                    text_target=sample["response"],
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_tokens,
                ),
                batched=True,
            )
        except Exception as e:
            print(f"Error durante la tokenización: {e}")
            return

    def get_parameters(self) -> dict:
        """
        Retrieve the current parameters of the model.
        :return: dict, the model parameters.
        """
        return {
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.frequency_penalty,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

    def get_default_training_arguments(self):
        """
        Devuelve una configuración por defecto para TrainingArguments.
        """
        # TODO: Aprender más que hace cada parámetro.
        return TrainingArguments(
            output_dir=self.training_output_dir,  # Directorio obligatorio
            eval_strategy=self.training_eval_strategy,  # Estrategia de evaluación
            learning_rate=5e-5,  # Tasa de aprendizaje
            per_device_train_batch_size=32,  # 4,  # Tamaño del lote
            num_train_epochs=1,  # Número de épocas
            weight_decay=0.01,  # Decaimiento del peso
            save_total_limit=2,  # Máximo número de checkpoints
            fp16=True,  # Precisión mixta si se usa GPU
            logging_dir=self.training_loggin_dir,  # Carpeta para logs
            logging_steps=10,  # Pasos entre logs
            max_steps=10, # ¿Ayuda a hacer grande el dataset para continuar en checkpoints?
            save_steps=500,  # Guardar modelo cada 500 pasos
        )
