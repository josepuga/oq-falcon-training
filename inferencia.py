#!/usr/bin/env python
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os



if len(sys.argv) < 3:
    print(f"Uso: {sys.argv[0]} <model_id> <prompt deseado>")
    exit(1)

data_path = f"./results-{sys.argv[1]}"
if not os.path.exists(data_path):
    print(f"Directorio {data_path} no existe.")
    exit(1)

prompt = " ".join(sys.argv[2:])

# Cargar el modelo entrenado
model = AutoModelForCausalLM.from_pretrained(data_path)
tokenizer = AutoTokenizer.from_pretrained(data_path)

# Realizar una inferencia
inputs = tokenizer(prompt, return_tensors="pt")

# Máscara de atención explícita, evitar el warning
# "Please pass your input's `attention_mask` to obtain reliable results"
inputs["attention_mask"] = (inputs.input_ids != tokenizer.pad_token_id).long()

# Generar el texto
# NOTA: Se puede crear más aleatoriedad indicando do_sample=True (por defecto es
# ----- false). Esto permite más parámetros.
# do_sample=True,
#  temperature=0.7,
#  top_p=0.9
    
outputs = model.generate(
    inputs.input_ids,
    attention_mask=inputs["attention_mask"],
    max_length=50,
    num_beams=3,  # Búsqueda con haz
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
