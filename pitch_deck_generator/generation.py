# generation.py

from openai import OpenAI
import pandas as pd
from prompts import generic_prompt, structured_prompt
from models_config import models_config
import time

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="{tu api key}"
)

def generate(df, models, temperature, top_p, max_tokens, num_samples, prompt_type, max_attempts=15):
    rows = []

    if num_samples is None or num_samples > len(df):
        num_samples = len(df)

    for model_name in models:
        if model_name not in models_config:
            print(f"Modelo no reconocido: {model_name}")
            continue

        print(f"\nUsando modelo: {model_name}")
        config = models_config[model_name]
        config["params"].update({
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        })

        for i in range(num_samples):
            row = df.iloc[i]
            category = row["category_name"]
            name = row["name"]
            blurb = row["blurb"]

            attempt = 0
            success = False

            while attempt < max_attempts and not success:
                try:
                    print(f"Generando muestra {i+1}, intento {attempt+1}...")

                    if prompt_type == "generic":
                        messages = [{"role": "user", "content": generic_prompt(category, name)}]
                    elif prompt_type == "structured":
                        messages = [
                            {"role": "system", "content": "You are a professional pitch deck generator."},
                            {"role": "user", "content": structured_prompt(category, name, blurb)}
                        ]
                    else:
                        raise ValueError("prompt_type no válido. Use 'generic' o 'structured'.")

                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=True,
                        **config["params"]
                    )

                    full_text = ""
                    for chunk in completion:
                        fragment = config["extract_func"](chunk)
                        if fragment:
                            full_text += fragment

                    rows.append({
                        "category_name": category,
                        "original_text": blurb,
                        "created_text": full_text,
                        "name": name,
                        "model_name": model_name,
                        "prompt_type": prompt_type
                    })

                    print(f"Generación exitosa para muestra {i+1}:\n{full_text[:200]}...\n")
                    success = True

                    if (i + 1) % 10 == 0:
                        print(f"{i+1}/{num_samples} completados.")

                except Exception as e:
                    attempt += 1
                    print(f"Error en intento {attempt} con modelo {model_name}, muestra {i+1}: {e}")
                    if attempt < max_attempts:
                        time.sleep(2)  # Delay entre intentos
                    else:
                        print(f"Se alcanzó el máximo de intentos para muestra {i+1}. Se omite.\n")

    return pd.DataFrame(rows)
