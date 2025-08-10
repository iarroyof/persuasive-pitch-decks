# 🧠 Generador de Pitch Decks con Modelos de Lenguaje

Este proyecto genera automáticamente *pitch decks* persuasivos a partir de una tabla con datos de startups (nombre, descripción y categoría). Utiliza modelos de lenguaje avanzados como los de NVIDIA NIM o cualquier otro que configures fácilmente.

---

## 🛠️ Requisitos

Puedes usar este proyecto con **Anaconda** y con **pip**. Ambos métodos están explicados abajo.

---
Nota: Asegurate de poner en el archivo "generation.py" tu api key de nvidia en el apartado de api_key="{tu api key}"

## 🐍 Opción A: Instalación con Conda (recomendada)

1. Abre Anaconda Prompt.
2. Ejecuta los siguientes comandos:

```bash
conda env create -f environment.yml  
conda activate pitchdeck
```

---

## 📦 Seguimiento B: Una vez activado el entorno se hace la instalación con pip


1. Instala las dependencias:

```bash
pip install -r requirements.txt
```

---

## 🚀 Cómo usar el generador

Una vez configurado el entorno, ejecuta el script principal `main.py` desde la línea de comandos.

### 📌 Sintaxis general

```bash
python main.py   --models MODEL1 MODEL2   --temperature 0.7   --top_p 0.9   --max_tokens 512   --num_samples 10   --prompt_type generic structured   --output_filename ./output/pitch_decks_generados.csv
```

Los modelos disponibles son: "marin/marin-8b-instruct", "deepseek-ai/deepseek-r1", "qwen/qwen3-235b-a22b", "openai/gpt-oss-20b" y "openai/gpt-oss-120b". En dado caso de no especificar ningún modelo se ejecutarán los 3, generando pitch decks con cada uno de ellos. 

### 🧠 Argumentos explicados

| Argumento             | Descripción                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `--models`            | Lista de modelos a usar (ver disponibles en `models_config.py`).            |
| `--temperature`       | Controla la creatividad del texto generado. Valores altos = más creativo.  |
| `--top_p`             | Probabilidad acumulada para muestreo nuclear.                              |
| `--max_tokens`        | Máximo de tokens por pitch generado.                                       |
| `--num_samples`       | Cuántas filas del dataset quieres procesar.                                |
| `--prompt_type`       | Tipo de prompt a usar: `generic`, `structured` o ambos.                     |
| `--output_filename`   | Ruta al archivo CSV donde guardar los resultados.                           |

---

## 📂 Archivos clave del proyecto

- `main.py`: punto de entrada principal para generar y clasificar los pitch decks.
- `generation.py`: genera el texto con los modelos LLM.
- `classification.py`: clasifica los textos generados en técnicas de persuasión.
- `models_config.py`: define configuraciones específicas por modelo.
- `prompts.py`: define los prompts genéricos o estructurados.
- `data/`: carpeta donde puedes poner tus archivos CSV de entrada.
- `output/`: carpeta donde se guardan los resultados generados.

---

## 📊 Ejemplo de uso real

Supongamos que quieres usar los modelos `marin/marin-8b-instruct` y `qwen/qwen3-235b-a22b`, generar hasta 10 muestras con ambos tipos de prompt, y guardar los resultados en un archivo llamado `resultados.csv`.

```bash
python main.py   --models marin/marin-8b-instruct qwen/qwen3-235b-a22b   --temperature 0.7   --top_p 0.9   --max_tokens 512   --num_samples 10   --prompt_type generic structured   --output_filename ./output/resultados.csv
```

---

## 🧯 Solución de problemas comunes

- Si ves errores como `ModuleNotFoundError`, asegúrate de haber activado el entorno antes de correr el script.
- Si `sentencepiece` no se instala, usa Conda en lugar de pip.
- Si el modelo falla al generar texto, el script intentará nuevamente automáticamente.

---

