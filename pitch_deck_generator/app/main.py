import argparse
import pandas as pd
import re
from generation import generate
from models_config import models_config
from metrics import evaluate_metrics
from classification import classify_and_plot



def remove_think_tags(text):
    if not isinstance(text, str):
        return text

    while True:
        start = text.find("<think>")
        end = text.find("</think>", start)

        if start == -1 or end == -1:
            break

        text = text[:start] + text[end + len("</think>"):]

    return text


def main():
    parser = argparse.ArgumentParser(description="Generador de Pitch Decks")

    parser.add_argument(
        "--models", nargs="+",
        default=list(models_config.keys()),
        help="Modelos a usar (por defecto: todos)"
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument(
        "--prompt_type",
        nargs="+",
        choices=["generic", "structured"],
        default=["generic", "structured"],
        help="Tipo de prompt: generic, structured, o ambos (por defecto)"
    )
    parser.add_argument("--output_filename", type=str, default="./output/pitch_decks_generados.csv")

    args = parser.parse_args()

    df = pd.read_json("./data/data_cleaned.json")
    print("Tamaño del dataset:", df.shape[0])


    all_results = []

    for prompt_type in args.prompt_type:
        print(f"\n🔹 Generando usando prompt tipo: {prompt_type}")
        df_result = generate(
            df=df,
            models=args.models,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            num_samples=args.num_samples,
            prompt_type=prompt_type
        )
        all_results.append(df_result)

    df_final = pd.concat(all_results, ignore_index=True)

    # Eliminar contenido entre etiquetas <think>...</think>
    df_final["created_text"] = df_final["created_text"].apply(remove_think_tags)

    df_final.to_csv(args.output_filename, index=False)
    print(f"\Pitch decks guardados en {args.output_filename}")

    df_final = pd.read_csv(args.output_filename)
    print("Longitud del df generado: ",len(df_final))

    # Evaluar métricas
    df_metrics, resumen = evaluate_metrics(df_final, output_path="./output/pitch_decks_metrics.csv")

    # Imprimir resumen
    print("\Resumen de métricas por modelo y tipo de prompt:")
    print(resumen)
        # Clasificación e histogramas
    df_with_classification = classify_and_plot(df_final)



if __name__ == "__main__":
    main()
