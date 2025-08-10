from dotenv import load_dotenv
import os

# Carga las variables del archivo .env
load_dotenv()  # Busca automáticamente el archivo .env en la misma carpeta

# Accede a las variables
api_key = os.getenv("API_KEY")
print("API Key:", api_key)