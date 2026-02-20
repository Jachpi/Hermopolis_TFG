import os
import re
import shutil

SOURCE_DIR = r"../Segmented_C3D/Segmented_C3D"

# Imprimir dir actual
print(f"Directorio actual: {os.getcwd()}")

pattern = re.compile(r"^P\d{2}T\d{2}C\d{2}(G\d{2})D\d{2}S\d{2}\.c3d$", re.IGNORECASE)

def classify_c3d_files(source_dir):

    if not os.path.isdir(source_dir):
        raise ValueError("El directorio especificado no existe.")
    
    source_dir = os.path.abspath(source_dir)

    #listar archivos en el directorio
    files = os.listdir(source_dir)

    print(files)

    for file in files:
        if not file.lower().endswith(".c3d"):
            continue

        match = pattern.match(file)

        if match:
            gesture_code = match.group(1)  # Extrae Gxx
            target_folder = os.path.join(source_dir, gesture_code)

            # Crear carpeta si no existe
            os.makedirs(target_folder, exist_ok=True)

            src_path = os.path.join(source_dir, file)
            dst_path = os.path.join(target_folder, file)

            # Evitar sobrescritura
            if not os.path.exists(dst_path):
                shutil.move(src_path, dst_path)
                print(f"Movido: {file} → {gesture_code}/")
            else:
                print(f"Ya existe en destino, se omite: {file}")
        else:
            print(f"Nombre no válido, se ignora: {file}")

    print("Clasificación completada.")

if __name__ == "__main__":
    classify_c3d_files(SOURCE_DIR)
