#!/bin/bash
# Script de instalación genérico para proyectos Python en Linux
# By José Puga. v0.1.0
#
# Este script:
# · Si no existe, crea el Entorno Virtual.
# · Activa el EV.
# · Si en el sistema no estuviera 'pipreqs' lo instalaría en el EV para no "ensuciar".
# · Crea el archivo requirements.txt con pipreqs.
# · Instala las dependencias.

VENV_PATH=./venv
REQ_FILE=requirements.txt

Setup() {
    # Comprobar que el script se llama con 'source'
    if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then 
        echo "Ejecutar con 'source $0' o '. $0'"
        echo "Esto es necesario para que se cree el Entorno Virtual en la shell actual."
        echo ""
        exit 1 
    fi

    # No ejecutar con EV activo
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo "El Entorno Virtual debe estar desactivado con 'deactivate'"
        return 1
    fi

    # Por seguridad, que el usuario borre manualmente el $VENV_PATH y lo intente
    if [[ -d "$VENV_PATH" ]]; then
        echo "El Entorno Virtual $VENV_PATH ya existe. Si quisieras regenerarlo, tendrás que borrarlo manualmente."
        echo "Pulsa 'ENTER' para continuar con el EV actual o 'CTRL-C' para cancelar"
        read -r
    else
        # Crear el entorno virtual
        echo "Creando EV..."
        if ! python -m venv venv; then
            echo "Error ejecutando 'python -m venv venv'".
            return 1
        fi
    fi

    # Comprobar que pipreqs esté instalado globalmente ante de activar el EV
    pipreqs_installed=$(command -v pipreqs &>/dev/null)
    
    # Activar el entorno virtual
    echo "Activando EV..."
    if ! source "$VENV_PATH/bin/activate"; then
        echo "Error activando en EV."
        return 1
    fi

    # Si pipreqs no estuviera globalmente, se instala en el EV
    if [[ "$pipreqs_installed" -ne 0 ]]; then
        echo "pipreqs no está instalado globalmente, se instalará en el EV"
        pip install pipreqs
    fi

    # Crear req..txt
    echo "Creando $REQ_FILE..."
    if ! pipreqs . --force; then
        echo "Error creando $REQ_FILE."
        return 1
    fi

    # Instalar las dependencias del req..txt
    echo "Instalando las dependencias"
    if ! pip install -r "$REQ_FILE"; then
        echo "Error al instalar la dependencias."
        return 1
    fi

    echo "Instalación y activación del EV completado."
    echo "Recuerda salir del EV cuando acabes con 'deactivate'."
}

Setup