"""Entry point principal para el CLI.

Uso:
    python -m proyecto_similitud <comando> [opciones]
    similitud <comando> [opciones]  (si se instaló con pip install -e .)
"""

from proyecto_similitud.cli.commands import cli

if __name__ == "__main__":
    cli()
