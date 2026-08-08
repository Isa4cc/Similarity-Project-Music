"""Comandos CLI para el buscador de canciones similares.

Define tres subcomandos:
- search: busca canciones similares a un conjunto de entrada
- catalog: agrega canciones al catálogo de embeddings
- healthcheck: verifica que CLAP funciona en el hardware actual
"""

import logging
import sys
import time
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from proyecto_similitud.audio.embedder import CLAPEmbedder, AudioEmbedder
from proyecto_similitud.data.itunes_client import find_best_match, search_tracks
from proyecto_similitud.data.preview_downloader import cleanup_preview, download_preview
from proyecto_similitud.similarity.engine import (
    DEFAULT_WEIGHT_MAIN,
    DEFAULT_WEIGHT_OPTIONAL,
    build_query_vector,
    rank_similar,
)
from proyecto_similitud.storage.cache import EmbeddingCache

logger = logging.getLogger(__name__)
console = Console()

# Path por defecto para la DB
DEFAULT_DB_PATH = Path("data/embeddings.db")


def _setup_logging(verbose: bool) -> None:
    """Configura el nivel de logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _get_or_compute_embedding(
    track: dict,
    cache: EmbeddingCache,
    embedder: AudioEmbedder,
) -> np.ndarray | None:
    """Obtiene el embedding de una canción (del cache o computándolo).

    Returns:
        El embedding, o None si no se pudo calcular.
    """
    track_id = track["trackId"]
    track_name = track["trackName"]
    artist_name = track["artistName"]

    # ¿Ya está en cache?
    cached = cache.get(track_id)
    if cached is not None:
        console.print(f"  ✓ [dim]{track_name}[/dim] — [green]cache hit[/green]")
        return cached

    # Necesitamos calcular: descargar preview y embeddear
    preview_url = track.get("previewUrl")
    if not preview_url:
        console.print(
            f"  ✗ [dim]{track_name}[/dim] — [red]sin preview disponible[/red]"
        )
        return None

    wav_path = None
    try:
        wav_path = download_preview(preview_url)
        embedding = embedder.embed(wav_path)

        # Guardar en cache
        cache.put(
            track_id=track_id,
            track_name=track_name,
            artist_name=artist_name,
            embedding=embedding,
            album_name=track.get("collectionName"),
            preview_url=preview_url,
            model_id=embedder.model_id,
        )

        console.print(
            f"  ✓ [dim]{track_name}[/dim] — "
            f"[cyan]embedding calculado ({embedding.shape[0]} dims)[/cyan]"
        )
        return embedding

    except Exception as e:
        console.print(
            f"  ✗ [dim]{track_name}[/dim] — [red]error: {e}[/red]"
        )
        logger.exception(f"Error procesando '{track_name}'")
        return None

    finally:
        if wav_path is not None:
            cleanup_preview(wav_path)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Mostrar logs detallados")
@click.option(
    "--db",
    type=click.Path(),
    default=str(DEFAULT_DB_PATH),
    help="Ruta a la base de datos de embeddings",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, db: str) -> None:
    """🎵 Buscador de canciones similares por embeddings de audio."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db
    ctx.obj["verbose"] = verbose


@cli.command()
@click.option(
    "--main",
    "-m",
    multiple=True,
    required=True,
    help="Canciones principales (5 recomendadas). Se pueden pasar múltiples.",
)
@click.option(
    "--optional",
    "-o",
    multiple=True,
    help="Canciones opcionales para contexto adicional.",
)
@click.option("--top", "-n", default=10, help="Número de resultados (default: 10)")
@click.option("--country", default="US", help="Código de país para iTunes (default: US)")
@click.option(
    "--weight-main",
    type=float,
    default=DEFAULT_WEIGHT_MAIN,
    help=f"Peso para canciones principales (default: {DEFAULT_WEIGHT_MAIN})",
)
@click.option(
    "--weight-optional",
    type=float,
    default=DEFAULT_WEIGHT_OPTIONAL,
    help=f"Peso para canciones opcionales (default: {DEFAULT_WEIGHT_OPTIONAL})",
)
@click.pass_context
def search(
    ctx: click.Context,
    main: tuple[str, ...],
    optional: tuple[str, ...],
    top: int,
    country: str,
    weight_main: float,
    weight_optional: float,
) -> None:
    """Busca canciones similares a un conjunto de entrada."""
    db_path = ctx.obj["db_path"]

    console.print("\n[bold]🎵 Buscador de Canciones Similares[/bold]\n")

    embedder = CLAPEmbedder()

    with EmbeddingCache(db_path) as cache:
        catalog_count = cache.count()
        if catalog_count == 0:
            console.print(
                "[yellow]⚠ El catálogo está vacío. Usa 'catalog' para agregar "
                "canciones primero.[/yellow]\n"
            )
            return

        console.print(
            f"[dim]Catálogo: {catalog_count} canciones con embeddings[/dim]\n"
        )

        # --- Fase 1: Resolver y embeddear canciones de entrada ---
        console.print("[bold]Procesando canciones principales:[/bold]")
        main_embeddings: list[np.ndarray] = []
        input_track_ids: set[int] = set()

        for song_name in main:
            track = find_best_match(song_name, country=country)
            if track is None:
                console.print(
                    f"  ✗ [red]No se encontró: '{song_name}'[/red]"
                )
                continue
            emb = _get_or_compute_embedding(track, cache, embedder)
            if emb is not None:
                main_embeddings.append(emb)
                input_track_ids.add(track["trackId"])

        optional_embeddings: list[np.ndarray] = []
        if optional:
            console.print("\n[bold]Procesando canciones opcionales:[/bold]")
            for song_name in optional:
                track = find_best_match(song_name, country=country)
                if track is None:
                    console.print(
                        f"  ✗ [red]No se encontró: '{song_name}'[/red]"
                    )
                    continue
                emb = _get_or_compute_embedding(track, cache, embedder)
                if emb is not None:
                    optional_embeddings.append(emb)
                    input_track_ids.add(track["trackId"])

        if not main_embeddings:
            console.print(
                "\n[red]Error: No se pudo obtener ningún embedding de las "
                "canciones principales. Abortando.[/red]"
            )
            return

        # --- Fase 2: Construir vector de consulta ---
        console.print(f"\n[dim]Construyendo vector de consulta "
                      f"({len(main_embeddings)} principales, "
                      f"{len(optional_embeddings)} opcionales)...[/dim]")

        query_vector = build_query_vector(
            main_embeddings,
            optional_embeddings,
            weight_main=weight_main,
            weight_optional=weight_optional,
        )

        # --- Fase 3: Ranking ---
        catalog = cache.get_all()
        results = rank_similar(
            query_vector, catalog, top_n=top, exclude_ids=input_track_ids
        )

        # --- Fase 4: Mostrar resultados ---
        if not results:
            console.print(
                "\n[yellow]No se encontraron canciones similares en el "
                "catálogo.[/yellow]"
            )
            return

        console.print(f"\n[bold]🎯 Top {len(results)} canciones similares:[/bold]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=3)
        table.add_column("Canción", style="cyan")
        table.add_column("Artista", style="green")
        table.add_column("Álbum", style="dim")
        table.add_column("Score", justify="right", style="yellow")

        for i, (track_id, score) in enumerate(results, 1):
            meta = cache.get_track_metadata(track_id)
            if meta:
                table.add_row(
                    str(i),
                    meta["track_name"],
                    meta["artist_name"],
                    meta.get("album_name") or "—",
                    f"{score:.4f}",
                )
            else:
                table.add_row(str(i), f"track_id={track_id}", "?", "?", f"{score:.4f}")

        console.print(table)
        console.print()


@cli.command()
@click.option("--artist", "-a", help="Buscar canciones de un artista")
@click.option("--query", "-q", help="Búsqueda genérica en iTunes")
@click.option(
    "--limit", "-l", default=10, help="Número de canciones a agregar (default: 10)"
)
@click.option("--country", default="US", help="Código de país para iTunes (default: US)")
@click.pass_context
def catalog(
    ctx: click.Context,
    artist: str | None,
    query: str | None,
    limit: int,
    country: str,
) -> None:
    """Agrega canciones al catálogo de embeddings."""
    db_path = ctx.obj["db_path"]

    if not artist and not query:
        console.print("[red]Debes especificar --artist o --query[/red]")
        return

    search_term = artist or query
    console.print(f"\n[bold]📀 Agregando canciones al catálogo[/bold]")
    console.print(f"[dim]Búsqueda: '{search_term}', límite: {limit}[/dim]\n")

    embedder = CLAPEmbedder()

    with EmbeddingCache(db_path) as cache:
        initial_count = cache.count()

        # Buscar en iTunes
        tracks = search_tracks(search_term, country=country, limit=limit)

        if not tracks:
            console.print(f"[yellow]No se encontraron resultados para '{search_term}'[/yellow]")
            return

        console.print(f"[dim]Encontradas {len(tracks)} canciones. Procesando...[/dim]\n")

        added = 0
        skipped = 0
        failed = 0

        for track in tracks:
            track_id = track.get("trackId")
            if not track_id:
                continue

            if cache.has(track_id):
                console.print(
                    f"  → [dim]{track['trackName']}[/dim] — "
                    f"[yellow]ya en catálogo[/yellow]"
                )
                skipped += 1
                continue

            emb = _get_or_compute_embedding(track, cache, embedder)
            if emb is not None:
                added += 1
            else:
                failed += 1

        final_count = cache.count()
        console.print(
            f"\n[bold]Resumen:[/bold] {added} agregadas, {skipped} ya existentes, "
            f"{failed} fallidas. Total en catálogo: {final_count}\n"
        )


@cli.command()
@click.pass_context
def healthcheck(ctx: click.Context) -> None:
    """Verifica que CLAP funciona correctamente en este hardware."""
    import psutil

    console.print("\n[bold]🏥 Health Check — Verificando CLAP[/bold]\n")

    # 1. Verificar ffmpeg
    console.print("[bold]1. Verificando ffmpeg...[/bold]")
    import shutil
    if shutil.which("ffmpeg"):
        console.print("  ✓ [green]ffmpeg encontrado[/green]")
    else:
        console.print("  ✗ [red]ffmpeg NO encontrado. Instálalo antes de continuar.[/red]")
        return

    # 2. Descargar un preview de prueba
    console.print("\n[bold]2. Descargando preview de prueba...[/bold]")
    try:
        test_track = find_best_match("Bohemian Rhapsody Queen")
        if not test_track or not test_track.get("previewUrl"):
            console.print("  ✗ [red]No se pudo obtener un preview de prueba[/red]")
            return
        console.print(
            f"  ✓ [green]Canción: '{test_track['trackName']}' "
            f"por {test_track['artistName']}[/green]"
        )

        wav_path = download_preview(test_track["previewUrl"])
        console.print(f"  ✓ [green]Preview descargado y convertido a WAV[/green]")
    except Exception as e:
        console.print(f"  ✗ [red]Error descargando preview: {e}[/red]")
        return

    # 3. Correr CLAP
    console.print("\n[bold]3. Ejecutando CLAP (esto puede tardar un momento)...[/bold]")

    ram_before = psutil.Process().memory_info().rss / (1024 * 1024)
    start_time = time.time()

    try:
        embedder = CLAPEmbedder()
        embedding = embedder.embed(wav_path)
        elapsed = time.time() - start_time
        ram_after = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception as e:
        console.print(f"  ✗ [red]Error ejecutando CLAP: {e}[/red]")
        cleanup_preview(wav_path)
        return
    finally:
        cleanup_preview(wav_path)

    # 4. Reportar resultados
    console.print(f"\n[bold]📊 Resultados:[/bold]")
    console.print(f"  • Dimensión del embedding: {embedding.shape[0]}")
    console.print(f"  • Norma L2: {np.linalg.norm(embedding):.4f}")
    console.print(f"  • Tiempo de inferencia: {elapsed:.1f}s")
    console.print(f"  • RAM usada (antes): {ram_before:.0f} MB")
    console.print(f"  • RAM usada (después): {ram_after:.0f} MB")
    console.print(f"  • RAM delta: {ram_after - ram_before:.0f} MB")

    # Veredicto
    if elapsed > 30:
        console.print(
            f"\n[yellow]⚠ CLAP tardó {elapsed:.0f}s (>30s). Considera usar "
            f"el fallback de Librosa o probar con ONNX Runtime.[/yellow]"
        )
    elif elapsed > 10:
        console.print(
            f"\n[yellow]⚠ CLAP tardó {elapsed:.0f}s. Aceptable pero algo lento. "
            f"La primera inferencia suele ser más lenta.[/yellow]"
        )
    else:
        console.print(
            f"\n[green]✓ CLAP funciona correctamente. {elapsed:.1f}s por canción "
            f"es excelente para uso personal.[/green]"
        )

    console.print()
