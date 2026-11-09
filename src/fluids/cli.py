from pathlib import Path

import typer

app = typer.Typer(
    name="fluids",
    help="Rendering stuff",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def render(
    vdb_dir: Path = typer.Argument(..., help="Directory containing the VDB sequence"),
    output_dir: Path = typer.Argument(..., help="Directory to save rendered frames"),
    threshold: float = typer.Option(
        0.1, help="Threshold value for volume-to-mesh conversion"
    ),
):
    """Render the vdbs in blender"""
    from fluids.render import run_render

    run_render(vdb_dir, output_dir, threshold)


@app.command("grid-to-vdb")
def grid_to_vdb(
    src: Path = typer.Argument(Path("data"), help="MPM grid dump file or directory"),
    dst: Path = typer.Argument(Path("data/vdb"), help="Output directory for VDB files"),
):
    """Convert bin files saved from simulation into vdbs"""
    from fluids.grid_to_vdb import convert_grids

    convert_grids(src, dst)


def main():
    app()
