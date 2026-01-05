"""
NEXUS Main Entry Point

Provides CLI and interactive demo functionality.
"""

import click
import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import NexusConfig
from .core import EmotionalPresets, EmotionalState
from .engine import BicameralEngine, ProcessingResult

console = Console()


def create_status_panel(engine: BicameralEngine, result: ProcessingResult) -> Panel:
    """Create a rich panel showing engine status."""
    state = result.final_state or EmotionalPresets.NEUTRAL
    routing = result.routing

    # Create status table
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("Emotional State", f"V={state.valence:+.2f} A={state.arousal:+.2f}")
    table.add_row("Quadrant", state.quadrant.name)

    if routing:
        table.add_row("Manifold", routing.primary_manifold.name)
        table.add_row("Gate Value", f"{routing.gate_value:.2f}")
        table.add_row("Intrinsic Dimension", f"{routing.intrinsic_dimension:.3f}")
        table.add_row("Regime", routing.fractal_result.regime if routing.fractal_result else "N/A")

    table.add_row("Task Type", result.detected_task)
    table.add_row("Processing Time", f"{result.processing_time_ms:.2f} ms")

    return Panel(table, title="[bold green]NEXUS Status[/]", border_style="green")


def create_circumplex_display(state: EmotionalState) -> str:
    """Create ASCII art of the Russell Circumplex with current position."""
    # 11x11 grid centered at (5, 5)
    grid = [[' ' for _ in range(21)] for _ in range(11)]

    # Draw axes
    for i in range(21):
        grid[5][i] = '─'  # Horizontal axis
    for i in range(11):
        grid[i][10] = '│'  # Vertical axis
    grid[5][10] = '┼'  # Center

    # Mark quadrants
    grid[1][3] = 'A'  # Activated Unpleasant
    grid[1][17] = 'P'  # Activated Pleasant
    grid[9][3] = 'U'  # Deactivated Unpleasant
    grid[9][17] = 'S'  # Deactivated Pleasant (Serene)

    # Plot current state
    x = int(10 + state.valence * 9)  # Map -1..1 to 1..19
    y = int(5 - state.arousal * 4)   # Map -1..1 to 9..1
    x = max(0, min(20, x))
    y = max(0, min(10, y))
    grid[y][x] = '●'

    # Convert to string
    lines = [''.join(row) for row in grid]
    return '\n'.join(lines)


def create_id_gauge(id_value: float) -> str:
    """Create a gauge showing intrinsic dimension relative to target."""
    target = 1.8
    gauge_width = 30

    # Map ID to gauge position
    pos = int((id_value - 1.0) / 1.5 * gauge_width)
    pos = max(0, min(gauge_width - 1, pos))

    target_pos = int((target - 1.0) / 1.5 * gauge_width)

    gauge = ['─'] * gauge_width
    gauge[target_pos] = '▼'  # Target marker
    gauge[pos] = '●'  # Current position

    return f"1.0 |{''.join(gauge)}| 2.5   ID={id_value:.2f}"


@click.group()
def cli() -> None:
    """NEXUS: Neuro-Epistemic eXploration and Unified Synthesis Engine"""
    pass


@cli.command()
def demo() -> None:
    """Run interactive NEXUS demo."""
    console.print(Panel.fit(
        "[bold magenta]NEXUS Demo[/]\n"
        "[dim]Neuro-Epistemic eXploration and Unified Synthesis Engine[/]",
        border_style="magenta"
    ))

    # Initialize engine
    engine = BicameralEngine()
    console.print("[green]✓[/] Engine initialized")

    # Demo loop
    console.print("\n[cyan]Enter text prompts to see NEXUS process them.[/]")
    console.print("[dim]Type 'quit' to exit, 'stats' for statistics, 'reset' to reset.[/]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/] ")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'stats':
            stats = engine.get_stats()
            console.print(Panel(str(stats), title="Engine Statistics"))
            continue
        elif user_input.lower() == 'reset':
            engine.reset()
            console.print("[yellow]Engine reset.[/]")
            continue

        # Generate synthetic input data
        input_data = np.random.randn(64).astype(np.float32)

        # Process through engine
        result = engine.process(input_data, text_input=user_input)

        # Display results
        console.print()
        console.print(create_status_panel(engine, result))

        # Show Circumplex
        if result.final_state:
            console.print("\n[bold]Russell Circumplex:[/]")
            console.print(create_circumplex_display(result.final_state))

        # Show ID gauge
        if result.routing:
            console.print("\n[bold]Fractal Dimension:[/]")
            console.print(create_id_gauge(result.routing.intrinsic_dimension))

        console.print()

    console.print("\n[magenta]NEXUS demo complete.[/]")


@cli.command()
def info() -> None:
    """Show NEXUS system information."""
    from . import __version__

    console.print(Panel.fit(
        f"[bold]NEXUS v{__version__}[/]\n\n"
        "[cyan]Components:[/]\n"
        "  • Emotional Control Plane (Russell Circumplex)\n"
        "  • PID Controller (Emotional Regulation)\n"
        "  • Geometric Router (ACC with NFE)\n"
        "  • Bicameral Engine (Logic + Creative)\n"
        "  • Dopamine Reward Function\n"
        "  • Synthesizer (4th Brain)\n\n"
        "[yellow]Target Fractal Dimension:[/] D_H ≈ 1.8 (Edge of Chaos)",
        title="System Information",
        border_style="blue"
    ))


@cli.command()
@click.option('--iterations', '-n', default=100, help='Number of test iterations')
def benchmark(iterations: int) -> None:
    """Run performance benchmark."""
    import time

    console.print(f"[cyan]Running {iterations} iterations...[/]")

    engine = BicameralEngine()

    times = []
    for i in range(iterations):
        input_data = np.random.randn(64).astype(np.float32)
        start = time.time()
        engine.process(input_data, text_input=f"Test prompt {i}")
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    avg = np.mean(times)
    std = np.std(times)

    console.print("\n[green]Results:[/]")
    console.print(f"  Average: {avg:.2f} ms")
    console.print(f"  Std Dev: {std:.2f} ms")
    console.print(f"  Min: {min(times):.2f} ms")
    console.print(f"  Max: {max(times):.2f} ms")

    stats = engine.get_stats()
    console.print("\n[cyan]Routing Stats:[/]")
    console.print(f"  Logic Ratio: {stats['logic_ratio']:.2%}")
    console.print(f"  Average ID: {stats['average_id']:.3f}")
    console.print(f"  Average Gate: {stats['average_gate']:.3f}")


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
