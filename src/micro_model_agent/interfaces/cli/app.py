"""Command-line interface entrypoint.

Typer turns these Python functions into terminal commands. The CLI mostly wires
domain/application services together, prints a short result, and exits with a
nonzero status when a command fails.
"""

from __future__ import annotations

import typer

from micro_model_agent.interfaces.cli.commands.dataset import register_dataset_commands
from micro_model_agent.interfaces.cli.commands.eval import register_eval_commands
from micro_model_agent.interfaces.cli.commands.loop import register_loop_command
from micro_model_agent.interfaces.cli.commands.mcp import register_mcp_command
from micro_model_agent.interfaces.cli.commands.promote import register_promote_commands
from micro_model_agent.interfaces.cli.commands.repo import register_repo_commands
from micro_model_agent.interfaces.cli.commands.train import register_train_commands

app = typer.Typer(help="MicroModelAgent CLI.")
dataset_app = typer.Typer(help="Dataset generation, validation, and export commands.")
train_app = typer.Typer(help="Local training commands.")
eval_app = typer.Typer(help="Evaluation commands.")
promote_app = typer.Typer(help="Promotion gate commands.")

# Sub-apps create command groups such as `micro-agent dataset validate`.
app.add_typer(dataset_app, name="dataset")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(promote_app, name="promote")
register_repo_commands(app)
register_loop_command(app)
register_mcp_command(app)
register_dataset_commands(dataset_app)
register_train_commands(train_app)
register_eval_commands(eval_app)
register_promote_commands(promote_app)


if __name__ == "__main__":
    app()
