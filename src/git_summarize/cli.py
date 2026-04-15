"""
CLI Layer for git-summarize.

Typer-based command-line interface with subcommands and flags.
"""

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from git_summarize import __version__
from git_summarize.config import Config, get_config
from git_summarize.git_ops import GitOps
from git_summarize.git_reader import GitReader, GitReaderError
from git_summarize.prompt_builder import PromptBuilder
from git_summarize.parser import ResponseParser
from git_summarize.providers import (
    AIProvider,
    ClaudeProvider,
    GeminiProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderError,
    ProviderRegistry,
)
from git_summarize.ui import UI, UserAction

# Create Typer app
app = typer.Typer(
    name="gcms",
    help="AI-powered Git commit message generator",
    add_completion=False,
)

console = Console()


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        typer.echo(f"gcm version {__version__}")
        raise typer.Exit()


def validate_provider(ctx: typer.Context, provider: str) -> str:
    """Validate provider name."""
    provider = provider.lower()
    available = ProviderRegistry.list_providers()
    if provider not in available:
        raise typer.BadParameter(
            f"Invalid provider '{provider}'. Available: {', '.join(available)}"
        )
    return provider


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """
    Main callback for gcm.
    
    If no subcommand is provided, runs the smart automated workflow.
    """
    if ctx.invoked_subcommand is None:
        # Load configuration
        config = get_config()
        ui = UI(console)

        # Step 1: Check if configured
        if not config.is_configured():
            asyncio.run(run_initial_setup(config, ui))
            return

        # Step 2: Run the automated workflow
        # (Stage all -> Generate -> Commit -> Push)
        console.print("[bold cyan]🚀 Starting automated Git workflow...[/bold cyan]")
        
        # We want to force stage_all and push for the default workflow
        config.no_add = False
        config.push = True
        
        asyncio.run(
            run_generation_flow(
                config=config,
                ui=ui,
            )
        )


@app.command()
def generate(
    provider: str = typer.Option(
        None,
        "--provider",
        "-p",
        callback=validate_provider,
        help="AI provider to use (claude, openai, ollama, gemini)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name (overrides provider default)",
    ),
    num_suggestions: int = typer.Option(
        3,
        "--num-suggestions",
        "-n",
        min=1,
        max=10,
        help="Number of suggestions to generate",
    ),
    auto: bool = typer.Option(
        False,
        "--auto",
        "-a",
        help="Auto-select first suggestion without interaction",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Preview suggestions without committing",
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Apply first suggestion directly",
    ),
    push: bool = typer.Option(
        False,
        "--push",
        help="Push after commit (asks for branch selection)",
    ),
    no_add: bool = typer.Option(
        False,
        "--no-add",
        help="Skip git add (only stage specific files)",
    ),
    repo_path: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Path to Git repository (default: current directory)",
    ),
) -> int:
    """
    Generate and apply a commit message from staged changes.

    Reads staged Git diffs and uses AI to suggest conventional
    commit messages. Select from suggestions or edit before committing.
    """
    # Load configuration
    config = get_config()

    # Override config with CLI options
    if provider:
        config.provider = provider
    if model:
        config.model = model
    if num_suggestions:
        config.num_suggestions = num_suggestions
    if auto:
        config.auto = auto
    if preview:
        config.preview = preview
    if apply:
        config.apply = apply
    if push:
        config.push = push
    if no_add:
        config.no_add = no_add

    # Initialize UI
    ui = UI(console)

    # Run the main generation flow
    return asyncio.run(
        run_generation_flow(
            config=config,
            ui=ui,
            repo_path=repo_path,
        )
    )


@app.command()
def config_cmd(
    show: bool = typer.Option(False, "--show", help="Show current config"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Set default provider"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Set default model"),
) -> int:
    """
    Manage git-summarize configuration.

    View or set configuration options.
    """
    config = get_config()

    if show:
        console.print("[bold]Current Configuration:[/bold]")
        console.print(f"  Provider: [cyan]{config.provider}[/cyan]")
        console.print(f"  Model: [cyan]{config.get_model(config.provider)}[/cyan]")
        console.print(f"  Suggestions: [cyan]{config.num_suggestions}[/cyan]")
        console.print(f"  Ollama Host: [cyan]{config.get_ollama_host()}[/cyan]")

        # Check API keys
        if config.get_api_key("claude"):
            console.print("  Anthropic API Key: [green]✓ set[/green]")
        else:
            console.print("  Anthropic API Key: [red]✗ not set[/red]")

        if config.get_api_key("openai"):
            console.print("  OpenAI API Key: [green]✓ set[/green]")
        else:
            console.print("  OpenAI API Key: [red]✗ not set[/red]")

        return 0

    # Set configuration values
    if provider:
        config.provider = provider
        console.print(f"[green]✓ Set default provider: {provider}[/green]")

    if model:
        config.model = model
        console.print(f"[green]✓ Set default model: {model}[/green]")

    if not provider and not model:
        # Show help
        console.print("[yellow]No options specified. Use --show to view config.[/yellow]")
        console.print("\n[dim]Usage:[/dim]")
        console.print("  gcm config --show")
        console.print("  gcm config --provider claude")
        console.print("  gcm config --model gpt-4")

    return 0


@app.command()
def providers() -> int:
    """
    List available AI providers.

    Shows all registered AI providers and their status.
    """
    console.print("[bold]Available AI Providers:[/bold]\n")

    providers_info = {
        "claude": {
            "name": "Anthropic Claude",
            "default_model": "claude-3-sonnet-20240229",
            "requires_key": True,
            "env_var": "ANTHROPIC_API_KEY",
        },
        "openai": {
            "name": "OpenAI GPT",
            "default_model": "gpt-4-turbo-preview",
            "requires_key": True,
            "env_var": "OPENAI_API_KEY",
        },
        "ollama": {
            "name": "Ollama (Local)",
            "default_model": "llama2",
            "requires_key": False,
            "env_var": None,
        },
    }

    config = get_config()

    for provider_name in ProviderRegistry.list_providers():
        info = providers_info.get(provider_name, {})
        name = info.get("name", provider_name.title())
        default_model = info.get("default_model", "unknown")
        requires_key = info.get("requires_key", False)
        env_var = info.get("env_var")

        console.print(f"[bold cyan]{provider_name}[/bold cyan]")
        console.print(f"  Name: {name}")
        console.print(f"  Default Model: {default_model}")

        if requires_key and env_var:
            key_set = bool(config.get_api_key(provider_name))
            status = "[green]✓ set[/green]" if key_set else "[red]✗ not set[/red]"
            console.print(f"  API Key ({env_var}): {status}")
        else:
            console.print("  API Key: [dim]Not required[/dim]")

        console.print()

    return 0


async def run_generation_flow(
    config: Config,
    ui: UI,
    repo_path: Optional[str] = None,
) -> int:
    """
    Run the main commit message generation flow.

    Args:
        config: Configuration object
        ui: UI instance
        repo_path: Optional repository path

    Returns:
        Exit code (0 for success)
    """
    try:
        # Step 1: Get AI provider and check availability first
        # This prevents unnecessary staging if the AI service is down
        provider_name = config.provider
        api_key = config.get_api_key(provider_name)
        model = config.get_model(provider_name)

        provider = create_provider(provider_name, api_key, model, config)

        # Validate configuration (API keys, etc.)
        is_valid, error = provider.validate()
        if not is_valid:
            ui.show_error(error, f"{provider.name} Error")
            return 1

        # Check if the provider is actually reachable
        with ui.show_spinner(f"Connecting to {provider.name}..."):
            if not await provider.check_availability():
                ui.show_provider_error(
                    provider_name, 
                    f"Could not connect to {provider.name}. The service might be down or unreachable."
                )
                return 1

        # Initialize Git operations
        git = GitOps(repo_path)

        # Step 2: Stage changes if not skipped
        if not config.no_add:
            with ui.show_spinner("Staging changes..."):
                if not git.stage_all():
                    ui.show_error("Failed to stage changes.", "Git Error")
                    return 1

        # Step 3: Read Git context
        with ui.show_spinner("Reading Git repository..."):
            try:
                reader = GitReader(repo_path)
                context = reader.get_context()
            except GitReaderError as e:
                ui.show_error(str(e), "Git Error")
                return 1

        # Check for staged changes
        if not context.has_changes:
            ui.show_no_changes()
            return 1

        # Show context summary
        ui.show_context(
            branch=context.branch_name,
            files_changed=context.files_changed,
            insertions=context.insertions,
            deletions=context.deletions,
        )

        # Step 4: Build prompt
        prompt_builder = PromptBuilder(
            num_suggestions=config.num_suggestions,
        )
        prompt = prompt_builder.build(context)

        # Step 5: Generate suggestions

        # Step 5: Generate suggestions
        try:
            with ui.show_spinner(f"Generating commit messages with {provider.name}..."):
                response = await provider.generate(
                    type(
                        "GenRequest",
                        (),
                        {
                            "prompt": prompt.user_prompt,
                            "system_prompt": prompt.system_prompt,
                            "max_tokens": 1024,
                            "temperature": 0.3,
                        },
                    )()
                )
        except ProviderError as e:
            ui.show_provider_error(provider_name, str(e))
            return 1

        # Step 6: Parse response
        parser = ResponseParser()
        result = parser.parse(response.text)

        if not result.has_suggestions:
            ui.show_error(
                "Could not parse any commit messages from AI response.",
                "Parse Error",
            )
            console.print(f"\n[dim]Raw response:[/dim]\n{response.text[:500]}...")
            return 1

        if result.parse_errors:
            for error in result.parse_errors:
                ui.show_warning(error, "Parse Warning")

        # Step 7: Display suggestions
        ui.show_suggestions(
            suggestions=result.suggestions,
            provider=provider_name,
            model=model,
        )

        # Step 8: Handle user interaction
        selected_message = await handle_user_interaction(
            ui=ui,
            suggestions=result.suggestions,
            config=config,
            provider=provider,
            prompt_builder=prompt_builder,
            context=context,
        )

        if selected_message is None:
            console.print("[yellow]Commit cancelled.[/yellow]")
            return 0

        # Step 9: Commit
        if config.preview:
            ui.prompt_preview(selected_message)
            return 0

        if config.apply or config.auto:
            success, commit_output = git.commit(selected_message)
        else:
            if ui.prompt_confirm_commit(selected_message):
                success, commit_output = git.commit(selected_message)
            else:
                console.print("[yellow]Commit cancelled.[/yellow]")
                return 0

        if not success:
            ui.show_error(f"Failed to create commit: {commit_output}", "Commit Error")
            return 1

        ui.show_commit_success(selected_message)

        # Step 10: Push if requested
        if config.push:
            push_result = await handle_push(git, ui, context.branch_name)
            if push_result != 0:
                return push_result

        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled.[/yellow]")
        return 0
    except Exception as e:
        ui.show_error(f"Unexpected error: {str(e)}", "Error")
        return 1


def create_provider(
    provider_name: str,
    api_key: Optional[str],
    model: Optional[str],
    config: Config,
) -> AIProvider:
    """Create an AI provider instance."""
    if provider_name == "claude":
        return ClaudeProvider(api_key=api_key, model=model)
    elif provider_name == "openai":
        return OpenAIProvider(api_key=api_key, model=model)
    elif provider_name == "ollama":
        return OllamaProvider(
            api_key=api_key,
            model=model,
            host=config.get_ollama_host(),
        )
    elif provider_name == "gemini":
        return GeminiProvider(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


async def handle_user_interaction(
    ui: UI,
    suggestions: list,
    config: Config,
    provider: AIProvider,
    prompt_builder: PromptBuilder,
    context,
) -> Optional[str]:
    """
    Handle user interaction for selecting/editing suggestions.

    Returns:
        Selected commit message, or None if cancelled
    """
    if config.auto:
        return str(suggestions[0])

    max_iterations = 10
    current_suggestions = suggestions

    for _ in range(max_iterations):
        action, index = ui.prompt_selection(
            num_suggestions=len(current_suggestions),
            allow_regenerate=True,
        )

        if action == UserAction.QUIT:
            return None

        elif action == UserAction.SELECT:
            selected = current_suggestions[index]
            return str(selected)

        elif action == UserAction.EDIT:
            selected = current_suggestions[index]
            edited = ui.prompt_edit(str(selected))
            if edited:
                return edited
            # If edit cancelled, continue loop

        elif action == UserAction.REGENERATE:
            with ui.show_spinner("Regenerating suggestions..."):
                try:
                    response = await provider.generate(
                        generation_request=type(
                            "GenRequest",
                            (),
                            {
                                "prompt": prompt_builder.user_prompt,
                                "system_prompt": prompt_builder.system_prompt,
                                "max_tokens": 1024,
                                "temperature": 0.5,  # Slightly higher for variety
                            },
                        )()
                    )
                    parser = ResponseParser()
                    result = parser.parse(response.text)
                    if result.has_suggestions:
                        current_suggestions = result.suggestions
                        ui.show_suggestions(
                            suggestions=current_suggestions,
                            provider=provider.name.lower(),
                            model=provider.model,
                        )
                    else:
                        ui.show_warning("Could not generate new suggestions. Try again.")
                except ProviderError as e:
                    ui.show_warning(f"Regeneration failed: {e}")

    ui.show_error("Maximum iterations reached.", "Error")
    return None


async def run_initial_setup(config: Config, ui: UI) -> None:
    """Run interactive onboarding for first-time users."""
    ui.show_onboarding_welcome()
    
    # 1. Select Provider
    providers = ProviderRegistry.list_providers()
    selected_provider = ui.prompt_provider_selection(providers)
    
    # 2. Get API Key if needed
    api_key = ""
    selected_model = None
    if selected_provider != "ollama":
        api_key = ui.prompt_api_key(selected_provider)
    else:
        # Try to list models for Ollama
        try:
            from git_summarize.providers import ProviderRegistry
            provider_cls = ProviderRegistry.get("ollama")
            # Create a temporary provider instance to list models
            # We use config for host settings if available
            provider_inst = provider_cls(host=config.get_ollama_host())
            
            with ui.show_spinner("Fetching local Ollama models..."):
                models = await provider_inst.list_models()
            
            if models:
                selected_model = ui.prompt_ollama_model_selection(models)
            else:
                ui.show_warning("No local Ollama models found. Using default: llama2")
        except Exception as e:
            ui.show_warning(f"Could not connect to Ollama server: {e}")

    # 3. Save Configuration
    with ui.show_spinner("Saving configuration..."):
        config.save_to_env(selected_provider, api_key, model=selected_model)
    
    ui.show_setup_success(selected_provider)
    
    # Ask if they want to run the tool now if there are changes
    if typer.confirm("\nWould you like to run gcms on your current changes now?", default=True):
        # Refresh config with new values
        new_config = get_config()
        new_config.push = True
        await run_generation_flow(new_config, ui)


async def handle_push(git: GitOps, ui: UI, current_branch: str) -> int:
    """
    Handle push workflow with branch selection.

    Args:
        git: GitOps instance
        ui: UI instance
        current_branch: Current branch name

    Returns:
        Exit code (0 for success)
    """
    try:
        # Check if repository has a remote
        if not git.has_remote():
            ui.show_error(
                "No remote configured. Add a remote with:\n"
                f"  git remote add origin <repository-url>",
                "Push Error",
            )
            return 1

        # Get remote branches
        remote_branches = git.get_remote_branches()

        # Select branch to push to
        if remote_branches:
            # Show branch selection UI
            selected_branch = ui.prompt_branch_selection(
                current_branch=current_branch,
                remote_branches=remote_branches,
            )

            if selected_branch is None:
                console.print("[yellow]Push cancelled.[/yellow]")
                return 0

            # Check if we need to set upstream
            set_upstream = current_branch not in remote_branches
        else:
            # No remote branches, push to current branch with upstream
            selected_branch = current_branch
            set_upstream = True

        # Confirm push
        if not ui.prompt_confirm_push(selected_branch, set_upstream):
            console.print("[yellow]Push cancelled.[/yellow]")
            return 0

        # Execute push
        with ui.show_spinner(f"Pushing to {selected_branch}..."):
            success, output = git.push(
                branch=selected_branch,
                set_upstream=set_upstream,
            )

        if not success:
            ui.show_error(f"Push failed: {output}", "Push Error")
            return 1

        console.print(f"[green]✓[/green] Pushed to {selected_branch}")
        return 0

    except Exception as e:
        ui.show_error(f"Push error: {str(e)}", "Error")
        return 1


def commit_changes(message: str) -> bool:
    """
    Create a Git commit with the given message.

    Args:
        message: Commit message

    Returns:
        True if successful, False otherwise
    """
    try:
        # Build commit command
        cmd = ["git", "commit", "-m", message]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        return True

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Git commit failed:[/red] {e.stderr}")
        return False
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        return False


# Entry point
def cli() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    cli()
