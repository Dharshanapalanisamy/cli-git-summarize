"""
Interactive UI Module for git-summarize.

Rich-based terminal UI for displaying suggestions and handling user interaction.
"""

from enum import Enum
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text
from rich.style import Style

from git_summarize.parser import CommitSuggestion


class UserAction(Enum):
    """Possible user actions after viewing suggestions."""

    SELECT = "select"  # Select a suggestion
    EDIT = "edit"  # Edit the selected suggestion
    REJECT = "reject"  # Reject all and skip commit
    REGENERATE = "regenerate"  # Generate new suggestions
    QUIT = "quit"  # Exit without committing


class UI:
    """
    Interactive terminal UI for git-summarize.

    Provides spinners, colored output, suggestion selection,
    and edit/reject flow.
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the UI.

        Args:
            console: Rich console instance (creates default if None)
        """
        self.console = console or Console()

    def show_spinner(self, message: str = "Generating commit messages...") -> Live:
        """
        Create a spinner for long-running operations.

        Args:
            message: Message to display with spinner

        Returns:
            Live context manager for the spinner
        """
        spinner = Spinner("dots", text=f"[cyan]{message}[/cyan]")
        return Live(spinner, console=self.console, transient=True)

    def show_error(self, message: str, title: str = "Error") -> None:
        """
        Display an error message.

        Args:
            message: Error message to display
            title: Error title
        """
        panel = Panel(
            f"[red]{message}[/red]",
            title=f"[bold red]❌ {title}[/bold red]",
            border_style="red",
        )
        self.console.print(panel)

    def show_warning(self, message: str, title: str = "Warning") -> None:
        """
        Display a warning message.

        Args:
            message: Warning message to display
            title: Warning title
        """
        panel = Panel(
            f"[yellow]{message}[/yellow]",
            title=f"[bold yellow]⚠️  {title}[/bold yellow]",
            border_style="yellow",
        )
        self.console.print(panel)

    def show_success(self, message: str, title: str = "Success") -> None:
        """
        Display a success message.

        Args:
            message: Success message to display
            title: Success title
        """
        panel = Panel(
            f"[green]{message}[/green]",
            title=f"[bold green]✅ {title}[/bold green]",
            border_style="green",
        )
        self.console.print(panel)

    def show_info(self, message: str, title: str = "Info") -> None:
        """
        Display an info message.

        Args:
            message: Info message to display
            title: Info title
        """
        panel = Panel(
            f"[blue]{message}[/blue]",
            title=f"[bold blue]ℹ️  {title}[/bold blue]",
            border_style="blue",
        )
        self.console.print(panel)

    def show_context(self, branch: str, files_changed: int, insertions: int, deletions: int) -> None:
        """
        Display Git context summary.

        Args:
            branch: Current branch name
            files_changed: Number of files changed
            insertions: Lines inserted
            deletions: Lines deleted
        """
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("label", style="dim")
        table.add_column("value")

        table.add_row("Branch:", f"[cyan]{branch}[/cyan]")
        table.add_row("Changes:", f"{files_changed} file(s), +{insertions}/-{deletions}")

        self.console.print(table)
        self.console.print()

    def show_suggestions(
        self,
        suggestions: list[CommitSuggestion],
        provider: str,
        model: str,
    ) -> None:
        """
        Display commit message suggestions.

        Args:
            suggestions: List of commit suggestions
            provider: AI provider name
            model: Model name used
        """
        self.console.print()
        self.console.print(
            f"[bold green]✨ Generated {len(suggestions)} commit message suggestions[/bold green]"
            f" [dim](via {provider}/{model})[/dim]"
        )
        self.console.print()

        for i, suggestion in enumerate(suggestions, 1):
            self._display_suggestion(suggestion, i)
            self.console.print()

    def _display_suggestion(self, suggestion: CommitSuggestion, index: int) -> None:
        """Display a single suggestion with formatting."""
        # Create the header
        header = Text()
        header.append(f"[{index}] ", style="bold cyan")

        # Highlight the type
        if suggestion.commit_type:
            header.append(f"{suggestion.commit_type}", style="bold green")
            if suggestion.scope:
                header.append("(", style="dim")
                header.append(suggestion.scope, style="green")
                header.append(")", style="dim")
            header.append(": ", style="dim")

        # Add description
        desc_start = len(f"{suggestion.commit_type}({suggestion.scope}): ") if suggestion.commit_type else 0
        header.append(suggestion.subject[desc_start:], style="white")

        self.console.print(header)

        # Show body if present
        if suggestion.body:
            self.console.print(f"    [dim]{suggestion.body}[/dim]")

        # Show footer if present
        if suggestion.footer:
            self.console.print(f"    [dim]{suggestion.footer}[/dim]")

        # Show validation warnings
        if not suggestion.is_valid and suggestion.validation_errors:
            errors = ", ".join(suggestion.validation_errors)
            self.console.print(f"    [yellow]⚠️  {errors}[/yellow]")

    def prompt_selection(
        self,
        num_suggestions: int,
        allow_edit: bool = True,
        allow_regenerate: bool = False,
    ) -> tuple[UserAction, int]:
        """
        Prompt user to select a suggestion.

        Args:
            num_suggestions: Number of available suggestions
            allow_edit: Whether to allow editing
            allow_regenerate: Whether to allow regeneration

        Returns:
            Tuple of (action, selected_index)
        """
        options = [str(i) for i in range(1, num_suggestions + 1)]
        options.extend(["e", "r", "q"])

        prompt_text = (
            f"[bold]Select[/bold] (1-{num_suggestions})"
            f" | [bold]e[/bold]dit"
            f" | [bold]r[/bold]egenerate"
            f" | [bold]q[/bold]uit"
        )

        while True:
            self.console.print()
            choice = Prompt.ask(
                Text.from_markup(prompt_text),
                choices=options,
                default="1",
                show_choices=False,
            )

            if choice == "e" and allow_edit:
                return UserAction.EDIT, 0
            elif choice == "r" and allow_regenerate:
                return UserAction.REGENERATE, 0
            elif choice == "q":
                return UserAction.QUIT, 0
            else:
                try:
                    index = int(choice) - 1
                    if 0 <= index < num_suggestions:
                        return UserAction.SELECT, index
                except ValueError:
                    pass

            self.console.print("[red]Invalid choice. Please try again.[/red]")

    def prompt_edit(self, original: str) -> Optional[str]:
        """
        Prompt user to edit a commit message.

        Args:
            original: Original commit message

        Returns:
            Edited message, or None if cancelled
        """
        self.console.print()
        self.console.print("[bold]Edit commit message:[/bold]")
        self.console.print("[dim](Press Enter twice to finish, or empty to cancel)[/dim]")
        self.console.print()

        lines = []
        while True:
            try:
                line = Prompt.ask(f"  [{len(lines) + 1}]")
                if line == "" and lines:
                    break
                elif line == "" and not lines:
                    self.console.print("[yellow]Edit cancelled.[/yellow]")
                    return None
                lines.append(line)
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Edit cancelled.[/yellow]")
                return None

        edited = "\n".join(lines)
        if not edited.strip():
            return None

        return edited.strip()

    def prompt_confirm_commit(self, message: str) -> bool:
        """
        Confirm before committing.

        Args:
            message: Commit message to confirm

        Returns:
            True if user confirms, False otherwise
        """
        self.console.print()
        self.console.print(Panel(
            f"[white]{message}[/white]",
            title="[bold]Commit Message[/bold]",
            border_style="green",
        ))

        return Confirm.ask(
            "[bold]Commit with this message?[/bold]",
            default=True,
        )

    def prompt_preview(self, message: str) -> None:
        """
        Show a preview of the commit message.

        Args:
            message: Commit message to preview
        """
        self.console.print()
        self.console.print("[bold blue]📋 Preview:[/bold blue]")
        self.console.print(Panel(
            f"[white]{message}[/white]",
            title="[bold]Commit Message[/bold]",
            border_style="blue",
        ))
        self.console.print()
        self.console.print("[dim]Run with --apply to commit, or --auto to skip this prompt.[/dim]")

    def show_commit_success(self, message: str) -> None:
        """
        Show success after committing.

        Args:
            message: The commit message used
        """
        # Truncate for display
        display_msg = message.split("\n")[0]
        if len(display_msg) > 50:
            display_msg = display_msg[:47] + "..."

        self.console.print()
        self.console.print(f"[green]✅ Committed:[/green] [dim]{display_msg}[/dim]")

    def show_no_changes(self) -> None:
        """Show message when no staged changes found."""
        self.console.print()
        self.show_warning(
            "No staged changes found. Use 'git add' to stage files first.",
            "No Changes",
        )

    def show_provider_error(self, provider: str, error: str) -> None:
        """
        Show error when provider fails with relevant tips.

        Args:
            provider: Provider name
            error: Error message
        """
        tips = [
            "Verify network connectivity",
            "Check for typos in your configuration"
        ]
        
        provider_lower = provider.lower()
        if provider_lower == "ollama":
            tips.insert(0, "Make sure the Ollama server is running (run [bold]ollama serve[/bold])")
            tips.insert(1, "Check if the Ollama desktop app is open")
        else:
            tips.insert(0, "Check your API key is set correctly in your [bold].env[/bold] file")
            tips.append(f"Try a different provider using [cyan]--provider ollama[/cyan] or [cyan]--provider gemini[/cyan]")

        tips_text = "\n".join([f"  • {tip}" for tip in tips])
        
        self.show_error(
            f"{error}\n\n"
            f"[dim]Provider: {provider}[/dim]\n\n"
            f"[yellow]Tips:[/yellow]\n{tips_text}",
            f"{provider} Error",
        )

    def prompt_branch_selection(
        self,
        current_branch: str,
        remote_branches: list[str],
    ) -> Optional[str]:
        """
        Prompt user to select a branch to push to.

        Args:
            current_branch: Current local branch name
            remote_branches: List of remote branch names

        Returns:
            Selected branch name, or None if cancelled
        """
        self.console.print()
        self.console.print("[bold]Select branch to push to:[/bold]")
        self.console.print()

        # Build options table
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Option", style="cyan")
        table.add_column("Branch", style="white")
        table.add_column("Info", style="dim")

        # Current branch (recommended)
        table.add_row(
            "[1]",
            f"[green]{current_branch}[/green]",
            "[dim](current branch - recommended)[/dim]",
        )

        # Other remote branches
        other_branches = [b for b in remote_branches if b != current_branch]
        for i, branch in enumerate(other_branches[:5], 2):  # Limit to 5
            table.add_row(f"[{i}]", branch, "")

        # New branch option
        new_option = len(other_branches) + 2
        table.add_row(f"[{new_option}]", "[yellow]<new branch>[/yellow]", "")

        # Cancel
        table.add_row(f"[q]", "[red]Cancel[/red]", "")

        self.console.print(table)
        self.console.print()

        options = [str(i) for i in range(1, new_option + 1)] + ["q"]

        while True:
            choice = Prompt.ask(
                "Enter choice",
                choices=options,
                default="1",
                show_choices=False,
            )

            if choice == "q":
                return None
            elif choice == "1":
                return current_branch
            elif choice.isdigit() and 2 <= int(choice) <= len(other_branches) + 1:
                return other_branches[int(choice) - 2]
            elif choice == str(new_option):
                # Create new branch
                new_branch = Prompt.ask("[bold]Enter new branch name[/bold]")
                if new_branch.strip():
                    return new_branch.strip()
                continue
            else:
                self.console.print("[red]Invalid choice. Please try again.[/red]")

    def prompt_confirm_push(self, branch: str, set_upstream: bool) -> bool:
        """
        Confirm before pushing.

        Args:
            branch: Branch name to push to
            set_upstream: Whether upstream will be set

        Returns:
            True if user confirms, False otherwise
        """
        self.console.print()
        push_msg = f"Push to [cyan]{branch}[/cyan]"
        if set_upstream:
            push_msg += " [dim](set as upstream)[/dim]"

        return Confirm.ask(
            f"[bold]{push_msg}?[/bold]",
            default=True,
        )

    def clear(self) -> None:
        """Clear the console."""
        self.console.clear()

    def show_onboarding_welcome(self) -> None:
        """Display a welcome message for first-time setup."""
        self.console.print()
        self.console.print(Panel(
            "[bold green]Welcome to git-summarize! 🤖[/bold green]\n\n"
            "It looks like this is your first time using the tool.\n"
            "I'll help you set up an AI provider so you can start generating "
            "standardized commit messages automatically.",
            title="[bold]Initial Setup[/bold]",
            border_style="green",
        ))
        self.console.print()

    def prompt_provider_selection(self, providers: list[str], default_provider: Optional[str] = None) -> str:
        """
        Prompt user to select an AI provider.

        Args:
            providers: List of available provider names
            default_provider: Optional provider to set as default

        Returns:
            Selected provider name
        """
        self.console.print("[bold]Select an AI provider:[/bold]")
        
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Option", style="cyan")
        table.add_column("Provider", style="white")
        table.add_column("Description", style="dim")

        descriptions = {
            "gemini": "Google Gemini (Free tier available)",
            "claude": "Anthropic Claude (High quality)",
            "openai": "OpenAI GPT-4 (Popular)",
            "ollama": "Local AI (No API key needed)",
        }

        default_index = "1"
        for i, provider in enumerate(providers, 1):
            desc = descriptions.get(provider, "")
            table.add_row(f"[{i}]", provider.title(), desc)
            if default_provider and provider == default_provider:
                default_index = str(i)

        self.console.print(table)
        self.console.print()

        options = [str(i) for i in range(1, len(providers) + 1)]
        choice = Prompt.ask(
            "Enter choice",
            choices=options,
            default=default_index,
            show_choices=False,
        )

        return providers[int(choice) - 1]

    def prompt_api_key(self, provider: str) -> str:
        """
        Prompt user for an API key.

        Args:
            provider: Provider name

        Returns:
            Entered API key
        """
        self.console.print()
        self.console.print(f"[bold]Enter your {provider.title()} API key:[/bold]")
        if provider == "gemini":
            self.console.print("[dim]Get a free key at: https://aistudio.google.com/app/apikey[/dim]")
        
        return Prompt.ask("API Key", password=False)

    def prompt_ollama_model_selection(self, models: list[str]) -> str:
        """
        Prompt user to select a model from local Ollama installation.

        Args:
            models: List of available model names

        Returns:
            Selected model name
        """
        self.console.print("[bold]Select an Ollama model:[/bold]")
        
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Option", style="cyan")
        table.add_column("Model", style="white")

        for i, model in enumerate(models, 1):
            table.add_row(f"[{i}]", model)

        self.console.print(table)
        self.console.print()

        options = [str(i) for i in range(1, len(models) + 1)]
        choice = Prompt.ask(
            "Enter choice",
            choices=options,
            default="1",
            show_choices=False,
        )

        return models[int(choice) - 1]

    def show_setup_success(self, provider: str) -> None:
        """Show success message after setup."""
        self.console.print()
        self.show_success(
            f"Configuration saved! git-summarize is now set up to use [bold cyan]{provider.title()}[/bold cyan].\n"
            "From now on, just run [bold]gcms[/bold] to automate your Git workflow.",
            "Setup Complete"
        )
        self.console.print()
