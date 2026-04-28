"""
Git Reader Module for git-summarize.

Handles reading staged Git diffs, branch information, and recent commits.
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GitDiff:
    """Represents a Git diff for a single file."""

    file_path: str
    old_file: Optional[str]
    new_file: Optional[str]
    diff_text: str
    is_new_file: bool = False
    is_deleted: bool = False
    is_renamed: bool = False


@dataclass
class GitContext:
    """
    Complete Git context for commit message generation.

    Contains all information needed to generate meaningful commit messages.
    """

    # Repository info
    repo_root: str
    branch_name: str
    is_dirty: bool

    # Staged changes
    staged_files: list[str] = field(default_factory=list)
    diffs: list[GitDiff] = field(default_factory=list)
    diff_summary: str = ""

    # Recent history
    recent_commits: list[str] = field(default_factory=list)

    # Statistics
    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0

    @property
    def diff_text(self) -> str:
        """Get combined diff text for all files."""
        return "\n\n".join(d.diff_text for d in self.diffs)

    @property
    def has_changes(self) -> bool:
        """Check if there are staged changes."""
        return len(self.staged_files) > 0


class GitReaderError(Exception):
    """Exception raised when Git operations fail."""

    pass


class GitReader:
    """
    Reads Git repository information and staged changes.

    Provides a clean interface for accessing Git data needed for
    commit message generation.
    """

    def __init__(self, repo_path: Optional[str] = None):
        """
        Initialize Git reader.

        Args:
            repo_path: Path to Git repository (defaults to current directory)
        """
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self._validate_repo()

    def _validate_repo(self) -> None:
        """Validate that the path is a Git repository."""
        try:
            self._run_git("rev-parse", "--git-dir")
        except GitReaderError:
            raise GitReaderError(
                f"Not a Git repository: {self.repo_path}. "
                "Run 'git init' or navigate to a Git repository."
            )

    def _run_git(self, *args: str, cwd: Optional[Path] = None) -> str:
        """
        Run a Git command and return output.

        Args:
            *args: Git command arguments
            cwd: Working directory (defaults to repo_path)

        Returns:
            Command output as string

        Raises:
            GitReaderError: If command fails
        """
        try:
            result = subprocess.run(
                ["git"] + list(args),
                cwd=cwd or self.repo_path,
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
                errors="replace",
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else str(e)
            raise GitReaderError(f"Git command failed: {error_msg}")

    def get_repo_root(self) -> str:
        """Get the absolute path to the repository root."""
        return self._run_git("rev-parse", "--show-toplevel")

    def get_branch_name(self) -> str:
        """Get the current branch name."""
        try:
            return self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        except GitReaderError:
            return "HEAD (detached)"

    def get_staged_files(self) -> list[str]:
        """Get list of staged files."""
        output = self._run_git("diff", "--cached", "--name-only")
        if not output:
            return []
        return output.split("\n")

    def get_staged_diff(self) -> str:
        """Get the full staged diff."""
        return self._run_git("diff", "--cached")

    def get_file_diff(self, file_path: str, context_lines: int = 3) -> str:
        """Get diff for a specific file."""
        return self._run_git("diff", "--cached", f"-U{context_lines}", "--", file_path)

    def get_diff_stats(self) -> tuple[int, int, int]:
        """
        Get diff statistics.

        Returns:
            Tuple of (files_changed, insertions, deletions)
        """
        output = self._run_git("diff", "--cached", "--stat")
        if not output:
            return 0, 0, 0

        # Parse the last line for stats
        lines = output.strip().split("\n")
        if not lines:
            return 0, 0, 0

        # Last line contains summary like: "3 files changed, 15 insertions(+), 5 deletions(-)"
        summary = lines[-1]
        files_changed = insertions = deletions = 0

        if "file" in summary:
            parts = summary.split(",")
            for part in parts:
                part = part.strip()
                if "file" in part and "changed" in part:
                    files_changed = int("".join(c for c in part if c.isdigit()) or 0)
                elif "insertion" in part:
                    insertions = int("".join(c for c in part if c.isdigit()) or 0)
                elif "deletion" in part:
                    deletions = int("".join(c for c in part if c.isdigit()) or 0)

        return files_changed, insertions, deletions

    def get_recent_commits(self, limit: int = 5) -> list[str]:
        """
        Get recent commit messages.

        Args:
            limit: Number of commits to retrieve

        Returns:
            List of commit messages (subject lines)
        """
        output = self._run_git(
            "log",
            f"-{limit}",
            "--format=%s",  # Subject line only
        )
        if not output:
            return []
        return output.split("\n")

    def is_repo_dirty(self) -> bool:
        """Check if working directory has unstaged changes."""
        try:
            output = self._run_git("status", "--porcelain")
            # Filter to only unstaged changes (not starting with space or index marker)
            for line in output.split("\n"):
                if line and not line.startswith(" "):
                    # Check if there are unstaged changes (second character is not space)
                    if len(line) >= 2 and line[1] != " ":
                        return True
            return bool(output.strip())
        except GitReaderError:
            return False

    def parse_diffs(self, context_lines: int = 3) -> list[GitDiff]:
        """
        Parse staged changes into structured GitDiff objects.

        Args:
            context_lines: Number of context lines to include in diff

        Returns:
            List of GitDiff objects for each changed file
        """
        output = self._run_git("diff", "--cached", "--name-status")
        if not output:
            return []

        diffs = []
        for line in output.split("\n"):
            if not line.strip():
                continue

            parts = line.split("\t")
            status = parts[0]
            file_path = parts[1] if len(parts) > 1 else ""
            old_file = parts[2] if len(parts) > 2 else None

            is_new = status.startswith("A")
            is_deleted = status.startswith("D")
            is_renamed = status.startswith("R")

            diff_text = self.get_file_diff(
                file_path if not is_renamed else parts[2],
                context_lines=context_lines
            )
            
            # Truncate extremely large individual file diffs
            if len(diff_text) > 8000:
                diff_text = diff_text[:8000] + "\n\n[... file diff truncated ...]"

            diffs.append(
                GitDiff(
                    file_path=file_path,
                    old_file=old_file if is_renamed else None,
                    new_file=file_path,
                    diff_text=diff_text,
                    is_new_file=is_new,
                    is_deleted=is_deleted,
                    is_renamed=is_renamed,
                )
            )

        return diffs

    def get_context(
        self, 
        include_recent_commits: int = 5,
        context_lines: int = 3
    ) -> GitContext:
        """
        Get complete Git context for commit message generation.

        Args:
            include_recent_commits: Number of recent commits to include
            context_lines: Number of context lines to include in diffs

        Returns:
            GitContext with all repository information
        """
        staged_files = self.get_staged_files()
        diffs = self.parse_diffs(context_lines=context_lines)
        files_changed, insertions, deletions = self.get_diff_stats()

        # Create a summary of changes
        diff_summary_parts = []
        if diffs:
            new_files = [d for d in diffs if d.is_new_file]
            deleted_files = [d for d in diffs if d.is_deleted]
            modified_files = [d for d in diffs if not d.is_new_file and not d.is_deleted]

            if new_files:
                diff_summary_parts.append(f"{len(new_files)} new file(s)")
            if modified_files:
                diff_summary_parts.append(f"{len(modified_files)} modified file(s)")
            if deleted_files:
                diff_summary_parts.append(f"{len(deleted_files)} deleted file(s)")

        diff_summary = ", ".join(diff_summary_parts) if diff_summary_parts else "No changes"

        return GitContext(
            repo_root=self.get_repo_root(),
            branch_name=self.get_branch_name(),
            is_dirty=self.is_repo_dirty(),
            staged_files=staged_files,
            diffs=diffs,
            diff_summary=diff_summary,
            recent_commits=self.get_recent_commits(include_recent_commits),
            files_changed=files_changed,
            insertions=insertions,
            deletions=deletions,
        )
