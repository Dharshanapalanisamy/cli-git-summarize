"""
Prompt Builder Module for git-summarize.

Formats Git diff and metadata into structured LLM prompts
for generating conventional commit messages.
"""

from dataclasses import dataclass
from typing import Optional

from git_summarize.git_reader import GitContext


@dataclass
class PromptComponents:
    """Components of a generated prompt."""

    system_prompt: str
    user_prompt: str


class PromptBuilder:
    """
    Builds prompts for AI commit message generation.

    Creates structured prompts that guide the AI to generate
    high-quality, conventional commit messages.
    """

    # System prompt that defines the AI's role and behavior
    SYSTEM_PROMPT = """You are an expert software engineer specializing in writing clean, descriptive commit messages.

Your task is to analyze Git diffs and generate commit messages that follow the Conventional Commits specification.

## Conventional Commits Format

<type>(<scope>): <description>

[Optional body]

[Optional footer]

### Types
- feat: A new feature
- fix: A bug fix
- docs: Documentation only changes
- style: Changes that don't affect code meaning (formatting, etc.)
- refactor: Code change that neither fixes a bug nor adds a feature
- perf: Performance improvements
- test: Adding or updating tests
- chore: Maintenance tasks, dependencies, build changes
- ci: CI/CD configuration changes
- build: Build system or external dependency changes
- revert: Reverts a previous commit

### Guidelines
1. Use imperative mood in descriptions ("add" not "added")
2. No period at the end of the subject line
3. Keep subject line under 72 characters
4. Be specific and descriptive
5. Reference issue numbers when applicable

## Output Format

Generate exactly {num_suggestions} commit message suggestions.
Each suggestion should be separated by "---" on its own line.

Note: The Git diff provided may be minimized (no context lines) or truncated to save tokens. Focus on the explicit changes shown (+/- lines) and use the summary information provided.

Example output:
```
feat(auth): add OAuth2 authentication support

Implemented OAuth2 flow with support for Google and GitHub providers.
Added token refresh mechanism and secure storage.

Closes #123
---
feat(security): implement OAuth2 login flow

- Add OAuth2 client configuration
- Support multiple identity providers
- Token management and refresh

Fixes #123, #125
---
feat: add OAuth2 authentication

New authentication system using OAuth2 protocol for third-party login.
```

Focus on accuracy, clarity, and following conventional commits standards."""

    # Template for the user prompt
    USER_PROMPT_TEMPLATE = """## Repository Context

Branch: {branch_name}
Recent commits:
{recent_commits}

## Changes Summary

{diff_summary}
Files changed: {files_changed}
Insertions: {insertions}
Deletions: {deletions}

## Staged Diff

{diff_text}

---

Based on the staged changes above, generate {num_suggestions} commit message suggestions following the Conventional Commits specification.

Consider:
1. The primary purpose of the changes
2. Affected components/modules (for scope)
3. Patterns from recent commits (for consistency)
4. Appropriate type based on change nature

Generate {num_suggestions} distinct suggestions with varying levels of detail."""

    def __init__(
        self,
        num_suggestions: int = 3,
        max_diff_length: int = 10000,
        include_recent_commits: bool = True,
    ):
        """
        Initialize the prompt builder.

        Args:
            num_suggestions: Number of commit suggestions to request
            max_diff_length: Maximum characters for diff in prompt
            include_recent_commits: Whether to include recent commit history
        """
        self.num_suggestions = num_suggestions
        self.max_diff_length = max_diff_length
        self.include_recent_commits = include_recent_commits

    def build(self, context: GitContext) -> PromptComponents:
        """
        Build a complete prompt from Git context.

        Args:
            context: Git context with diff and repository information

        Returns:
            PromptComponents with system and user prompts
        """
        system_prompt = self.SYSTEM_PROMPT.format(
            num_suggestions=self.num_suggestions
        )

        user_prompt = self._build_user_prompt(context)

        return PromptComponents(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def _build_user_prompt(self, context: GitContext) -> str:
        """Build the user portion of the prompt."""
        # Format recent commits
        recent_commits_str = self._format_recent_commits(context.recent_commits)

        # Truncate diff if too long
        diff_text = context.diff_text
        if len(diff_text) > self.max_diff_length:
            diff_text = self._truncate_diff(diff_text, self.max_diff_length)

        return self.USER_PROMPT_TEMPLATE.format(
            branch_name=context.branch_name,
            recent_commits=recent_commits_str,
            diff_summary=context.diff_summary,
            files_changed=context.files_changed,
            insertions=context.insertions,
            deletions=context.deletions,
            diff_text=diff_text,
            num_suggestions=self.num_suggestions,
        )

    def _format_recent_commits(self, commits: list[str]) -> str:
        """Format recent commits for the prompt."""
        if not commits or not self.include_recent_commits:
            return "  (none)"

        formatted = []
        for i, commit in enumerate(commits[:5], 1):
            formatted.append(f"  {i}. {commit}")
        return "\n".join(formatted)

    def _truncate_diff(self, diff_text: str, max_length: int) -> str:
        """
        Truncate diff text while preserving important information.

        Strategy:
        1. Keep all file headers (diff --git lines)
        2. Keep all hunk headers (@@ lines)
        3. Truncate context lines within hunks if needed
        """
        if len(diff_text) <= max_length:
            return diff_text

        lines = diff_text.split("\n")
        result_lines = []
        current_length = 0
        truncated = False

        for line in lines:
            line_length = len(line) + 1  # +1 for newline

            # Always include file headers and hunk headers
            if line.startswith("diff --git") or line.startswith("@@"):
                if current_length + line_length <= max_length:
                    result_lines.append(line)
                    current_length += line_length
                else:
                    truncated = True
                    break
            # Include added/removed lines with priority
            elif line.startswith("+") or line.startswith("-"):
                if current_length + line_length <= max_length:
                    result_lines.append(line)
                    current_length += line_length
                else:
                    truncated = True
            # Context lines (space-prefixed) are lowest priority
            else:
                if current_length + line_length <= max_length:
                    result_lines.append(line)
                    current_length += line_length
                else:
                    truncated = True

        result = "\n".join(result_lines)

        if truncated:
            result += "\n\n[... diff truncated for length ...]"

        return result

    def build_minimal_prompt(self, context: GitContext) -> PromptComponents:
        """
        Build a minimal prompt for faster generation.

        Useful when full context is not needed or for quick iterations.

        Args:
            context: Git context with diff information

        Returns:
            PromptComponents with simplified prompts
        """
        system_prompt = """You are a commit message assistant. Generate a single, concise conventional commit message based on the provided diff.

Format: <type>(<scope>): <description>

Be specific and use imperative mood."""

        # Just the essential diff info
        files_changed = ", ".join(context.staged_files[:10])
        if len(context.staged_files) > 10:
            files_changed += f" ... and {len(context.staged_files) - 10} more"

        user_prompt = f"""Files changed: {files_changed}

Diff:
{context.diff_text[:2000]}

Generate one conventional commit message."""

        return PromptComponents(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
