# run format tasks
[group('format')]
format: format-import format-source

# format code import
[group('format')]
format-import:
    uv run ruff check --select=I --fix {{SOURCES}} {{TESTS}} {{WORKFLOW_SCRIPTS}}

# format code source
[group('format')]
format-source:
    uv run ruff format {{SOURCES}} {{TESTS}} {{WORKFLOW_SCRIPTS}}
