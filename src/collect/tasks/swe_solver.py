"""SWE-bench agent scaffold: stateless bash + python.

The inspect_evals default SWE-bench agent uses ``bash_session`` + ``text_editor``, but both
carry optional parameters (``bash_session.input``, ``text_editor.file_text``) that are absent
from their JSON-schema ``required`` list — which OpenAI's function-calling rejects
(``invalid_function_parameters``), so the default scaffold cannot run OpenAI models. Plain
``bash`` and ``python`` have strict-compliant schemas that every provider accepts (verified:
gpt-5 runs a full trajectory on this scaffold). Using one uniform toolset across all models
keeps transcripts comparable, which pairwise judging requires.

Trade-off vs the default: no structured editor, so models edit via bash/python (sed, python
file writes) rather than str_replace. That is the price of a single OpenAI-compatible scaffold.
"""

from textwrap import dedent

from inspect_ai.agent import Agent, agent, react
from inspect_ai.tool import bash, python

DEFAULT_TOOL_TIMEOUT = 210


@agent
def swe_bash_python(tool_timeout: int = DEFAULT_TOOL_TIMEOUT) -> Agent:
    prompt = dedent("""
    You are an expert software engineer, and you are interacting with a
    standard Ubuntu machine with bash commands and python tools.
    Your bash session is NOT stateful, so all commands must be self-contained
    (directory changes with cd do not persist between commands).
    You will be given an issue to fix.
    Your objective is to modify the code on the file system to fix the issue.
    The repository code is already checked out to the current working directory.
    You do NOT need to change branches or commit the fix.
    Once you are done, use your submit tool.
    """)
    return react(
        description="Software engineering agent with bash and python tools",
        prompt=prompt,
        tools=[
            bash(timeout=tool_timeout),
            python(timeout=tool_timeout),
        ],
    )
