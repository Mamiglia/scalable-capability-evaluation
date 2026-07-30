import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageUser,
    ChatMessageSystem,
    ChatMessageTool,
    ContentText,
    ContentReasoning,
    ContentImage,
    ContentAudio,
    ContentVideo,
    ModelOutput,
)
from inspect_ai.tool import ToolCall, ToolCallError
from inspect_ai.scorer import Score

from util import format_message, extract_transcript, extract_score


class TestFormatMessage:
    """Unit tests for format_message in util.py."""

    def test_format_message_string_content(self):
        """Test formatting when msg.content is a plain string."""
        msg = ChatMessageAssistant(content="Simple string response")
        assert format_message(msg) == "Simple string response"

    def test_format_message_text_content_list(self):
        """Test formatting when msg.content is a list containing text content."""
        msg = ChatMessageAssistant(content=[ContentText(type="text", text="First line")])
        assert format_message(msg) == "First line"

    def test_format_message_reasoning_content_list(self):
        """Test formatting when msg.content is a list containing reasoning content."""
        msg = ChatMessageAssistant(
            content=[ContentReasoning(type="reasoning", reasoning="Thinking step by step")]
        )
        result = format_message(msg)
        assert "<THOUGHT_START>" in result
        assert "<THOUGHT_END>" in result
        assert "Thinking step by step" in result

    def test_format_message_mixed_content_list(self):
        """Test formatting when msg.content contains both reasoning and text."""
        msg = ChatMessageAssistant(
            content=[
                ContentReasoning(type="reasoning", reasoning="Step 1"),
                ContentText(type="text", text="Final Answer"),
            ]
        )
        result = format_message(msg)
        assert "<THOUGHT_START>" in result
        assert "Step 1" in result
        assert "Final Answer" in result
        assert result.endswith("Final Answer")

    def test_format_message_tool_calls(self):
        """Test formatting assistant message with tool calls."""
        msg = ChatMessageAssistant(
            content="Running bash script",
            tool_calls=[ToolCall(id="tc1", function="bash", arguments={"cmd": "ls -l"})],
        )
        result = format_message(msg)
        assert "Running bash script" in result
        assert "[Tool Call: bash({'cmd': 'ls -l'})]" in result

    def test_format_message_tool_response(self):
        """Test formatting tool response message (ChatMessageTool)."""
        msg = ChatMessageTool(
            content="file1.txt\nfile2.txt",
            tool_call_id="tc1",
            function="bash",
            error=None,
        )
        result = format_message(msg)
        assert "[Tool Output (bash)]: file1.txt\nfile2.txt" in result

    def test_format_message_tool_response_with_error(self):
        """Test formatting tool response message containing an error."""
        msg = ChatMessageTool(
            content="Permission denied",
            tool_call_id="tc1",
            function="bash",
            error=ToolCallError(type="generic", message="EACCES"),
        )
        result = format_message(msg)
        assert "[Tool Output (bash) (Error: EACCES)]: Permission denied" in result

    def test_format_message_list_transcript(self):
        """Test formatting a sequence (list) of chat messages."""
        msgs = [
            ChatMessageUser(content="How many files are here?"),
            ChatMessageAssistant(
                content="Checking files...",
                tool_calls=[ToolCall(id="tc1", function="bash", arguments={"cmd": "ls"})],
            ),
            ChatMessageTool(content="a.py\nb.py", tool_call_id="tc1", function="bash"),
            ChatMessageAssistant(content="There are 2 files."),
        ]
        result = format_message(msgs)
        assert "[User]: How many files are here?" in result
        assert "[Assistant]: Checking files...\n[Tool Call: bash({'cmd': 'ls'})]" in result
        assert "[Tool]: [Tool Output (bash)]: a.py\nb.py" in result
        assert "[Assistant]: There are 2 files." in result

    def test_format_message_rich_content_types(self):
        """Test formatting messages with rich media content (image, etc.)."""
        msg = ChatMessageAssistant(
            content=[
                ContentText(type="text", text="Here is the plot:"),
                ContentImage(image="data:image/png;base64,..."),
            ]
        )
        result = format_message(msg)
        assert "Here is the plot:" in result
        assert "[Image]" in result

    def test_format_message_unknown_content_type_fallback(self):
        """Test that unknown content types fall back gracefully without raising an exception."""
        mock_content = MagicMock()
        mock_content.type = "custom_widget"
        msg = MagicMock(spec=ChatMessageAssistant)
        msg.content = [mock_content]
        msg.role = "assistant"
        msg.tool_calls = None

        result = format_message(msg)
        assert "[custom_widget]" in result


    def test_format_message_audio_and_video(self):
        """Test formatting audio and video content blocks."""
        msg = ChatMessageAssistant(
            content=[
                ContentText(type="text", text="Media:"),
                ContentAudio(audio="http://example.com/a.mp3", format="mp3"),
                ContentVideo(video="http://example.com/v.mp4", format="mp4"),
            ]
        )
        res = format_message(msg)
        assert "[Audio]" in res
        assert "[Video]" in res

    def test_format_message_direct_string(self):
        """Test formatting when string is passed directly to format_message."""
        assert format_message("Direct string message") == "Direct string message"


class TestExtractTranscript:
    """Unit tests for extract_transcript in util.py."""

    def test_extract_transcript_single_output(self):
        """Test extracting transcript from single output choice message."""
        sample = SimpleNamespace(
            output=ModelOutput.from_content(model="m1", content="Direct completion answer")
        )
        res = extract_transcript(sample)
        assert res == "Direct completion answer"

    def test_extract_transcript_multi_turn_history(self):
        """Test extracting transcript from full message history (ReAct turns)."""
        sample = SimpleNamespace(
            messages=[
                ChatMessageSystem(content="System prompt"),
                ChatMessageUser(content="Problem statement"),
                ChatMessageAssistant(
                    content="Thinking...",
                    tool_calls=[ToolCall(id="1", function="solve", arguments={})],
                ),
                ChatMessageTool(content="Solved", tool_call_id="1", function="solve"),
                ChatMessageAssistant(content="Done"),
            ]
        )
        res = extract_transcript(sample)
        # System & user prompt filtered out when assistant/tool turns exist
        assert "[Assistant]: Thinking..." in res
        assert "[Tool Call: solve({})]" in res
        assert "[Tool]: [Tool Output (solve)]: Solved" in res
        assert "[Assistant]: Done" in res

    def test_extract_transcript_only_user_and_system_messages(self):
        """Test transcript extraction when messages list contains only system/user messages."""
        sample = SimpleNamespace(
            messages=[
                ChatMessageSystem(content="System instruction"),
                ChatMessageUser(content="User prompt"),
            ]
        )
        res = extract_transcript(sample)
        assert "[System]: System instruction" in res
        assert "[User]: User prompt" in res

    def test_extract_transcript_empty_sample(self):
        """Test transcript extraction when sample has no messages or output."""
        sample = SimpleNamespace()
        assert extract_transcript(sample) == ""


class TestExtractScore:
    """Unit tests for extract_score in util.py."""

    def test_extract_score_correct(self):
        """Test score extraction when score value is 'C' (Correct)."""
        sample = SimpleNamespace(score=Score(value="C"))
        assert extract_score(sample) is True

    def test_extract_score_incorrect(self):
        """Test score extraction when score value is 'I' (Incorrect)."""
        sample = SimpleNamespace(score=Score(value="I"))
        assert extract_score(sample) is False

    def test_extract_score_no_answer(self):
        """Test score extraction when score value is 'N' or NOANSWER."""
        sample = SimpleNamespace(score=Score(value="N"))
        assert extract_score(sample) is False
