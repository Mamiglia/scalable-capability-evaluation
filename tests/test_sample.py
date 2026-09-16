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

from evals.sample import format_message, extract_transcript, extract_score, format_content, strip_think


class TestFormatMessage:
    """Unit tests for format_message in util.py."""

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

    def test_format_message_summary_is_never_shown(self):
        """A redacted block holds a blob and a provider summary: neither reaches the judge."""
        msg = ChatMessageAssistant(
            content=[
                ContentReasoning(type="reasoning", reasoning="<blob>", summary="Gist", redacted=True),
                ContentText(type="text", text="Final Answer"),
            ]
        )
        result = format_message(msg)
        assert "Gist" not in result and "<blob>" not in result
        assert "THOUGHT" not in result and "SUMMARY" not in result
        assert result.strip().endswith("Final Answer")

    def test_format_message_inline_think_is_dropped_with_a_warning(self, caplog):
        """An un-normalized log must warn and drop the CoT, never leak it as answer text."""
        msg = ChatMessageAssistant(content=[ContentText(type="text", text="<think>Step 1</think>Final Answer")])
        result = format_message(msg)
        assert "Step 1" not in result and "<think>" not in result
        assert result.strip().endswith("Final Answer")
        assert any("not normalized" in r.getMessage() for r in caplog.records)

    def test_format_message_empty_summary_renders_nothing(self):
        """Redacted with an empty summary (opus-5, fable-5) must not look like reasoning."""
        msg = ChatMessageAssistant(
            content=[
                ContentReasoning(type="reasoning", reasoning="<blob>", summary="", redacted=True),
                ContentText(type="text", text="Final Answer"),
            ]
        )
        result = format_message(msg)
        assert "SUMMARY" not in result and "THOUGHT" not in result
        assert result.strip().endswith("Final Answer")

    def test_format_message_tool_calls(self):
        """Test formatting assistant message with tool calls."""
        msg = ChatMessageAssistant(
            content="Running bash script",
            tool_calls=[ToolCall(id="tc1", function="bash", arguments={"cmd": "ls -l"})],
        )
        result = format_message(msg)
        assert "Running bash script" in result
        assert "[Tool Call: bash({'cmd': 'ls -l'})]" in result

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

    def test_extract_transcript_keeps_later_user_turns(self):
        """Only the initial user prompt is dropped; scaffold user feedback stays."""
        sample = SimpleNamespace(
            messages=[
                ChatMessageUser(content="Problem statement"),
                ChatMessageAssistant(content="First try"),
                ChatMessageUser(content="Incorrect, try again"),
                ChatMessageAssistant(content="Second try"),
            ]
        )
        res = extract_transcript(sample)
        assert "Problem statement" not in res
        assert "[User]: Incorrect, try again" in res
        assert "[Assistant]: Second try" in res

    def test_extract_transcript_empty_sample(self):
        """Test transcript extraction when sample has no messages or output."""
        sample = SimpleNamespace()
        assert extract_transcript(sample) == ""


class TestExtractScore:
    """Unit tests for extract_score in util.py."""

    def test_extract_score_correct(self):
        """Test score extraction when score value is 'C' (Correct)."""
        sample = SimpleNamespace(scores={"scorer": Score(value="C")})
        assert extract_score(sample) is True

    def test_extract_score_incorrect(self):
        """Test score extraction when score value is 'I' (Incorrect)."""
        sample = SimpleNamespace(scores={"scorer": Score(value="I")})
        assert extract_score(sample) is False

    def test_extract_score_swe_bench_float(self):
        """SWE-bench's test-run scorer stores a float 1.0/0.0, not 'C'/'I'."""
        assert extract_score(SimpleNamespace(scores={"s": Score(value=1.0)})) is True
        assert extract_score(SimpleNamespace(scores={"s": Score(value=0.0)})) is False

    def test_extract_score_hle_dict(self):
        """HLE nests the label under 'score' alongside a confidence."""
        assert extract_score(SimpleNamespace(scores={"s": Score(value={"score": "C", "confidence": 80})})) is True
        assert extract_score(SimpleNamespace(scores={"s": Score(value={"score": "I", "confidence": 80})})) is False

    def test_extract_score_no_scores(self):
        """A sample with no scores is not correct."""
        assert extract_score(SimpleNamespace(scores={})) is False


class TestStripThink:
    """Inline DeepSeek <think> CoT must be dropped when reasoning is excluded."""

    def test_closed_think_keeps_only_answer(self):
        assert strip_think("<think>lots of reasoning</think>\n\nThe answer is 42.") == "The answer is 42."

    def test_unclosed_think_drops_everything(self):
        # model looped and never closed the tag / never produced an answer
        assert strip_think("<think>reasoning that never ends") == ""

    def test_no_think_unchanged(self):
        assert strip_think("Just a plain answer.") == "Just a plain answer."

    def test_format_content_drops_inline_think_in_both_conditions(self):
        """Un-normalized inline CoT is dropped either way -- never shown, never leaked."""
        block = [ContentText(type="text", text="<think>secret CoT</think>Answer")]
        assert format_content(block, include_reasoning=False) == "Answer"
        assert format_content(block, include_reasoning=True) == "Answer"

    def test_format_content_str_content_stripped(self):
        assert format_content("<think>x</think>Y", include_reasoning=False) == "Y"


class TestChannels:
    """A channel spec keeps or drops one of reasoning, text and tools (calls with their outputs)."""

    turns = [
        ChatMessageAssistant(
            content=[ContentReasoning(type="reasoning", reasoning="Plan"), ContentText(type="text", text="Looking")],
            tool_calls=[ToolCall(id="1", function="bash", arguments={"cmd": "ls"})]),
        ChatMessageTool(content="file.txt", tool_call_id="1", function="bash"),
        ChatMessageAssistant(content=[ContentReasoning(type="reasoning", reasoning="Done"),
                                      ContentText(type="text", text="ANSWER: file.txt")]),
    ]

    def render(self, channels):
        return format_message(self.turns, channels=channels)

    def test_full_is_unchanged(self):
        assert self.render("") == format_message(self.turns)

    def test_only_one_channel(self):
        reasoning = self.render("only-reasoning")
        assert "Plan" in reasoning and "Done" in reasoning
        assert "Looking" not in reasoning and "bash" not in reasoning and "file.txt" not in reasoning
        text = self.render("only-text")
        assert "Looking" in text and "ANSWER" in text and "THOUGHT" not in text and "Tool" not in text
        tools = self.render("only-tools")
        assert "[Tool Call: bash" in tools and "[Tool Output (bash)]: file.txt" in tools
        assert "Plan" not in tools and "Looking" not in tools and "ANSWER" not in tools

    def test_drop_one_channel(self):
        assert "THOUGHT" not in self.render("drop-reasoning") and "Looking" in self.render("drop-reasoning")
        assert "Looking" not in self.render("drop-text") and "Plan" in self.render("drop-text")
        dropped = self.render("drop-tools")
        assert "Tool" not in dropped and "Plan" in dropped and "Looking" in dropped

    def test_empty_turns_vanish(self):
        """A turn with nothing left in it is not shown as an empty role line."""
        assert "[Tool]" not in self.render("only-text")
        assert not any(line.strip() == "[Assistant]:" for line in self.render("only-tools").splitlines())

    def test_reasoning_hidden_beats_channels(self):
        assert "THOUGHT" not in format_message(self.turns, include_reasoning=False, channels="only-reasoning")

    def test_bad_spec(self):
        import pytest
        with pytest.raises(ValueError):
            self.render("only-images")
