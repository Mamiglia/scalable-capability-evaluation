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

from util import (extract_score, extract_transcript, extract_transcript_parts,
                  first_correct_by_id, format_content, format_message,
                  render_transcript_parts, strip_think, transcript_content_hash)


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

    def test_structured_parts_reconcile_marker_like_content(self):
        sample = SimpleNamespace(messages=[
            ChatMessageUser(content="Problem statement"),
            ChatMessageAssistant(content="Literal marker:\n\n[Tool]: not a role"),
            ChatMessageTool(content="actual output", tool_call_id="1", function="solve"),
        ])

        parts = extract_transcript_parts(sample, include_reasoning=False)
        rendered = render_transcript_parts(parts)

        assert rendered == extract_transcript(sample, include_reasoning=False)
        assert [part["role"] for part in parts] == ["assistant", "tool"]
        assert rendered[parts[0]["start"]:parts[0]["end"]] == "Literal marker:\n\n[Tool]: not a role"

    def test_content_hash_covers_question_roles_and_rendering(self):
        parts = [{"role": "assistant", "content": "answer", "start": 13,
                  "end": 19, "bare": False}]
        original = transcript_content_hash("question", "[Assistant]: answer", parts)
        assert original == transcript_content_hash("question", "[Assistant]: answer", parts)
        assert original != transcript_content_hash("other", "[Assistant]: answer", parts)
        assert original != transcript_content_hash("question", "[Assistant]: answer!", parts)

    def test_tool_subparts_keep_spans_and_raw_metadata(self):
        sample = SimpleNamespace(messages=[
            ChatMessageUser(content="Problem"),
            ChatMessageAssistant(content="Checking", tool_calls=[
                ToolCall(id="call-1", function="bash", arguments={"cmd": "pwd"})]),
            ChatMessageTool(content=[ContentText(type="text", text="/tmp")],
                            tool_call_id="call-1", function="bash"),
        ])
        parts = extract_transcript_parts(sample, include_reasoning=False)
        rendered = render_transcript_parts(parts)
        tool_call = parts[0]["subparts"][1]
        tool_output = parts[1]["subparts"][0]

        assert (tool_call["kind"], tool_call["call_id"], tool_call["function"],
                tool_call["raw_payload"]) == ("tool_call", "call-1", "bash", {"cmd": "pwd"})
        assert (tool_output["kind"], tool_output["call_id"], tool_output["function"],
                tool_output["raw_payload"]) == (
                    "tool_output", "call-1", "bash", [{"type": "text", "text": "/tmp"}])
        assert rendered[tool_call["start"]:tool_call["end"]] == "[Tool Call: bash({'cmd': 'pwd'})]"
        assert rendered[tool_output["start"]:tool_output["end"]].endswith(": /tmp")


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

    def test_extract_score_numeric_and_nested(self):
        assert extract_score(SimpleNamespace(scores={"scorer": Score(value=1)})) is True
        assert extract_score(SimpleNamespace(scores={"scorer": Score(value={"score": 1})})) is True
        assert extract_score(SimpleNamespace(scores={"scorer": Score(value=0)})) is False


class TestFirstCorrectById:
    """Epoch selection must match build_dataset: earliest CORRECT epoch first."""

    @staticmethod
    def _log(*samples):
        return SimpleNamespace(samples=list(samples))

    @staticmethod
    def _sample(item, epoch, correct, text):
        return SimpleNamespace(id=item, epoch=epoch,
                               scores={"scorer": Score(value="C" if correct else "I")},
                               messages=[ChatMessageAssistant(content=text)])

    def test_earliest_correct_epoch_wins_over_later_epochs(self):
        """Later epochs must not overwrite the first correct one (the original bug)."""
        log = self._log(
            self._sample("q1", 1, True, "first correct"),
            self._sample("q1", 2, True, "second correct"),
            self._sample("q1", 3, True, "last epoch"),
        )
        groups = first_correct_by_id(log)
        assert extract_transcript(groups["q1"][0]).endswith("first correct")
        # a drift back to last-epoch selection would put "last epoch" here
        assert extract_transcript(groups["q1"][1]).endswith("second correct")

    def test_incorrect_epochs_are_skipped(self):
        """The first correct epoch is neither the first nor the last epoch here."""
        log = self._log(
            self._sample("q1", 1, False, "wrong early"),
            self._sample("q1", 2, True, "right"),
            self._sample("q1", 3, False, "wrong late"),
        )
        groups = first_correct_by_id(log)
        assert len(groups["q1"]) == 1
        assert extract_transcript(groups["q1"][0]).endswith("right")

    def test_out_of_order_samples_are_sorted_by_epoch(self):
        """Log order must not decide the winner; epoch does."""
        log = self._log(
            self._sample("q1", 1, True, "epoch one"),
            self._sample("q1", 3, True, "epoch three"),
            self._sample("q1", 2, True, "epoch two"),
        )
        assert extract_transcript(first_correct_by_id(log)["q1"][0]).endswith("epoch one")

    def test_item_with_no_correct_epoch_is_absent(self):
        log = self._log(self._sample("q1", 1, False, "wrong"))
        assert "q1" not in first_correct_by_id(log)

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

    def test_format_content_strips_inline_think_only_when_excluded(self):
        block = [ContentText(type="text", text="<think>secret CoT</think>Answer")]
        assert format_content(block, include_reasoning=False) == "Answer"
        assert "secret CoT" in format_content(block, include_reasoning=True)

    def test_format_content_str_content_stripped(self):
        assert format_content("<think>x</think>Y", include_reasoning=False) == "Y"
