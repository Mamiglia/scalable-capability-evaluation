from inspect_ai.model import ChatMessageAssistant
from inspect_ai.dataset import Sample

def format_message(msg : ChatMessageAssistant) -> str:
    """
    Format the message object into a readable string.
    """
    if isinstance(msg.content, str):
        return msg.text
    
    texts = []
    for content in msg.content:
        match content.type:
            case "text":
                texts.append(content.text)
            case "reasoning":
                texts.append(f"<THOUGHT_START>\n {content.text}\n<THOUGHT_END>")
            case _:
                raise ValueError(f"Unknown content type: {content.type}")
    return "\n".join(texts)

def extract_score(sample: Sample) -> bool:
    """
    Extract the score from the sample's output.
    """
    score = sample.score
    return score.value == 'C'