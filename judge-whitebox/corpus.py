"""Control corpus: same transcripts as judging pairs, no judging-task framing.

Two frames per transcript: a "summarise" chat turn and a raw "continuation" text,
used to compute activity/attribution baselines against non-judging text.
"""

SUMMARISE_SYSTEM = "You are a helpful assistant."
SUMMARISE_USER = "Summarise the following solution:\n---\n{text}\n---"


def build_control(pairs_df, limit=None):
    """One row per pair_uid (order=0 and order=1 share the same two transcripts); limit caps the
    number of pairs. The same transcript text often recurs across different pair_uids (one model's
    answer to an item, paired against several opponents), so transcripts are also deduped by text
    (first occurrence kept, in row order) before emitting the two kinds per unique transcript.
    """
    df = pairs_df.drop_duplicates(subset="pair_uid", keep="first")
    if limit is not None:
        df = df.head(limit)

    out = []
    seen_text = set()
    for _, row in df.iterrows():
        for text in (row["text_1"], row["text_2"]):
            if text in seen_text:
                continue
            seen_text.add(text)
            out.append({
                "kind": "summarise",
                "messages": [
                    {"role": "system", "content": SUMMARISE_SYSTEM},
                    {"role": "user", "content": SUMMARISE_USER.format(text=text)},
                ],
            })
            out.append({"kind": "continuation", "text": row["question"] + "\n" + text})
    return out
