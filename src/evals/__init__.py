"""Reading the collected .eval logs (data/<bench>/<hash>/<model>/<repeat>/*.eval).

    sample.py       one sample -> its transcript text (extract_transcript) or correctness (extract_score)
    transcripts.py  one log -> Transcript per item (text, options, tokens), cached, random access by item
    responses.py    a domain -> the outcome table: one row per (bench, model, item, repeat), correct + guess;
                    eval_logs / repeat0_paths locate the logs
"""
