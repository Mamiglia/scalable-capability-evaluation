"""Plain-language definitions for the readable feature blocks."""
import pandas as pd


DEFINITIONS: dict[str, str] = {
    # Surface features.
    "log_chars": "Natural log of the number of characters in assistant-authored text.",
    "n_words": "Whitespace-separated word count in assistant-authored text.",
    "personal_pronouns_count": "Count of first-person pronouns such as I, me, we and our.",
    "hedging_count": "Count of hedging phrases such as perhaps, likely, might and I think.",
    "backtracking_count": "Count of explicit reconsideration phrases such as wait, rethink and try again.",
    "uncertainty_count": "Count of explicit uncertainty phrases such as unsure, ambiguous and I don't know.",
    "confidence_count": "Count of strong-confidence phrases such as obviously, definitely and must be.",
    "questions_count": "Number of question marks in assistant-authored text.",
    "verification_count": "Count of lexical verification cues such as check, confirm, verify and substitute back.",
    "connectives_count": "Count of causal/consequential connectives such as therefore, because and hence.",
    "contrast_count": "Count of contrast markers such as however, although and alternatively.",
    "numbered_steps_count": "Number of lines beginning with a numbered-list marker.",
    "bullets_count": "Number of lines beginning with a bullet marker.",
    "code_chars": "Characters contained inside fenced code blocks.",
    "latex_chars": "Characters contained inside recognised LaTeX spans outside code blocks.",

    # Structure and presentation.
    "assistant_turns": "Number of assistant messages in the transcript.",
    "tool_calls": "Number of tool-call subparts.",
    "tool_errors": "Number of tool outputs matching an error/failure pattern.",
    "tool_outputs_empty": "Number of tool outputs whose body is empty.",
    "submit_calls": "Number of calls to the submit tool.",
    "test_commands": "Number of tool calls whose payload appears to run tests.",
    "repeated_tool_calls": "Number of repeated identical tool-call payloads beyond the first call.",
    "distinct_tools": "Number of distinct tool/function names used.",
    "prose_chars": "Characters in assistant prose, excluding tool calls and outputs.",
    "tool_call_chars": "Characters in tool-call payloads.",
    "tool_output_chars": "Characters in tool outputs.",
    "prose_share": "Assistant prose characters divided by prose plus tool-call and tool-output characters.",
    "bare_answer": "Indicator that fewer than 40 non-answer characters precede the final ANSWER line.",
    "display_equations": "Count of recognised display-maths spans in assistant prose.",
    "inline_math": "Count of recognised inline-maths spans in assistant prose.",
    "option_references": "Count of explicit references to lettered answer options.",
    "elimination_terms": "Count of terms indicating that an option or claim is ruled out.",
    "final_answer_restated": "Indicator that the last 300 characters restate or signal the final answer.",
    "bold_spans": "Number of paired Markdown bold markers in assistant prose.",
    "headers": "Number of Markdown header lines in assistant prose.",

    # GPT-5-mini rubric annotations.
    "reasoning_errors": "Number of incorrect intermediate claims or reasoning steps identified by GPT-5-mini.",
    "unjustified_claims": "Number of conclusions asserted without adequate support, annotated by GPT-5-mini.",
    "redundant_steps": "Number of unnecessary repeated or redundant reasoning steps, annotated by GPT-5-mini.",
    "self_corrections": "Number of places where the transcript notices and corrects its own mistake.",
    "verifies_result": "0/1 GPT-5-mini annotation that the transcript checks its result by testing, substitution or re-derivation.",
    "explains_reasoning": "0/1 GPT-5-mini annotation that the transcript gives a rationale rather than only an answer/actions.",
    "considers_alternatives": "0/1 GPT-5-mini annotation that alternative options, approaches or hypotheses are considered and rejected.",
    "wasted_actions": "GPT-5-mini count of failed, unnecessary or unproductive tool actions; zero for non-agentic transcripts.",
    "clarity": "GPT-5-mini 1–5 rating of how easy the reasoning is to follow.",
    "rigour": "GPT-5-mini 1–5 rating of how carefully each reasoning step is justified.",
    "efficiency": "GPT-5-mini 1–5 rating of how directly the solution is reached without detours.",
    "depth": "GPT-5-mini 1–5 rating of genuine problem understanding beyond the minimum needed.",

    # Exposition.
    "think_aloud_per_100_words": "Think-aloud/process phrases per 100 assistant words.",
    "option_matching_per_100_words": "Approximation or answer-option matching phrases per 100 assistant words.",
    "conditionals_per_100_words": "Conditional, alternative and contrast terms per 100 assistant words.",
    "answer_assertion_per_100_words": "Direct correct-answer assertion phrases per 100 assistant words.",
    "consequence_per_100_words": "Consequence markers such as therefore and hence per 100 assistant words.",
    "label_line_share": "Share of non-empty prose lines beginning with a short declarative label.",
    "step_headers": "Number of lines formatted as a numbered Step header.",
    "bold_labels_per_100_words": "Short bold-labelled spans per 100 assistant words.",
    "mean_line_words": "Mean number of words per non-empty assistant prose line.",

    # Derivation.
    "stated_relations": "Count of equality, approximation, equivalence and implication symbols.",
    "equation_lines": "Number of mathematical spans or prose clauses containing a relation symbol.",
    "chained_equalities": "Number of equation units containing at least two relation symbols.",
    "consequence_markers": "Count of lexical consequence markers such as therefore, yields and we obtain.",
    "justification_markers": "Count of lexical justification markers such as because, by symmetry and given that.",
    "named_methods": "Count of explicitly named theorems, laws, identities, rules or methods.",
    "intermediate_numbers": "Distinct numbers in the working that appear in neither the question/options nor final answer.",
    "defined_symbols": "Number of mathematical symbols first introduced on a left-hand side.",
    "symbol_links": "Number of later equation units that reuse a previously defined symbol.",
    "answer_derived": "Indicator that a final-answer number appears to the right of an equality before the ANSWER line.",
    "relations_per_100_words": "Stated mathematical relations per 100 assistant words.",

    # Codex intermediate-error annotations.
    "error_count": "Total quoted intermediate slips identified by Codex in a correct solution.",
    "any_error": "Indicator that Codex identified at least one intermediate slip.",
    "errors_per_100_words": "Codex-identified intermediate slips per 100 words.",
    "errors_nomenclature": "Count of terminology or naming slips identified by Codex.",
    "errors_notation": "Count of notation slips identified by Codex.",
    "errors_constant_or_formula": "Count of incorrect constants or formulas identified by Codex.",
    "errors_arithmetic": "Count of arithmetic slips identified by Codex.",
    "errors_contradiction": "Count of internally contradictory claims identified by Codex.",
    "errors_definition": "Count of incorrect definitions identified by Codex.",
    "errors_other": "Count of Codex-identified slips not assigned to another error type.",

    # Generic clause-graph descriptors (question node excluded).
    "density": "Undirected edge density: 2E divided by N(N−1).",
    "cyclomatic": "Undirected cycle-rank statistic: edges minus nodes plus connected components.",
    "mean_degree": "Mean undirected node degree.",
    "degree_cv": "Standard deviation of undirected degree divided by mean degree.",
    "max_degree_share": "Maximum undirected degree divided by the number of undirected edges.",
    "assortativity": "Undirected degree assortativity coefficient.",
    "clustering": "Mean local clustering coefficient of the undirected graph.",
    "transitivity": "Global triangle transitivity of the undirected graph.",
    "small_world_sigma": "Clustering-to-path-length small-world ratio relative to an Erdős–Rényi reference.",
    "path_length_mean": "Mean shortest-path length inside the largest connected component.",
    "diameter": "Longest shortest-path distance inside the largest connected component.",
    "giant_component_share": "Share of nodes in the largest connected component.",
    "components_per_node": "Number of connected components divided by node count.",
    "betweenness_centralisation": "Graph-level concentration of undirected betweenness centrality.",
    "communities_per_node": "Greedy-modularity community count divided by node count.",
    "modularity": "Greedy-community modularity of the undirected graph.",
    "source_share": "Share of directed nodes with zero in-degree.",
    "sink_share": "Share of directed nodes with zero out-degree.",
    "edge_length_mean": "Mean forward node-index distance of directed edges.",
    "edge_length_median": "Median forward node-index distance of directed edges.",
    "edge_length_max": "Maximum forward node-index distance of directed edges.",
    "edge_length_cv": "Standard deviation divided by mean directed edge length.",
    "adjacent_edge_share": "Share of directed edges connecting immediately adjacent nodes.",
    "long_edge_share": "Share of directed edges spanning more than five node positions.",
    "edge_length_entropy": "Entropy of directed edge lengths, with lengths capped at 20.",
    "type_entropy": "Entropy of clause/node type frequencies.",
    "relation_entropy": "Entropy of edge-relation label frequencies.",

    # Named clause motifs.
    "in_degree_max": "Maximum directed in-degree over clause-network nodes.",
    "out_degree_max": "Maximum directed out-degree over clause-network nodes.",
    "fan_in_share": "Share of nodes with at least two parents.",
    "fan_out_share": "Share of nodes with at least two children.",
    "sequential_share": "Share of nodes with at most one parent and at most one child.",
    "depth_max": "Longest directed ancestry depth in node order, measured in edges from a root.",
    "width_max": "Largest number of nodes occurring at the same directed depth.",
    "depth_over_width": "Maximum depth plus one divided by maximum width.",
    "answer_in_degree": "Largest in-degree of an answer node.",
    "sibling_convergence_max": "Largest count of same-type parents converging on one node.",
    "plan_fan_out": "Out-degree of the highest-fan-out node in the first fifth of the transcript.",
    "plan_coverage": "Share of nodes descending from that early high-fan-out plan candidate.",
    "motif_chain": "Directed three-node chain motifs per node.",
    "motif_fan_in": "Directed three-node fan-in motifs per node.",
    "motif_fan_out": "Directed three-node fan-out motifs per node.",
    "motif_feed_forward": "Directed feed-forward triangle motifs per node.",
    "after_tool_backtrack": "Share of tool outputs followed by a child clause marked as backtracking.",
    "after_tool_verify": "Share of tool outputs followed by a child clause marked as verification.",
    "after_tool_adversative": "Share of tool outputs followed by an adversative child clause.",
    "after_tool_another_tool": "Share of tool outputs followed directly by another tool call.",
    "after_error_backtrack": "Share of error tool outputs followed by a backtracking child clause.",
    "equation_explained": "Share of equation nodes followed by a non-equation explanatory clause.",
    "equation_to_equation": "Share of equation nodes with an equation child.",
    "equation_from_clause": "Share of equation nodes with at least one non-equation parent.",
    "heading_fan_out": "Share of heading nodes with at least two children.",
    "backtrack_share": "Share of all nodes marked as backtracking.",
    "verify_share": "Share of all nodes marked as verification.",

    # Explicit plan-realisation features.
    "plan_items": "Number of detected plan items linked to later tool calls.",
    "plan_present": "Indicator that at least two early plan items were detected and linked.",
    "plan_executed_share": "Share of detected plan items matched to a later tool call of the intended action class.",
    "plan_order_concordance": "Kendall correlation between plan-item order and matched execution order.",
    "plan_unplanned_share": "Share of later tool calls not used as the target of any plan item.",
    "plan_position": "Node position of the first plan item divided by transcript node count.",
}


ANNOTATED_BLOCKS: dict[str, str] = {
    "rubric": "LLM annotation",
    "intermediate-errors": "LLM annotation",
}


def definitions_for(dictionary: pd.DataFrame) -> pd.DataFrame:
    """Attach definitions and measurement provenance; fail on undocumented features."""
    required = {"feature", "block"}
    if missing := required - set(dictionary):
        raise ValueError(f"feature dictionary missing columns: {sorted(missing)}")
    out = dictionary.copy()
    out["definition"] = out.feature.map(DEFINITIONS)
    if missing := out.loc[out.definition.isna(), "feature"].tolist():
        raise ValueError(f"features missing definitions: {', '.join(missing[:10])}")
    out["measurement"] = out.block.map(ANNOTATED_BLOCKS).fillna("Deterministic")
    return out
