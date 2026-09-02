# Clause-network construction: methods and references

This note explains the individual methods used to construct the clause network. It treats the overall graph design as given and distinguishes established methods from project-specific engineering choices.

## Rhetorical Structure Theory and elementary discourse units

Rhetorical Structure Theory (RST) represents a text as recursively related spans. Its leaves are elementary discourse units (EDUs), which usually correspond to clauses. Internal relations are either mononuclear, with a central nucleus and a supporting satellite, or multinuclear, with multiple peer spans. The `gumrrg` version of the IsaNLP RST parser is trained on English GUM and Russian RRG annotations.

References:

- Mann, W. C., and Thompson, S. A. (1988). [Rhetorical Structure Theory: Toward a Functional Theory of Text Organization](https://www.sfu.ca/rst/pdfs/Mann_Thompson_1987.pdf).
- Chistova, E. (2024). [Bilingual Rhetorical Structure Parsing with Large Parallel Annotations](https://aclanthology.org/2024.findings-acl.577/).
- [IsaNLP RST parser documentation](https://github.com/tchewik/isanlp_rst).

## Reversible masking and exact span anchoring

Display mathematics, fenced code, headings, and `ANSWER:` lines are replaced by reversible placeholders so that a prose-trained discourse parser does not invent clause boundaries inside structured material. Inline mathematics, inline code, and Markdown emphasis syntax are masked for the same reason. This is structure-aware preprocessing rather than a standard RST algorithm.

The original transcript remains immutable. Parsed units are attached to it using half-open character intervals, following the general standoff-annotation pattern in which annotations are stored separately from the source text.

Reference:

- [brat standoff annotation format](https://brat.nlplab.org/standoff.html).

## Boundary repair as constrained monotonic alignment

Parser offsets are treated as approximate boundary suggestions. Because RST leaves have a known order, they can be reconstructed as an ordered partition of the original whole-word tokens. This can be described as **constrained monotonic alignment from predicted EDU boundaries to original token boundaries**.

The reconstruction is accepted only when it satisfies four invariants:

1. every semantic source token is assigned exactly once;
2. reconstructed units remain in transcript order;
3. adjacent units do not overlap or leave semantic gaps; and
4. every boundary falls between complete word tokens.

This repair procedure is a project-specific correctness wrapper around the parser, not a literature-standard component of RST parsing.

## Nucleus-head propagation

An RST parser returns a constituency tree containing non-terminal spans, whereas the clause network needs relations between observable clause nodes. Nucleus-head propagation converts the constituency tree into a discourse dependency representation:

1. recursively assign each span a representative leaf;
2. for a mononuclear relation, propagate the representative leaf of the nuclear child;
3. connect the representative leaves of the related child spans; and
4. propagate the selected representative upward.

Multinuclear relations do not supply a unique head. Any leftmost, rightmost, or relation-specific choice is therefore a conversion convention and must be stated explicitly. Nuclearity and dependency headedness are related but not equivalent.

Reference:

- Morey, M., Muller, P., and Asher, N. (2018). [A Dependency Perspective on RST Discourse Parsing and Evaluation](https://aclanthology.org/J18-2001.pdf).

## Typed transcript and tool nodes

Questions, prose clauses, equations, code blocks, headings, answers, tool calls, tool outputs, and user feedback form a heterogeneous node ontology. The `inspect`/`act`, `ok`/`error`, backtracking, and verification labels are lightweight annotations over that ontology rather than outputs prescribed by RST.

A tool call and its output can be understood as an activity and a generated entity in a provenance graph. The subsequent output-to-assistant edge is a directly-follows relation: it records temporal succession and available context, but does not by itself establish semantic causation.

Reference:

- [W3C PROV Data Model](https://www.w3.org/TR/prov-dm/).

## Cosine-similarity edges as a thresholded k-nearest-neighbour graph

For unit-normalised embeddings, cosine similarity reduces to a dot product:

\[
\operatorname{cos}(x,y)=x^\top y.
\]

Each target node is connected to at most its top-\(k\) cross-message neighbours whose similarity exceeds a threshold. The top-\(k\) restriction bounds degree, while the threshold rejects neighbours that are merely the best among poor candidates. Restricting candidates to other messages makes these edges represent semantic recurrence across conversational turns rather than local paraphrase within one message.

The rule that similarity edges should not exceed half of all edges is a project-specific density diagnostic. It prevents approximate semantic links from overwhelming higher-confidence structural links, but it is not a statistical calibration of edge precision.

Reference:

- Liu, Y., Cheng, H., and Cui, J. (2021). [Revisiting k-Nearest Neighbor Graph Construction on High-Dimensional Data](https://arxiv.org/abs/2112.02234).

## Shared identifiers as sparse lexical or entity chains

Repeated identifiers provide a high-precision signal that separated clauses refer to the same concrete object. Linking each occurrence to the nearest earlier occurrence produces a sparse, recency-biased chain rather than a clique over every occurrence. This is a code-aware analogue of lexical chains and entity-based coherence models.

References:

- Morris, J., and Hirst, G. (1991). [Lexical Cohesion Computed by Thesaural Relations as an Indicator of the Structure of Text](https://www.cs.toronto.edu/pub/gh/Morris%2BHirst-91.pdf).
- Barzilay, R., and Lapata, M. (2008). [Modeling Local Coherence: An Entity-Based Approach](https://aclanthology.org/J08-1001/).

## Parent prioritisation and degree capping

Structural RST and tool-provenance parents are retained before approximate similarity and identifier parents. A total parent cap then sparsifies the graph and prevents a few highly connected nodes from dominating degree-based features. The priority order and cap are modelling choices rather than properties of RST or k-nearest-neighbour graphs.

## Node embeddings and long-text pooling

`Alibaba-NLP/gte-modernbert-base` produces 768-dimensional text embeddings and supports inputs of up to 8,192 tokens. The 2,048-token chunk length is therefore a chosen compute and representation setting, not the model's maximum context length.

For a node split into \(m\) chunks with embeddings \(h_1,\ldots,h_m\), the graph uses

\[
z=\frac{\sum_{i=1}^{m}h_i}{\left\lVert\sum_{i=1}^{m}h_i\right\rVert_2}.
\]

This gives each chunk equal influence and returns a unit vector suitable for cosine similarity. If the final chunk is substantially shorter than the others, a token-count-weighted mean is an alternative; equal weighting should therefore be documented as a deliberate choice.

Caching by the SHA-1 hash of the exact text is content-addressed memoisation: identical text maps to identical cached work independently of transcript identity or job assignment. This is valid when the embedding depends only on the text; any later inclusion of node type or conversational context would need to become part of the cache key.

Reference:

- [Alibaba-NLP/gte-modernbert-base model card](https://huggingface.co/Alibaba-NLP/gte-modernbert-base).

## Edge representations

Given source and target embeddings \(s\) and \(t\), an edge is initially represented as

\[
e=[s,\ t,\ t-s,\ s\odot t].
\]

The endpoint vectors retain the source and target content. The signed difference records directional change, while the elementwise product records coordinate-wise agreement. This is a standard heuristic-matching construction for paired representations.

Principal component analysis (PCA) then projects the \(4d\)-dimensional vectors onto orthogonal directions that retain the greatest variance. Because PCA is a learned transformation, it should be fitted on development items only, along with the edge SAE, and then applied unchanged to evaluation items.

References:

- Mou, L. et al. (2016). [Natural Language Inference by Tree-Based Convolution and Heuristic Matching](https://aclanthology.org/P16-2022.pdf).
- Jolliffe, I. T., and Cadima, J. (2016). [Principal Component Analysis: A Review and Recent Developments](https://pmc.ncbi.nlm.nih.gov/articles/PMC4792409/).

## Top-K sparse autoencoders

A Top-K sparse autoencoder learns an overcomplete dictionary while retaining only the \(K\) largest latent activations for each input. For encoder activations \(a=W_e x+b_e\), the sparse code is

\[
z=\operatorname{TopK}(a,K),
\]

and a decoder reconstructs \(x\) from \(z\). Fixing the number of active latents directly controls sparsity and avoids balancing reconstruction loss against a separate sparsity penalty.

Separate node and edge SAEs allow the two vector spaces to learn different dictionaries. Training separate models by benchmark group permits group-specific factors, while fitting on development items only prevents the learned basis from adapting to evaluation examples. Applying SAEs to clause and edge embeddings is an adaptation of sparse dictionary learning; the cited work does not specifically validate this graph interpretation.

References:

- Makhzani, A., and Frey, B. (2013). [k-Sparse Autoencoders](https://arxiv.org/abs/1312.5663).
- Gao, L. et al. (2024). [Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093).

## Graph features and motifs

Typed degree, depth, component, ancestry, edge-share, and relation-share features summarise the graph at transcript level. Shares normalise raw counts for transcript length and graph density. Depth and ancestry must be defined either on an acyclic structural subgraph or on a condensation of strongly connected components if non-structural edges can introduce cycles.

Tool/error patterns and marker combinations are typed local-subgraph counts. They should be called **motif counts** unless enrichment is measured against a random-graph null model. In the stricter network-science usage, a network motif is a pattern that occurs more often than expected under such a null model.

Reference:

- Milo, R. et al. (2002). [Network Motifs: Simple Building Blocks of Complex Networks](https://www.cs.cornell.edu/courses/cs6241/2020sp/readings/Milo-2002-motifs.pdf).

## Left-minus-right pair features

For a transcript-level feature map \(\phi\), the paired representation

\[
\Delta\phi=\phi(\text{left})-\phi(\text{right})
\]

encodes the direction and magnitude of each within-pair difference. Swapping the two transcripts negates the vector, which gives the representation a useful antisymmetry. It also removes feature components shared by both responses to the same benchmark item. The left/right orientation must remain consistent with the target label throughout export and modelling.

## Project-specific choices to report explicitly

The following are defensible design decisions, but should not be presented as canonical algorithms:

- reversible placeholder and masking rules;
- monotonic whole-token boundary repair and its validation invariants;
- the node ontology and inspect/act, ok/error, backtracking, and verification labels;
- the 2,048-token chunk length and equal-weight chunk pooling;
- structural-parent priority and the total-parent cap;
- the similarity threshold and 50% edge-share safeguard; and
- the exact aggregation of node and edge SAE activations into transcript features.
