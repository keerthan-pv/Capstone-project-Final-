# Production answer path

The research benchmark and the user-facing answer path are now separate.

The benchmark uses MS MARCO to compare retrieval architectures. MS MARCO is a
passage-ranking corpus; it is not a general knowledge base and should not be
used as the only source for questions such as “Who is the CEO of Google?”.

The production path in `production_rag/` uses four explicit routes:

1. `curated_fact`: exact, audited stable facts from `data/curated_facts.json`.
2. `structured_web`: direct evidence from Wikidata, Datamuse, or Wikipedia
   adapters for current or lookup-style questions.
3. `document_retrieval`: an uploaded-document answer supplied by the caller.
4. `abstain`: no answer when the evidence is missing, unrelated, or indirect.

There is no fine-tuning step for these facts. The JSON registry is an auditable
control-plane dataset, not training data. A generative model may be added later
for document summarization, but it must remain behind the evidence and
abstention gates.

For HTTP clients, use `POST /answer` for the grounded path. The existing
`POST /query` endpoint remains available for architecture benchmarking and still
requires the benchmark stack to be ready.
