<!-- Reference copy of the template used by c2k/eval/entailment.py
     (ENTAIL_PROMPT_TEMPLATE). Kept here per the repo layout in SPEC.md §4;
     the code owns the live copy. -->

You are checking whether a piece of evidence text supports a claimed fact.

Fact: {subject} --[{relation}]--> {object}

Evidence (verbatim span from the source document):
"{evidence_text}"

Does the evidence entail the fact? Answer strictly from the evidence text; do
not use outside knowledge.
