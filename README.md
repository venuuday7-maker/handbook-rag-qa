# Handbook RAG Q&A

Retrieval-augmented question answering over my [Java & Spring Boot handbook](https://github.com/venuuday7-maker/java-spring-boot-handbook) (chapters 1–2 included in `data/`).

1. **Chunk** the Markdown by section heading (`### 2.5 ...`), splitting long sections into overlapping 220-word windows.
2. **Retrieve** with TF-IDF + cosine similarity. It runs offline with no API key.
3. **Answer** with the best passage, citing its section number. An optional `--llm` flag sends the top passages to Claude for a grounded answer.

```bash
pip install -r requirements.txt
python src/rag.py "What does the JIT compiler do?"
python src/eval.py        # retrieval evaluation
pytest -q
```

## Evaluation (honest numbers)
`data/eval_questions.json` holds 14 hand-written questions, each labelled with the section(s) that really answer it. Retrieval ranks sections and I score hit@k and MRR.

| Retriever | hit@1 | hit@3 | MRR |
|---|---|---|---|
| TF-IDF unigrams (default) | 0.29 | 0.93 | 0.58 |
| TF-IDF uni+bigrams | 0.29 | 0.86 | 0.58 |

**Reading this:** the right section is almost always in the top 3 (13 of 14), but it is first only about 3 times in 10. Keyword matching can't tell neighbouring sections on the same topic apart (for example "JDK/JRE/JVM" vs "lifecycle of a program"). With only 14 questions, differences between settings are within noise, so I did not tune further to avoid overfitting.

## Limitations and next steps
- Add a semantic retriever (sentence embeddings) and a reranker, then re-measure on a larger eval set.
- `--llm` generation (Claude, `ANTHROPIC_API_KEY` required) is implemented but **not tested live** here; everything above was measured with retrieval only.
