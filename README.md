*This project has been created as part of the 42 curriculum by dgaillet.*

# RAG Against the Machine

## Description

**RAG Against the Machine** is a local Retrieval-Augmented Generation (RAG) system designed to answer questions about a source-code repository.

The project indexes the provided vLLM corpus, retrieves relevant source-code or documentation chunks, and passes those chunks to the local `Qwen/Qwen3-0.6B` language model. The model is instructed to answer only from the retrieved evidence to reduce hallucinations.

The pipeline contains four main stages:

1. **Indexing** — read files, divide them into searchable chunks, and persist the indexes.
2. **Retrieval** — rank chunks using BM25, semantic similarity, or hybrid retrieval.
3. **Augmentation** — insert the retrieved source text into the model prompt.
4. **Generation** — use Qwen to produce a concise, grounded answer.

Retrieval quality can be measured with **recall@k**, using file-path equality and character-range overlap against a ground-truth dataset.

## Features

### Mandatory features

- Python 3.10 or later
- Pydantic data models
- Python Fire command-line interface
- Progress bars with `tqdm`
- Configurable chunk size, with a maximum of 2,000 characters
- Separate Python and Markdown/text chunking strategies
- BM25 lexical retrieval
- Single-query and dataset retrieval
- Local answer generation with `Qwen/Qwen3-0.6B`
- Recall@k evaluation
- Persistent indexes under `data/processed/`
- Graceful handling of common invalid inputs

### Additional features

- Semantic embeddings using `all-MiniLM-L6-v2`
- Semantic retrieval through cosine similarity
- Hybrid BM25 and semantic retrieval
- Weighted Reciprocal Rank Fusion
- Incremental file chunking
- Search-result caching
- Answer caching
- Experimental local FastAPI interface

---

## System Architecture

```text
                         data/raw/
                             |
                             v
                  +---------------------+
                  |       Indexer       |
                  |---------------------|
                  | File discovery      |
                  | Type-aware chunking |
                  | Character positions |
                  +---------------------+
                       |             |
                       v             v
                +-----------+   +------------------+
                | BM25 index|   | Semantic vectors |
                +-----------+   +------------------+
                       \             /
                        \           /
                         v         v
                     +---------------+
Question ----------> |    Search     |
                     |---------------|
                     | BM25          |
                     | Semantic      |
                     | Hybrid RRF    |
                     +---------------+
                             |
                             v
                    Retrieved sources
                             |
                             v
                     +---------------+
                     |    Answer     |
                     |---------------|
                     | Build context |
                     | Ground prompt |
                     | Local Qwen    |
                     +---------------+
                             |
                             v
                      Grounded answer
```

### Components

#### `Indexer`

The indexer scans supported files under `data/raw/`, creates chunks, records their exact character positions, and persists:

- BM25 index data
- BM25 tokenizer vocabulary and stop words
- Semantic embeddings
- Chunk metadata
- File modification metadata

Generated files are stored under `data/processed/`.

#### `Search`

The search engine loads the persisted indexes and supports three retrieval modes:

- `bm25`
- `semantic`
- `hybrid`

It returns `MinimalSource` objects containing:

- The exact corpus-relative file path
- The first character index
- The last character index
- An internal chunk identifier

The internal `chunk_id` is useful for index management, while the other three fields form the minimal source location expected by the project format.

#### `Answer`

The answer generator:

1. Retrieves the top-k source chunks, unless sources are already supplied.
2. Reads the exact text represented by each source range.
3. Builds a prompt containing the question and retrieved evidence.
4. Instructs Qwen to use only the supplied sources.
5. Generates a concise answer.
6. Caches the generated result.

#### `Evaluate`

The evaluator compares retrieved sources with ground-truth sources. A source is considered found when:

- Both sources have exactly the same `file_path`.
- Their character ranges have an Intersection over Union of at least `0.05`.

Recall is reported at:

- Recall@1
- Recall@3
- Recall@5
- Recall@10

---

## Project Structure

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   ├── datasets/
│   │   ├── AnsweredQuestions/
│   │   └── UnansweredQuestions/
│   └── output/
│       ├── search_results/
│       └── search_results_and_answer/
├── src/
│   ├── classes/
│   │   └── models.py
│   ├── __init__.py
│   ├── __main__.py
│   ├── answer.py
│   ├── evaluate.py
│   ├── indexer.py
│   ├── local_llm.py
│   └── search.py
├── Makefile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Chunking Strategy

The corpus contains both source code and natural-language documentation. These formats have different structures, so the project uses type-specific recursive splitters.

### Python chunking

Python files use LangChain's Python-aware recursive splitter:

```python
RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    ...
)
```

The splitter attempts to preserve meaningful Python boundaries such as classes, functions, and blocks before falling back to smaller separators.

### Markdown chunking

Markdown files use the Markdown-aware splitter. It prioritizes structural boundaries such as:

- Headings
- Sections
- Paragraphs
- Lines

This helps keep related documentation together.

### RST and plain-text chunking

- `.rst` files use the RST-aware splitter.
- `.txt` files use the generic recursive character splitter.

### Chunk overlap

Chunks use an overlap of 200 characters. The overlap reduces the chance that an important definition or explanation is lost at a chunk boundary.

### Exact character positions

Whitespace stripping is disabled, and each generated document includes its starting index. The source range is calculated as:

```text
first_character_index = start_index
last_character_index  = start_index + length_of_chunk
```

The last index is treated as an exclusive boundary when reading the source with Python slicing.

### Configurable size

The default maximum chunk size is 2,000 characters:

```bash
uv run python -m src index --max_chunk_size 2000
```

A smaller size may improve retrieval precision but can separate related information. A larger size provides more context but increases noise.

---

## Retrieval Method

### BM25

BM25 is the default mandatory lexical retrieval method.

During indexing:

1. Chunk text is tokenized.
2. English stemming is applied with PyStemmer.
3. The BM25 index is created with `bm25s`.
4. The index and tokenizer data are persisted.

During retrieval:

1. The question is tokenized with the saved tokenizer.
2. BM25 scores the indexed chunks.
3. Results are ranked by relevance.
4. The top-k source locations are returned.

BM25 is particularly effective when a question contains:

- Function names
- Class names
- Configuration options
- Command-line flags
- Error messages
- Other identifiers found directly in the corpus

### Semantic retrieval

Semantic retrieval uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Every indexed chunk is converted into a normalized embedding. A query is encoded with the same model, and cosine similarity is calculated using the dot product between normalized vectors.

Semantic retrieval can help when the question paraphrases the source instead of using its exact vocabulary.

### Hybrid retrieval

Hybrid mode combines BM25 and semantic rankings using weighted Reciprocal Rank Fusion:

```text
score(document) =
    lexical_weight / (60 + lexical_rank)
    + semantic_weight / (60 + semantic_rank)
```

The default semantic weight is `0.05`, giving BM25 a stronger influence:

```text
lexical_weight = 0.95
semantic_weight = 0.05
```

Rank fusion is used instead of directly adding BM25 and cosine-similarity scores because those scores have different numerical scales.

### Candidate pool

Before selecting the final top-k results, the retriever can consider a larger candidate pool:

```text
candidate_count = k × candidate_multiplier
```

The default candidate multiplier is `10`.

---

## Incremental Indexing and Caching

### Incremental chunking

File metadata is saved in:

```text
data/processed/file_inf_cache.json
```

For each source file, the cache records:

- File path
- Modification timestamp in nanoseconds
- File size

On a later indexing run:

- Unchanged files reuse their existing chunks.
- New files are chunked.
- Modified files are chunked again.
- Chunks belonging to deleted files are removed.

The searchable indexes are then rebuilt from the resulting current chunk collection.

### Search cache

Search results are cached in:

```text
data/processed/search_cache.json
```

Repeated identical queries can reuse previously retrieved sources.

### Answer cache

Generated answers are cached in:

```text
data/processed/answer_cache.json
```

The answer cache also records the value of `k`, because changing the number of retrieved sources can change the generated answer.

Delete these files when comparing retrieval configurations to avoid reusing results produced by an older configuration.

---

## Instructions

### Requirements

- Python 3.10 or later
- `uv`
- Sufficient disk space for:
  - The vLLM corpus
  - Transformer dependencies
  - Qwen model weights
  - Sentence-transformer weights
  - Generated indexes

A GPU is optional. Qwen uses CUDA when it is available and otherwise runs on the CPU.

---

## Usage

All mandatory commands use Python Fire and follow this form:

```bash
uv run python -m src <command> [arguments]
```

### 1. Build the index

```bash
uv run python -m src index --max_chunk_size 2000
```

This reads supported files under `data/raw/` and stores generated index files under `data/processed/`.

Do not set `max_chunk_size` above 2,000.

### 2. Search a single query

BM25 search:

```bash
uv run python -m src search \
  "How is the OpenAI-compatible server configured?" \
  --k 5
```

Semantic search:

```bash
uv run python -m src search \
  "How is the OpenAI-compatible server configured?" \
  --k 5 \
  --retrieval_mode semantic
```

Hybrid search:

```bash
uv run python -m src search \
  "How is the OpenAI-compatible server configured?" \
  --k 5 \
  --retrieval_mode hybrid \
  --semantic_weight 0.05 \
  --candidate_multiplier 10
```

Each result contains the exact file path and character range of a retrieved chunk.

### 3. Search a dataset

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

Hybrid dataset retrieval can be selected with:

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions \
  --retrieval_mode hybrid \
  --semantic_weight 0.05 \
  --candidate_multiplier 10
```

The output file follows the `StudentSearchResults` Pydantic model.

### 4. Evaluate retrieval

```bash
uv run python -m src evaluate \
  --student_search_results_path \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path \
  data/datasets/AnsweredQuestions/dataset_docs_public.json
```

This reports local recall at several values of `k`.

For the official evaluation, use the provided moulinette:

```bash
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 \
  --max_context_length 2000
```

### 5. Answer a single question

```bash
uv run python -m src answer \
  "How is the OpenAI-compatible server configured?" \
  --k 5
```

The answer generator retrieves sources with BM25 by default and passes their contents to `Qwen/Qwen3-0.6B`.

The first execution may download the model from Hugging Face.

### 6. Generate answers for search results

```bash
uv run python -m src answer_dataset \
  --student_search_results_path \
  data/output/search_results/UnansweredQuestions \
  --save_directory \
  data/output/search_results_and_answer/UnansweredQuestions
```

The generated output follows the `StudentSearchResultsAndAnswer` model.

Create the destination directory first if necessary:

```bash
mkdir -p \
  data/output/search_results_and_answer/UnansweredQuestions
```

---

## Output Format

### Search output

Dataset search produces JSON in this form:

```json
{
  "search_results": [
    {
      "question_id": "q1",
      "question": "How is the OpenAI-compatible server configured?",
      "retrieved_sources": [
        {
          "file_path": "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md",
          "first_character_index": 9867,
          "last_character_index": 10100,
          "chunk_id": 42
        }
      ]
    }
  ],
  "k": 5
}
```

### Answer output

Dataset answer generation produces:

```json
{
  "search_results": [
    {
      "question_id": "q1",
      "question": "How is the OpenAI-compatible server configured?",
      "retrieved_sources": [
        {
          "file_path": "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md",
          "first_character_index": 9867,
          "last_character_index": 10100,
          "chunk_id": 42
        }
      ],
      "answer": "The server is configured by..."
    }
  ],
  "k": 5
}
```

The additional `chunk_id` field is internal metadata permitted by the extensible output models.

---

## Makefile Commands

### Install dependencies

```bash
make install
```

### Run the CLI

```bash
make run
```

### Run with Python's debugger

```bash
make debug
```

### Run linting and type checking

```bash
make lint
```

This executes:

```bash
flake8 .
mypy . \
  --warn-return-any \
  --warn-unused-ignores \
  --ignore-missing-imports \
  --disallow-untyped-defs \
  --check-untyped-defs
```

### Strict linting

```bash
make lint-strict
```

### Run tests

```bash
make test
```

### Remove caches and generated data

```bash
make clean
```

Warning: the current `clean` rule removes generated indexes, outputs, and files under `data/datasets/`. Preserve any dataset files you need before running it.

---

## Local HTTP API

An experimental FastAPI interface is included as an additional feature.

Start it with:

```bash
make api
```

The interactive API documentation is normally available at:

```text
http://127.0.0.1:8000/docs
```

The mandatory and primary interface remains the Python Fire CLI.

---

## Performance Analysis

The project requirements are:

| Metric | Required result |
|---|---:|
| Full indexing time | At most 5 minutes |
| Retrieval time for 200 questions | At most 90 seconds |
| Documentation recall@5 | At least 80% |
| Code recall@5 | At least 50% |
| Maximum returned chunk length | 2,000 characters |

Performance should be measured on the target evaluation machine because CPU model speed, disk performance, corpus state, and cache state can significantly affect results.

Use the following commands to measure indexing and retrieval:

```bash
time uv run python -m src index --max_chunk_size 2000
```

```bash
time uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

Record measured results before submission:

| Configuration | Index time | 200-query search time | Docs recall@5 | Code recall@5 |
|---|---:|---:|---:|---:|
| BM25, 2,000-character chunks | 5s | 7.5s | 84% | 53.5% |
| Semantic, 2,000-character chunks | 2.33m | 30s | 64% | 36.4% |
| Hybrid, semantic weight 0.05 | 2.38m | 30s | 85% | 54.5% |

### Expected trade-offs

- **Larger chunks** preserve more context but can introduce irrelevant terms.
- **Smaller chunks** improve precision but can divide related definitions and explanations.
- **BM25** is fast and strong for exact identifiers.
- **Semantic retrieval** handles paraphrased questions better but is more expensive.
- **Hybrid retrieval** can improve coverage by combining both signals.
- **Caching** speeds up repeated queries but must be cleared for fair cold-run benchmarks.
- **Semantic indexing** adds model-encoding time to the indexing stage.

No benchmark values should be claimed until they have been measured on the actual corpus and machine.

---

## Design Decisions

### Store locations instead of duplicated source text

Chunk metadata stores a file path and character range rather than duplicating every chunk's text in `chunks.json`.

Advantages include:

- Smaller metadata files
- Exact traceability to the original source
- Easy validation of returned ranges
- Grounding answers in the original file content

### Preserve exact corpus paths

The evaluator compares file paths exactly. The project therefore stores paths as discovered under `data/raw/`, such as:

```text
data/raw/vllm-0.10.1/docs/...
```

### Use exclusive end indexes

Python slicing naturally uses an exclusive end index:

```python
content[first_character_index:last_character_index]
```

Using the same convention during indexing and reading avoids off-by-one differences.

### Use BM25 as the default

BM25 is fast, explainable, persistent, and particularly appropriate for code questions containing exact identifiers.

### Use rank fusion for hybrid retrieval

BM25 and cosine similarity scores are not directly comparable. Reciprocal Rank Fusion combines their rank positions without requiring score normalization.

### Use Pydantic at stage boundaries

Pydantic validates the data exchanged between indexing, retrieval, answer generation, dataset processing, and evaluation.

---

## Challenges Faced

### Preserving exact character ranges

Recursive chunking is useful, but retrieval evaluation requires exact source locations. Whitespace stripping was disabled, and the splitters were configured to return each chunk's starting position.

### Supporting both code and documentation

Code and prose do not have the same natural boundaries. Separate language-aware splitters were used for Python, Markdown, and RST files.

### Keeping chunks below the evaluator limit

The evaluator rejects any retrieved source wider than 2,000 characters. The index uses a configurable maximum size, and metadata is validated when the search engine loads.

### Combining incompatible score scales

BM25 scores and cosine similarities have different ranges. Weighted Reciprocal Rank Fusion avoids comparing their raw values directly.

### Avoiding unnecessary re-chunking

Large repositories are expensive to process repeatedly. Modification timestamps and file sizes are stored so unchanged files can reuse their existing chunks.

### Running models on CPU-only machines

Both required models can run on CPU, but model loading and generation may be slow. The embedding model uses batching, semantic embeddings are persisted, and repeated answers can be cached.

### Preventing hallucinations

The Qwen system prompt explicitly tells the model to:

- Use only retrieved sources
- Avoid inventing information
- Admit when the sources are insufficient

Retrieval quality remains essential because generation cannot recover information that was not retrieved.

---

## Error Handling

The CLI is designed to handle common invalid situations without exposing an unhandled traceback, including:

- Empty queries
- Non-positive `k`
- Missing index files
- Missing dataset files
- Malformed JSON
- Invalid retrieval modes
- Invalid semantic weights
- Unreadable source files
- Corrupted cache files
- Model-loading and generation errors

When a cache is malformed, the project reports a warning and rebuilds it rather than relying on invalid data.

---

## Resources

### RAG and information retrieval

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*  
  <https://arxiv.org/abs/2005.11401>
- Robertson and Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*  
  <https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf>
- Reciprocal Rank Fusion  
  <https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf>
- `bm25s` documentation  
  <https://github.com/xhluca/bm25s>

### Models and machine learning

- Qwen3-0.6B model page  
  <https://huggingface.co/Qwen/Qwen3-0.6B>
- Sentence Transformers documentation  
  <https://www.sbert.net/>
- `all-MiniLM-L6-v2` model page  
  <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
- Hugging Face Transformers documentation  
  <https://huggingface.co/docs/transformers/>

### Python libraries

- Pydantic documentation  
  <https://docs.pydantic.dev/>
- Python Fire documentation  
  <https://google.github.io/python-fire/>
- LangChain text splitters  
  <https://python.langchain.com/docs/concepts/text_splitters/>
- tqdm documentation  
  <https://tqdm.github.io/>
- uv documentation  
  <https://docs.astral.sh/uv/>
- FastAPI documentation  
  <https://fastapi.tiangolo.com/>

### AI usage

AI was used as a development assistant for:

- Reviewing the project subject and extracting mandatory requirements
- Discussing RAG architecture and retrieval terminology
- Suggesting documentation structure
- Reviewing possible edge cases
- Explaining BM25, semantic retrieval, and Reciprocal Rank Fusion
- Helping draft and improve this README

AI-generated suggestions were treated as proposals rather than authoritative answers. The implementation, behavior, tests, performance measurements, and final submitted content were reviewed and remain the responsibility of the project author.

---

## Testing Checklist

Before submission, verify the complete workflow:

```bash
uv sync
make lint
```

```bash
uv run python -m src index --max_chunk_size 2000
```

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

```bash
uv run python -m src evaluate \
  --student_search_results_path \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path \
  data/datasets/AnsweredQuestions/dataset_docs_public.json
```

```bash
mkdir -p \
  data/output/search_results_and_answer/UnansweredQuestions
```

```bash
uv run python -m src answer_dataset \
  --student_search_results_path \
  data/output/search_results/UnansweredQuestions \
  --save_directory \
  data/output/search_results_and_answer/UnansweredQuestions
```
