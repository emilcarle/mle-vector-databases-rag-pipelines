# Vector Databases, Similarity Search and RAG Pipelines

A five-notebook learning path for building retrieval-augmented generation (RAG) systems from their retrieval foundation upward. Notebooks 01-03 expose chunking, vector similarity, and FAISS indexing. Notebooks 04-05 apply those decisions with LangChain, trained Sentence Transformers embeddings, and Qdrant local mode, then evaluate the same retrieval stack.

The notebooks share a small course-handbook corpus. Notebook 04 can call [Groq](https://console.groq.com/docs) or [Mistral](https://docs.mistral.ai/) through LangChain's OpenAI-compatible integration. Without a key, it returns a clearly labeled extractive example, not an LLM answer; the trained embeddings and vector database still run locally.

## Learning Objectives

By the end of this repository, you should be able to:

- Implement a document chunking strategy and explain its effect on retrieval quality and grounding evidence.
- Build a vector index with FAISS and perform approximate nearest neighbor search.
- Assemble a complete RAG pipeline that embeds documents, stores and retrieves vectors, and generates a cited answer from retrieved evidence.
- Evaluate retrieval quality with HitRate@k and MRR, then diagnose failures.

## Learning Path

| File / Folder | Description |
|---|---|
| [**01 - Chunking for Retrieval**](01-chunking-for-retrieval.ipynb) | Split source documents into retrieval units, compare overlap choices, and reason about evidence lost at chunk boundaries. |
| [**02 - Embeddings and Similarity Search**](02-embeddings-and-similarity-search.ipynb) | Build transparent normalized embeddings with NumPy, compute cosine similarity, and inspect top-k retrieval results. |
| [**03 - FAISS and ANN Indexes**](03-faiss-and-ann-indexes.ipynb) | Build exact and approximate FAISS indexes, map retrieved vector IDs back to metadata, and measure the accuracy-speed trade-off. |
| [**04 - RAG Pipeline and Grounded Generation**](04-rag-pipeline-and-grounded-generation.ipynb) | Use LangChain, trained MiniLM embeddings, and Qdrant local mode to retrieve evidence, compose a provider prompt, and validate citations. |
| [**05 - Retrieval Evaluation**](05-retrieval-evaluation.ipynb) | Compare chunking candidates with the same MiniLM/Qdrant stack, measure HitRate@k and MRR, and check separate validation questions. |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**assets**](assets/) | Visual aids referenced in the repository. |
| [**data**](data/) | Course handbook corpus shared across notebooks. |
| [**.env.example**](.env.example) | Provider configuration template. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a
  **placeholder**. Replace it, including the `< >` brackets, with your own
  value. For example, `cd <repo-name>` becomes `cd mle-vector-databases-rag-pipelines`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

This installs all dependencies and creates a virtual environment in `.venv/`.

```bash
cd <repo-name>
uv sync
```

Notebooks 04-05 download `sentence-transformers/all-MiniLM-L6-v2` on first use, then reuse the Hugging Face cache. This needs internet access initially but no API key or GPU. Qdrant runs in memory, so Docker and a database account are not required; restarting the kernel rebuilds its small collection.

---

### 5. Set Up Environment Variables (Optional)

For real LLM generation in notebook 04, copy the example file and fill in your API key:

```bash
cp .env.example .env
```

Open `.env` and set `LLM_PROVIDER` to `groq` or `mistral`, then add the matching key:

- **Groq (primary):** Create a key at [console.groq.com/keys](https://console.groq.com/keys) and set `GROQ_API_KEY`.
- **Mistral (alternative):** Create a key following the [Mistral API key documentation](https://docs.mistral.ai/admin/security-access/api-keys) and set `MISTRAL_API_KEY`.

All five notebooks run without an API key. Notebook 04's extractive example copies the best evidence with a citation; a real generation call is necessary to evaluate how an LLM uses that evidence or abstains. Notebook 05 evaluates retrieval without calling an LLM.

> [!CAUTION]
> Never commit `.env` to version control. It contains your personal API keys.

---

### 6. Open the Repository in VS Code

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open a notebook and select the Python environment created by `uv sync` as the kernel.

## API Usage Notes

- Provider free tiers, rate limits, model names, and billing rules can change. Check the provider console before running live experiments.

## Professional Tooling Context

The learning path moves from visible mechanics to one concrete integration:

- **01-03:** NumPy hash vectors expose the retrieval mechanics and FAISS demonstrates exact and approximate search. These are not trained semantic embeddings.
- **04-05:** Sentence Transformers supplies learned embeddings, Qdrant stores vectors and metadata locally, and LangChain connects the components. Local in-memory search does not demonstrate server deployment or ANN scalability.
- **Later extensions:** LlamaIndex is an alternative RAG framework. Evaluation libraries such as Ragas and TruLens extend retrieval checks with groundedness and answer-quality evaluation.
