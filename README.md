The folder is named decoder as requested, but it intentionally contains no generative decoder. The CODEFEST stage prohibits decoder architectures during indexing and retrieval, and this project only needs an encoder to transform source fragments and questions into comparable vectors. The single encoder.py file is therefore the complete component. It does not extract documents, create chunks, build a FAISS index, rank results, call an API, or generate text.

The architecture is intfloat/multilingual-e5-small, a multilingual retrieval encoder with approximately 118 million parameters, twelve transformer layers, and a 384-dimensional output. Spanish, English, and Portuguese text is represented in the same semantic space, so a question in one language can retrieve a related passage in another language without translation. The implementation uses the portable ONNX O4 graph through FastEmbed instead of PyTorch. FastEmbed performs attention-aware mean pooling, and encoder.py normalizes every output to Euclidean length one so a later dot product is cosine similarity. The public model is downloaded without an API key and can run entirely offline after the first execution.

E5 is an asymmetric retrieval model, so questions and source passages must not be encoded as identical raw input. Encoder.encode_queries automatically adds the literal query prefix, while Encoder.encode_passages automatically adds the literal passage prefix. These English prefixes are part of the model's training protocol and remain unchanged when the content is Spanish or Portuguese. Encoder.count_tokens uses the checkpoint's exact tokenizer, allowing the caller to keep each complete input below the model limit of 512 subword tokens. The main project reserves additional space and normally limits passage input to 448 tokens.

Step one is to open a terminal in the project root, which is the directory containing the decoder folder. On Windows PowerShell, the command can be written as follows.

```powershell
Set-Location "C:\Users\alejo\Desktop\ad astra"
```

On macOS or Linux, change to the equivalent directory containing the project. All later commands assume that the current directory contains decoder/encoder.py.

Step two is to verify Python. Python 3.11 or 3.12 is recommended because binary wheels for ONNX Runtime and NumPy are readily available. On Windows, run the following command.

```powershell
py -3.12 --version
```

On macOS or Linux, run `python3.12 --version`. A displayed Python 3.12 version means the interpreter is ready. If Python 3.12 is unavailable, Python 3.11 can be substituted consistently in every command.

Step three is to create an isolated virtual environment. This prevents the encoder dependencies from modifying the system Python installation. On Windows PowerShell, run the following command from the project root.

```powershell
py -3.12 -m venv .venv
```

On macOS or Linux, run `python3.12 -m venv .venv`. Activation is optional because the commands below call the environment's interpreter directly. This is particularly useful on Windows computers where PowerShell blocks Activate.ps1.

Step four is to upgrade pip and install the smallest direct dependency set needed by this standalone file. On Windows PowerShell, run the following commands.

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "fastembed>=0.7,<0.8" "numpy>=1.26,<3"
```

On macOS or Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`. FastEmbed installs ONNX Runtime, Hugging Face Hub, tokenizers, and its other internal requirements automatically. They should not be installed separately unless troubleshooting a package-resolution problem. No PyTorch, Transformers, CUDA package, vector database, API client, or server is required.

Step five is the only stage that normally needs internet access. Construct the encoder once with offline mode disabled so FastEmbed can download the public O4 graph and tokenizer into .model_cache. No Hugging Face token or API key is needed. On Windows PowerShell, run the following command.

```powershell
.\.venv\Scripts\python.exe -c "from decoder.encoder import Encoder; Encoder(cache_dir='.model_cache', threads=2); print('Model downloaded and loaded successfully')"
```

On macOS or Linux, use the same Python expression with `.venv/bin/python -c`. The command can take several minutes on the first execution. Completion creates .model_cache and prints `Model downloaded and loaded successfully`. A warning about an unauthenticated Hugging Face request only describes rate limiting and does not mean that a key is required.

Step six is normal program usage. Import Encoder, create one instance, and reuse it for every batch rather than recreating the model for each string. Pass raw fragment text to encode_passages and raw questions to encode_queries because the methods add the E5 prefixes themselves. The following minimal program encodes two passages and one query, then computes cosine scores using matrix multiplication.

```python
from decoder.encoder import Encoder

encoder = Encoder(cache_dir=".model_cache", threads=2, offline=True)
passage_vectors = encoder.encode_passages(
    [
        "Artificial intelligence supports threat detection in defense operations.",
        "This recipe explains how to bake bread in a home oven.",
    ]
)
query_vector = encoder.encode_queries(
    ["inteligencia artificial para detectar amenazas militares"]
)
scores = passage_vectors @ query_vector[0]
print(scores)
```

The returned passage matrix has shape `(2, 384)`, the query matrix has shape `(1, 384)`, and the first score should be greater than the second because the first passage is semantically related to the Spanish question. Since every row is normalized, no separate cosine-similarity library is required.

Step seven is to run a complete automated smoke test without creating another script. The following PowerShell command sends a temporary test program directly to Python. It verifies the two output shapes, float32 representation, unit-length normalization, exact-tokenizer access, a finite cosine score, and a sensible cross-language ranking.

```powershell
@'
import numpy as np
from decoder.encoder import Encoder

encoder = Encoder(cache_dir=".model_cache", threads=2, offline=True)
passages = encoder.encode_passages([
    "Artificial intelligence supports threat detection in defense operations.",
    "This recipe explains how to bake bread in a home oven.",
])
queries = encoder.encode_queries([
    "inteligencia artificial para detectar amenazas militares"
])
scores = passages @ queries[0]

assert passages.shape == (2, 384)
assert queries.shape == (1, 384)
assert passages.dtype == np.float32
assert queries.dtype == np.float32
assert np.allclose(np.linalg.norm(passages, axis=1), 1.0, atol=1e-5)
assert np.allclose(np.linalg.norm(queries, axis=1), 1.0, atol=1e-5)
assert encoder.count_tokens("passage: test text") > 0
assert np.isfinite(scores).all()
assert scores[0] > scores[1]

print("Encoder test passed")
print("Passage shape:", passages.shape)
print("Query shape:", queries.shape)
print("Cosine scores:", scores.tolist())
'@ | .\.venv\Scripts\python.exe -
```

On macOS or Linux, the equivalent smoke test can be run with the following command.

```bash
.venv/bin/python - <<'PY'
import numpy as np
from decoder.encoder import Encoder

encoder = Encoder(cache_dir=".model_cache", threads=2, offline=True)
passages = encoder.encode_passages([
    "Artificial intelligence supports threat detection in defense operations.",
    "This recipe explains how to bake bread in a home oven.",
])
queries = encoder.encode_queries([
    "inteligencia artificial para detectar amenazas militares"
])
scores = passages @ queries[0]

assert passages.shape == (2, 384)
assert queries.shape == (1, 384)
assert passages.dtype == np.float32
assert queries.dtype == np.float32
assert np.allclose(np.linalg.norm(passages, axis=1), 1.0, atol=1e-5)
assert np.allclose(np.linalg.norm(queries, axis=1), 1.0, atol=1e-5)
assert encoder.count_tokens("passage: test text") > 0
assert np.isfinite(scores).all()
assert scores[0] > scores[1]

print("Encoder test passed")
print("Passage shape:", passages.shape)
print("Query shape:", queries.shape)
print("Cosine scores:", scores.tolist())
PY
```

A successful run begins with `Encoder test passed`, reports passage shape `(2, 384)` and query shape `(1, 384)`, and prints two cosine scores with the first larger than the second. The precise scores can vary slightly across ONNX Runtime versions and processors, so the test checks their ordering rather than requiring fixed numerical values.

Step eight verifies truly offline operation. Disconnect the computer from the network, or simply run the following command after the cache has been created. The offline flag tells both FastEmbed and Hugging Face Hub to reject network access and use local files only.

```powershell
.\.venv\Scripts\python.exe -c "from decoder.encoder import Encoder; e=Encoder(cache_dir='.model_cache', threads=2, offline=True); print(e.encode_queries(['prueba sin internet']).shape)"
```

The expected output is `(1, 384)`. If the command raises LocalEntryNotFoundError, the first online download did not finish or a different cache directory was supplied. Run step five again with internet access, then repeat the offline check using the same cache path. For a different fully disconnected computer, copy decoder/encoder.py and the complete .model_cache directory, install the two Python packages from locally available wheels, and construct Encoder with offline=True.

The threads argument controls CPU usage rather than correctness. Two threads are a conservative default for a modest computer; four can improve throughput on a typical laptop. Batch size defaults to sixteen and can be lowered when memory is constrained. An empty input returns a float32 array with shape `(0, 384)`. The Encoder instance should remain alive while processing multiple batches because loading the ONNX graph repeatedly wastes time and memory.
