# Turning externally hosted files into an addressable model

The [model lift contract](MODEL_LIFT_CONTRACT.md) develops this address scheme into a process for compiling and qualifying a virtual transition.

A model page can be the starting point for a calculation. A worker that follows the page to an immutable checkpoint can locate one learned quantity, read its bytes, and interpret them under the model’s equations. VL-I asks a further question: **for the behavior we want to reproduce, which quantities must the worker actually observe and which calculations can it omit?**

## From a page to one learned value

Consider the request “read the embedding that Qwen assigns to token 63204.” The checkpoint index places the embedding tensor in shard 3 of 18. The shard header declares 248,320 rows of 5,120 BF16 values. The worker can therefore name a row before knowing any of its values. On a pinned revision, the 10,240 bytes for this row occupy the half-open file interval $[647209112,647219352)$. We retrieved that interval from the public host and recorded its SHA-256. This is a trained embedding at the input to the text decoder. A contextual representation of the same token later in a prompt also depends on the preceding tokens and intervening layers. [Pinned Qwen index](https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/model.safetensors.index.json) · [Worked byte address and reproduction command in VL-64](https://github.com/Kier73/VL-64/blob/research/model-readout/docs/HOSTED_MODEL_ADDRESSES.md)

There are three useful addresses in that example. A **source address** names the repository, revision, shard and byte interval. A **model address** names the tensor, its coordinates, and the rules that decode its bytes. A **causal address** names the model and input history, position, layer, and requested observation. A worker can reuse an immutable embedding row across prompts. Reusing an attention result also requires the relevant causal state and numerical conventions to agree.

The **Model Ground** connects those addresses. The file index and headers establish ownership, shapes, encodings and extents. A family-specific interpretation supplies operator semantics, dependencies and state updates. Tokenizer and prompt-format identities matter when input is text. A screenshot can reveal a repository name or point to a revision. Following that lead to a pinned artifact gives another worker a way to repeat the observation.

## When navigation can replace computation

Fix a requested behavior $Y$ and let $E$ be everything the VL has observed, including retained summaries. An exact answer is determined when every checkpoint and state compatible with $E$ gives the same $Y$. This gives the interpreter a stopping rule. If the evidence still admits two different answers, the next read should distinguish those possibilities. If the answer is already determined, the open engineering question is how cheaply the VL can realize it. [Formal criterion and a trained readout experiment](WHAT_A_GLANCE_MUST_PRESERVE.md)

**Qwen offers a question about recurrent memory.** For a gated-delta head, fix the quantities supplied by the current token so that its update takes the affine form $S_t=A_tS_{t-1}+B_t$. A requested projection obeys $q^\top S_t=(A_t^\top q)^\top S_{t-1}+q^\top B_t$. The VL can carry the question backward through an update. A useful experiment must count the work of obtaining the token-dependent gates and preserving the state needed by later tokens. The repository already has a numerical reference for two original tokens through eight trained layers, with an independent check on layer 7. [Qwen temporal experiment](QUALIFIED_ORIGINAL_TEMPORAL_ATTENTION.md)

**DeepSeek offers a question about routing.** At a learned routing layer, ask which six of 384 experts receive a text token. Suppose each expert’s final selection score lies in a verified interval $[L_e,U_e]$. A six-expert candidate set $K$ is settled when

$$
\min_{e\in K}L_e > \max_{j\notin K}U_j.
$$

Several VL-64 workers could maintain bounds for disjoint expert rows. A coordinator could request further observations only where the sixth-place boundary is still uncertain. A tie requires the reference implementation’s tie behavior or a more precise observation. DeepSeek uses a bias when selecting experts and the unbiased scores when weighting their contributions. Identifying the six winners therefore settles a precise intermediate question. Running the layer also requires their mixture weights, the expert outputs, the shared expert and the surrounding state updates. [Pinned DeepSeek router implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/inference/model.py)

## A benchmark that could change the execution path

The page-navigation experiment follows actual search results and pinned directory entries to these tensor coordinates. It also compares a full SmolLM decoder observation with an independent reference. [From a model webpage to a Virtual Layer operation](ECPU_MODEL_PAGE_NAVIGATION.md)

The [first router-bound experiment](DEEPSEEK_ROUTER_BOUNDS.md) exercised this stopping rule on the pinned gate with three supplied embedding probes. It records both agreement and the cost of late elimination.

Pin one DeepSeek revision, one routed layer, the numerical convention, and a held-out set of genuine incoming text states. Give the workers the tensor map and identify every summary they obtained in advance. Have them commit the six IDs before a separate implementation reveals the route. Record disagreements, ties, fallbacks, bytes read, arithmetic, communication, latency, and cold construction costs. Compare against the checkpoint’s ordinary sparse router on the same states. The repository’s earlier DeepSeek evidence belongs to revision 1bae0e1; a benchmark on revision 08c8657 needs its own source and numerical checks. [Earlier evidence and scope](LARGE_MODEL_PARITY_FRONTIER.md)

Once a route can be determined economically from a supplied state, move the experiment one dependency upstream and include the cost of producing that state. For generation, repeat the test after feeding the selected token back into the model. The stronger achievement is a Ground that preserves both the present answer and enough state to answer the next question.
