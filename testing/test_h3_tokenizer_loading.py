"""H3 tokenizer-only repositories must not require a model config.json."""
import ast
from pathlib import Path
from types import SimpleNamespace


def test_h3_passes_explicit_qwen2_type_to_tokenizer_and_processor(monkeypatch):
    # Execute the real loader's processor-loading prefix without importing the
    # GPU-only model registry or constructing a 32B text encoder.
    source = Path(__file__).resolve().parents[1] / (
        "extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py"
    )
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_load_text_encoder")
    stop = next(i for i, n in enumerate(method.body)
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "te_path" for t in n.targets))
    method.body = method.body[:stop] + [ast.parse("return tokenizer, processor").body[0]]
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    calls = []

    def loader(kind):
        def load(repo, **kwargs):
            # The Qwen2 selector bypasses AutoConfig's absent-config probe.
            assert kwargs["tokenizer_type"] == "qwen2"
            calls.append((kind, repo, kwargs["subfolder"]))
            return kind
        return load

    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", loader("tokenizer"))
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", loader("processor"))
    namespace = {"ORIGINAL_REPO": "MiniMaxAI/MiniMax-H3"}
    exec(compile(module, str(source), "exec"), namespace)
    assert namespace["_load_text_encoder"](SimpleNamespace()) == ("tokenizer", "processor")
    assert calls == [
        ("tokenizer", "MiniMaxAI/MiniMax-H3", "FL2VA/tokenizer"),
        ("processor", "MiniMaxAI/MiniMax-H3", "FL2VA/processor"),
    ]
