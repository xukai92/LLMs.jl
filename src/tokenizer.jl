"""
Tokenizer for Llama-3.2 models.

Uses Python's `transformers` or `tokenizers` library via a subprocess.
This is the pragmatic approach — a pure Julia BPE implementation can come later.
"""

struct Tokenizer
    model_dir::String
    # Cache the Python script path
    _script::String
end

function Tokenizer(model_dir::String)
    # Write a minimal Python helper script
    script = joinpath(model_dir, "_julia_tokenizer.py")
    if !isfile(script)
        write(script, """
import sys, json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])

for line in sys.stdin:
    cmd = json.loads(line.strip())
    if cmd["action"] == "encode":
        ids = tokenizer.encode(cmd["text"], add_special_tokens=cmd.get("add_special", True))
        print(json.dumps({"ids": ids}), flush=True)
    elif cmd["action"] == "decode":
        text = tokenizer.decode(cmd["ids"], skip_special_tokens=cmd.get("skip_special", False))
        print(json.dumps({"text": text}), flush=True)
    elif cmd["action"] == "encode_chat":
        messages = cmd["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        print(json.dumps({"ids": ids, "text": text}), flush=True)
    elif cmd["action"] == "quit":
        break
""")
    end
    return Tokenizer(model_dir, script)
end

"""
    encode(tok, text; add_special=true) -> Vector{Int}

Encode text to token IDs (0-indexed).
"""
function encode(tok::Tokenizer, text::String; add_special::Bool=true)
    cmd = """{"action":"encode","text":$(JSON3.write(text)),"add_special":$(add_special)}"""
    result = _run_tokenizer_cmd(tok, cmd)
    return Int[id for id in result.ids]
end

"""
    decode(tok, ids; skip_special=false) -> String

Decode token IDs (0-indexed) to text.
"""
function decode(tok::Tokenizer, ids::Vector{Int}; skip_special::Bool=false)
    cmd = """{"action":"decode","ids":$(JSON3.write(ids)),"skip_special":$(skip_special)}"""
    result = _run_tokenizer_cmd(tok, cmd)
    return String(result.text)
end

"""
    encode_chat(tok, messages) -> (ids::Vector{Int}, text::String)

Apply chat template and encode.
messages: Vector of (role, content) pairs.
"""
function encode_chat(tok::Tokenizer, messages::Vector{Tuple{String, String}})
    msgs = [Dict("role" => r, "content" => c) for (r, c) in messages]
    cmd = """{"action":"encode_chat","messages":$(JSON3.write(msgs))}"""
    result = _run_tokenizer_cmd(tok, cmd)
    return Int[id for id in result.ids], String(result.text)
end

function _run_tokenizer_cmd(tok::Tokenizer, cmd::String)
    # Run as a one-shot command via Python
    py_code = """
import sys, json
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
cmd = json.loads(sys.argv[2])
if cmd["action"] == "encode":
    ids = tokenizer.encode(cmd["text"], add_special_tokens=cmd.get("add_special", True))
    print(json.dumps({"ids": ids}))
elif cmd["action"] == "decode":
    text = tokenizer.decode(cmd["ids"], skip_special_tokens=cmd.get("skip_special", False))
    print(json.dumps({"text": text}))
elif cmd["action"] == "encode_chat":
    messages = cmd["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    print(json.dumps({"ids": ids, "text": text}))
"""
    # Use uv to run with transformers installed
    output = read(`uv run --with transformers --with tokenizers python3 -c $py_code $(tok.model_dir) $cmd`, String)
    return JSON3.read(strip(output))
end
