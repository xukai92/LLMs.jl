"""
Launch the OpenAI-compatible inference server.

Usage:
    julia --project=. scripts/serve.jl [model_dir] [port]

Test with curl:
    curl http://localhost:8080/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model":"llama-3.2-3b-instruct","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50}'

Test streaming:
    curl http://localhost:8080/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model":"llama-3.2-3b-instruct","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50,"stream":true}'
"""

using LLMs

function main()
    model_dir = get(ARGS, 1,
        expanduser("~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct-4bit/snapshots/7f0dc925e0d0afb0322d96f9255cfddf2ba5636e"))
    port = parse(Int, get(ARGS, 2, "8080"))

    serve(model_dir; port=port)
end

main()
