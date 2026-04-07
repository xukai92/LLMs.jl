"""
OpenAI-compatible HTTP API server with SSE streaming.

Endpoints:
  POST /v1/chat/completions — chat completion (streaming and non-streaming)
  GET  /v1/models           — list available models
  GET  /health              — health check

Uses HTTP.jl stream handlers for SSE streaming support.
"""

using HTTP
using JSON3
using Dates: now, UTC, datetime2unix

# ── Request/Response types ──

struct ChatMessage
    role::String
    content::String
end

struct CompletionRequest
    model::String
    messages::Vector{ChatMessage}
    max_tokens::Int
    temperature::Float64
    stream::Bool
end

function parse_completion_request(body::String)
    req = JSON3.read(body)
    messages = ChatMessage[]
    for msg in req.messages
        push!(messages, ChatMessage(String(msg.role), String(msg.content)))
    end
    CompletionRequest(
        get(req, :model, "llama-3.2-3b-instruct"),
        messages,
        get(req, :max_tokens, 256),
        Float64(get(req, :temperature, 0.0)),
        Bool(get(req, :stream, false)),
    )
end

# ── Chat template ──

"""
Apply Llama 3 Instruct chat template.
"""
function apply_chat_template(messages::Vector{ChatMessage})
    parts = String["<|begin_of_text|>"]
    for msg in messages
        push!(parts, "<|start_header_id|>$(msg.role)<|end_header_id|>\n\n")
        push!(parts, msg.content)
        push!(parts, "<|eot_id|>")
    end
    push!(parts, "<|start_header_id|>assistant<|end_header_id|>\n\n")
    return join(parts)
end

# ── Inference Engine ──

mutable struct InferenceEngine
    model::LlamaModel
    tokenizer::Tokenizer
    prefix_cache::PrefixCache
    model_id::String
    lock::ReentrantLock  # serialize inference requests
end

function InferenceEngine(model_dir::String; model_id::String="llama-3.2-3b-instruct",
                         cache_tokens::Int=16384)
    model = load_llama_model(model_dir)
    tokenizer = Tokenizer(model_dir)
    pcache = PrefixCache(model.config; max_tokens=cache_tokens)
    InferenceEngine(model, tokenizer, pcache, model_id, ReentrantLock())
end

"""
Generate tokens one at a time, calling `on_token` for each.
on_token(token_text::String, is_done::Bool)
"""
function generate_streaming(on_token::Function, engine::InferenceEngine,
                           prompt_ids::Vector{Int}, max_tokens::Int)
    model = engine.model
    config = model.config
    pcache = engine.prefix_cache
    eos_ids = config.eos_token_ids

    total_len = length(prompt_ids) + max_tokens + 16
    cache = KVCache(config; max_seq_len=total_len)

    matched_len, _, kv_segments = prefix_match(pcache, prompt_ids)
    if matched_len > 0
        restore_kv!(cache, pcache, kv_segments)
        remaining_ids = prompt_ids[matched_len+1:end]
    else
        remaining_ids = prompt_ids
    end

    if !isempty(remaining_ids)
        prompt_gpu = MtlArray(Int32.(remaining_ids))
        logits = forward(model, prompt_gpu, cache)
        Metal.synchronize()
    else
        last_tok = MtlArray(Int32[prompt_ids[end]])
        cache.seq_len -= 1
        logits = forward(model, last_tok, cache)
        Metal.synchronize()
    end

    next_token = argmax_last_col_cpu(logits)
    token_text = decode(engine.tokenizer, [Int(next_token)]; skip_special=true)
    is_eos = Int(next_token) in eos_ids
    on_token(token_text, is_eos)

    if !is_eos
        for step in 2:max_tokens
            token_gpu = MtlArray(Int32[next_token])
            logits = forward(model, token_gpu, cache)
            Metal.synchronize()

            next_token = argmax_last_col_cpu(logits)
            token_text = decode(engine.tokenizer, [Int(next_token)]; skip_special=true)
            is_eos = Int(next_token) in eos_ids
            on_token(token_text, is_eos)

            if is_eos
                break
            end
        end
    end

    insert_prefix!(pcache, prompt_ids, cache, matched_len)
end

# ── SSE formatting ──

function make_chunk_json(id::String, model::String, content::String;
                         finish_reason::Union{String, Nothing}=nothing)
    delta = finish_reason !== nothing ?
        Dict{String, Any}() :
        Dict{String, Any}("content" => content, "role" => "assistant")
    chunk = Dict{String, Any}(
        "id" => id,
        "object" => "chat.completion.chunk",
        "created" => Int(floor(datetime2unix(now(UTC)))),
        "model" => model,
        "choices" => [Dict{String, Any}(
            "index" => 0,
            "delta" => delta,
            "finish_reason" => finish_reason,
        )]
    )
    return JSON3.write(chunk)
end

function make_completion_json(id::String, model::String, content::String,
                              prompt_tokens::Int, completion_tokens::Int;
                              finish_reason::String="stop")
    resp = Dict{String, Any}(
        "id" => id,
        "object" => "chat.completion",
        "created" => Int(floor(datetime2unix(now(UTC)))),
        "model" => model,
        "choices" => [Dict{String, Any}(
            "index" => 0,
            "message" => Dict("role" => "assistant", "content" => content),
            "finish_reason" => finish_reason,
        )],
        "usage" => Dict(
            "prompt_tokens" => prompt_tokens,
            "completion_tokens" => completion_tokens,
            "total_tokens" => prompt_tokens + completion_tokens,
        ),
    )
    return JSON3.write(resp)
end

# ── Stream handler ──

function handle_stream(engine::InferenceEngine, stream::HTTP.Stream)
    method = stream.message.method
    target = stream.message.target

    # CORS
    HTTP.setheader(stream, "Access-Control-Allow-Origin" => "*")
    HTTP.setheader(stream, "Access-Control-Allow-Methods" => "GET, POST, OPTIONS")
    HTTP.setheader(stream, "Access-Control-Allow-Headers" => "Content-Type, Authorization")

    if method == "OPTIONS"
        HTTP.setstatus(stream, 204)
        HTTP.startwrite(stream)
        return
    end

    if target == "/health" && method == "GET"
        HTTP.setheader(stream, "Content-Type" => "application/json")
        HTTP.setstatus(stream, 200)
        HTTP.startwrite(stream)
        write(stream, """{"status":"ok"}""")
        return
    end

    if target == "/v1/models" && method == "GET"
        HTTP.setheader(stream, "Content-Type" => "application/json")
        HTTP.setstatus(stream, 200)
        HTTP.startwrite(stream)
        resp = Dict("object" => "list", "data" => [Dict(
            "id" => engine.model_id, "object" => "model",
            "created" => 0, "owned_by" => "local")])
        write(stream, JSON3.write(resp))
        return
    end

    if target == "/v1/chat/completions" && method == "POST"
        body = String(read(stream))
        creq = try
            parse_completion_request(body)
        catch e
            HTTP.setheader(stream, "Content-Type" => "application/json")
            HTTP.setstatus(stream, 400)
            HTTP.startwrite(stream)
            write(stream, JSON3.write(Dict("error" => Dict(
                "message" => "Invalid request: $(sprint(showerror, e))",
                "type" => "invalid_request_error"))))
            return
        end

        prompt_text = apply_chat_template(creq.messages)
        request_id = "chatcmpl-" * string(rand(UInt64), base=16)

        # Serialize inference access
        lock(engine.lock) do
            prompt_ids = encode(engine.tokenizer, prompt_text; add_special=false)
            n_prompt = length(prompt_ids)

            if creq.stream
                HTTP.setheader(stream, "Content-Type" => "text/event-stream")
                HTTP.setheader(stream, "Cache-Control" => "no-cache")
                HTTP.setheader(stream, "X-Accel-Buffering" => "no")
                HTTP.setstatus(stream, 200)
                HTTP.startwrite(stream)

                n_completion = 0
                generate_streaming(engine, prompt_ids, creq.max_tokens) do text, is_done
                    n_completion += 1
                    if is_done
                        chunk = make_chunk_json(request_id, creq.model, "";
                            finish_reason="stop")
                        write(stream, "data: $chunk\n\n")
                        write(stream, "data: [DONE]\n\n")
                    else
                        chunk = make_chunk_json(request_id, creq.model, text)
                        write(stream, "data: $chunk\n\n")
                    end
                end
            else
                generated_text = IOBuffer()
                n_completion = 0

                generate_streaming(engine, prompt_ids, creq.max_tokens) do text, is_done
                    n_completion += 1
                    if !is_done
                        write(generated_text, text)
                    end
                end

                content = String(take!(generated_text))
                resp = make_completion_json(request_id, creq.model, content,
                    n_prompt, n_completion)

                HTTP.setheader(stream, "Content-Type" => "application/json")
                HTTP.setstatus(stream, 200)
                HTTP.startwrite(stream)
                write(stream, resp)
            end
        end
        return
    end

    # 404
    HTTP.setheader(stream, "Content-Type" => "application/json")
    HTTP.setstatus(stream, 404)
    HTTP.startwrite(stream)
    write(stream, JSON3.write(Dict("error" => Dict(
        "message" => "Not found: $target", "type" => "not_found"))))
end

# ── Server entry point ──

"""
    serve(model_dir; host="127.0.0.1", port=8080, cache_tokens=16384)

Start the OpenAI-compatible inference server.
"""
function serve(model_dir::String; host::String="127.0.0.1", port::Int=8080,
               cache_tokens::Int=16384)
    println("Initializing inference engine...")
    engine = InferenceEngine(model_dir; cache_tokens=cache_tokens)

    println("Server starting on http://$host:$port")
    println("\nEndpoints:")
    println("  POST http://$host:$port/v1/chat/completions")
    println("  GET  http://$host:$port/v1/models")
    println("  GET  http://$host:$port/health")
    println("\nTest with:")
    println("  curl http://$host:$port/v1/chat/completions \\")
    println("    -H 'Content-Type: application/json' \\")
    println("    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":50}'")
    println("\nReady for requests.\n")

    HTTP.serve(host, port; stream=true) do stream
        handle_stream(engine, stream)
    end
end
