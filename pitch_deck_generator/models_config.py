models_config = {
 "marin/marin-8b-instruct": {
        "params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1024
        },
        "extract_func": lambda chunk: chunk.choices[0].delta.content
    },
    "deepseek-ai/deepseek-r1": {
        "params": {
            "temperature": 0.6,
            "top_p": 0.7,
            "max_tokens": 4096
        },
        "extract_func": lambda chunk: chunk.choices[0].delta.content
    },
    "qwen/qwen3-235b-a22b": {
        "params": {
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 8192,
            "extra_body": {"chat_template_kwargs": {"thinking": True}}
        },
        "extract_func": lambda chunk: (
            getattr(chunk.choices[0].delta, "reasoning_content", "") or ""
        ) + (chunk.choices[0].delta.content or "")
    }
}