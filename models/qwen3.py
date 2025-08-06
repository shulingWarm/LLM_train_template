from transformers.models.qwen3 import Qwen3ForCausalLM, Qwen3Config
from transformers import AutoConfig, AutoModelForCausalLM

class CustomQwen3Config(Qwen3Config):
    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            tie_word_embeddings,
            rope_theta,
            rope_scaling,
            attention_bias,
            use_sliding_window,
            sliding_window,
            max_window_layers,
            attention_dropout,
            **kwargs
        )

class CustomQwen3(Qwen3ForCausalLM):
    config_class = CustomQwen3Config

    def __init__(self, config):
        super().__init__(config)

# 注册模型本体
def register_model():
    # AutoConfig.register("qwen3", MyCustomModel)  # 别名可选
    AutoModelForCausalLM.register(CustomQwen3.config_class, CustomQwen3)