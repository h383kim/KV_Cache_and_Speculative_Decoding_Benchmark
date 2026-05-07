from llm_inference_lab.kv_cache.estimator import estimate_kv_cache_memory


def test_formula_gpt2_shape_fp16():
    # gpt2 small: 12 layers, 12 heads, head_dim 64, seq 1024, fp16, batch 1
    # 2 (K+V) * 12 * 1 * 1024 * 12 * 64 * 2 bytes = 37,748,736 bytes
    got = estimate_kv_cache_memory(
        num_layers=12,
        batch_size=1,
        seq_len=1024,
        num_kv_heads=12,
        head_dim=64,
        bytes_per_element=2,
    )
    assert got == 2 * 12 * 1 * 1024 * 12 * 64 * 2


def test_batch_scales_linearly():
    base = estimate_kv_cache_memory(12, 1, 512, 12, 64, 2)
    quad = estimate_kv_cache_memory(12, 4, 512, 12, 64, 2)
    assert quad == 4 * base


def test_seq_scales_linearly():
    short = estimate_kv_cache_memory(12, 1, 128, 12, 64, 2)
    long = estimate_kv_cache_memory(12, 1, 1024, 12, 64, 2)
    assert long == 8 * short


def test_gqa_smaller_than_mha():
    mha = estimate_kv_cache_memory(32, 1, 2048, 32, 128, 2)
    gqa = estimate_kv_cache_memory(32, 1, 2048, 8, 128, 2)
    assert gqa == mha // 4
    assert gqa < mha


def test_dtype_doubles_memory():
    fp16 = estimate_kv_cache_memory(12, 1, 1024, 12, 64, 2)
    fp32 = estimate_kv_cache_memory(12, 1, 1024, 12, 64, 4)
    assert fp32 == 2 * fp16
