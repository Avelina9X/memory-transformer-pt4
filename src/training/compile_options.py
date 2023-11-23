TORCH_COMPILE_OPTIONS = {
    'options': {
        'triton.cudagraphs': False,
        'triton.cudagraph_trees': True,
        'triton.autotune_pointwise': True,
        'triton.autotune_cublasLt': True,
        'shape_padding': True,
        
        'max_autotune': True,
        'max_autotune_pointwise': True,
        'max_autotune_gemm': True,
        
        'allow_buffer_reuse': True,
        'epilogue_fusion': True,
    },
    'fullgraph': False,
}