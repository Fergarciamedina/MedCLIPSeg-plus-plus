# Bundled vision operators

Location `shared/medclipseg/biomedclip/layers/` · [Up to biomedclip/](../BIOMEDCLIP_COMPONENTS.md)

The upstream BiomedCLIP path retains these reusable layer files. Grouping them by the operations they implement helps locate a relevant module without tracing every import. The files contain their own upstream credit notices where supplied. [BiomedCLIP components](../BIOMEDCLIP_COMPONENTS.md) · [Model attribution](../../../../ATTRIBUTION.md).

| Operation | Files | Responsibility |
| --- | --- | --- |
| Package and configuration | [__init__.py](__init__.py) · [config.py](config.py) | Exports layers and controls scriptability, fused attention and other operator settings. |
| Layer factories | [create_act.py](create_act.py) · [create_attn.py](create_attn.py) · [create_conv2d.py](create_conv2d.py) · [create_norm.py](create_norm.py) · [create_norm_act.py](create_norm_act.py) | Select activation, attention, convolution or normalization from a configuration. |
| Activations and regularization | [activations.py](activations.py) · [activations_me.py](activations_me.py) · [drop.py](drop.py) · [patch_dropout.py](patch_dropout.py) · [layer_scale.py](layer_scale.py) | Provides activation variants, stochastic regularization and learned residual scaling. |
| Attention blocks | [attention2d.py](attention2d.py) · [bottleneck_attn.py](bottleneck_attn.py) · [halo_attn.py](halo_attn.py) · [non_local_attn.py](non_local_attn.py) · [lambda_layer.py](lambda_layer.py) | Implements spatial attention or context mixing with different receptive fields and computational structures. |
| Channel and context recalibration | [cbam.py](cbam.py) · [eca.py](eca.py) · [gather_excite.py](gather_excite.py) · [global_context.py](global_context.py) · [squeeze_excite.py](squeeze_excite.py) | Reweights features using spatial context, pooled statistics or interactions among channels. |
| Attention pooling | [attention_pool.py](attention_pool.py) · [attention_pool2d.py](attention_pool2d.py) | Summarizes spatial tokens through learned queries or attention over a two-dimensional grid. |
| Convolution blocks | [cond_conv2d.py](cond_conv2d.py) · [conv2d_same.py](conv2d_same.py) · [conv_bn_act.py](conv_bn_act.py) · [mixed_conv2d.py](mixed_conv2d.py) · [selective_kernel.py](selective_kernel.py) · [separable_conv.py](separable_conv.py) · [split_attn.py](split_attn.py) · [std_conv.py](std_conv.py) | Supports conditional kernels, same-padding, combined convolution blocks, multiple branches and standardized weights. |
| Pooling and downsampling | [adaptive_avgmax_pool.py](adaptive_avgmax_pool.py) · [blur_pool.py](blur_pool.py) · [median_pool.py](median_pool.py) · [pool2d_same.py](pool2d_same.py) · [test_time_pool.py](test_time_pool.py) | Provides average, maximum, blur or median downsampling and test-time pooling. |
| Normalization | [evo_norm.py](evo_norm.py) · [fast_norm.py](fast_norm.py) · [filter_response_norm.py](filter_response_norm.py) · [grn.py](grn.py) · [inplace_abn.py](inplace_abn.py) · [norm.py](norm.py) · [norm_act.py](norm_act.py) · [split_batchnorm.py](split_batchnorm.py) | Provides normalization variants, including combined activation and split batch-statistic paths. |
| Patches and geometry | [format.py](format.py) · [grid.py](grid.py) · [hybrid_embed.py](hybrid_embed.py) · [interpolate.py](interpolate.py) · [padding.py](padding.py) · [patch_embed.py](patch_embed.py) · [space_to_depth.py](space_to_depth.py) | Handles tensor layouts, patch projection or resampling, coordinate grids, padding and spatial-to-channel reshaping. |
| Position | [pos_embed.py](pos_embed.py) · [pos_embed_rel.py](pos_embed_rel.py) · [pos_embed_sincos.py](pos_embed_sincos.py) | Handles resizing and constructing absolute, relative and sinusoidal position encodings. |
| Projections and heads | [classifier.py](classifier.py) · [linear.py](linear.py) · [ml_decoder.py](ml_decoder.py) · [mlp.py](mlp.py) | Defines linear or feed-forward projections and output classification heads. |
| Initialization | [weight_init.py](weight_init.py) | Initializes model parameter tensors for the bundled layers. |
| Shared helpers | [helpers.py](helpers.py) · [trace_utils.py](trace_utils.py) · [typing.py](typing.py) | Normalizes dimensions, supports symbolic tracing and defines common layer argument types. |
