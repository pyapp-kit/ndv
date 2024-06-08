from __future__ import annotations

try:
    import jax.numpy as jnp
except ImportError:
    raise ImportError("Please install jax to run this example")
import ndv

jax_arr = jnp.asarray(ndv.data.nd_sine_wave())
v = ndv.imshow(jax_arr)
