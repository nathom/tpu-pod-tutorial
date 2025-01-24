import jax
import pdb
import jax.numpy as jnp
from jax import random

jax.distributed.initialize()
def test_jax_functionality():

    key = random.PRNGKey(0)
    x = random.normal(key, (1000,))
    y = jnp.sin(x)
    return y.mean()

if __name__ == "__main__" and jax.process_index() == 1:
  print(jax.device_count())
  pdb.set_trace()
  print(test_jax_functionality())
