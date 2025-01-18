import jax

jax.distributed.initialize()

if jax.process_index() == 0:
  print(jax.device_count())