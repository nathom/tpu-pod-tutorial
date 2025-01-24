import jax

jax.distributed.initialize()

if __name__ == "__main__" and jax.process_index() == 0:
  print(jax.device_count())
