# tpu-pod-tutorial

This tutorial will get you started with multi-host VM
TPU pods.

## Setup this tutorial

First, fork this repo on GitHub. Then, clone your fork
onto one of the hosts on the TPU VM.

```sh
git clone https://github.com/your_username/tpu-pod-tutorial ~/your_folder/tpu-pod-tutorial
```


## Setup uv project

To make the builds reproducible across pods, we will be using 
`uv` as a package manager. It's like `pip` but *much* better.

> If `uv` is not installed, follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

`cd` into `your_folder` and initialize `uv`

```sh
uv init .
```

You should see something like:

```
$ uv init .
Initialized project `tpu-pod-tutorial` at `/home/ucsdwanglab/nathan/tpu-pod-tutorial`
$ ls
LICENSE  README.md  hello.py  pyproject.toml
```

Let's add JAX as a dependency

```sh
uv add --prerelease=allow "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

```
Using CPython 3.10.16
Creating virtual environment at: .venv
Resolved 14 packages in 154ms
Installed 13 packages in 61ms
 + certifi==2024.12.14
 + charset-normalizer==3.4.1
 + idna==3.10
 + jax==0.5.0
 + jaxlib==0.5.0
 + libtpu==0.0.8
 + libtpu-nightly==0.1.dev20241010+nightly.cleanup
 + ml-dtypes==0.5.1
 + numpy==2.2.1
 + opt-einsum==3.4.0
 + requests==2.32.3
 + scipy==1.15.1
 + urllib3==2.3.0
$ ls
LICENSE  README.md  hello.py  pyproject.toml  uv.lock
```

## Run distributed code

Our program is *distributed*. This means that it involves several machines 
(we call them hosts), each of whom run a process, and communicate with each other over
network cables. Crucially, they **do not share a filesystem**, so we need to distribute all of our code and data
before we can run. 

Let's begin:

First, let's make sure we can connect to all the hosts. To do this, we will use the `eopod` tool, 
which can be run with `uv`'s execuation tool `uvx`.

```sh
uvx eopod run echo "Hello, world!"
```

```
Started at: 2025-01-18 20:27:48
Executing: echo Hello, world!
Executing command on worker all: echo Hello, world!
Using ssh batch size of 1. Attempting to SSH into 1 nodes with a total of 4 workers.
SSH: Attempting to connect to worker 0...
SSH: Attempting to connect to worker 1...
SSH: Attempting to connect to worker 2...
SSH: Attempting to connect to worker 3...
Hello, world!
Hello, world!
Hello, world!
Hello, world!
Command completed successfully

Command completed successfully!
Completed at: 2025-01-18 20:27:52
Duration: 0:00:03.858619
```

You should see one `Hello, world!` per host. I have 4 hosts, so I see 4. Running Python
code is a bit more involved.

Edit `hello.py` to contain

```python
import jax

jax.distributed.initialize()

if jax.process_index() == 0:
  print(jax.device_count())
```

This will initialize JAX's distributed computing system, and print the total number
of detected devices from Host 0.

Remember that our hosts don't share a filesystem, so we need to sync this code among all the machines. We'll
use `git` to do this


```sh
git add . && git commit -am "Add hello.py" && git push
```