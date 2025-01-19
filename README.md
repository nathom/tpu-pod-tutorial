# tpu-pod-tutorial

This tutorial will get you started with multi-host VM TPU pods.
If you don't want to follow the tutorial step by step, a reference edited version
is on the `run` branch.

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
Initialized project `tpu-pod-tutorial` at `/home/ucsdwanglab/your_folder/tpu-pod-tutorial`
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

You should see one `Hello, world!` per host. I have 4 hosts, so I see 4.

Running Python code is a bit more involved. First, let's install uv
on all hosts

```sh
uvx eopod run 'curl -LsSf https://astral.sh/uv/install.sh | sh'
```

```
Started at: 2025-01-18 21:02:46
Executing: curl -LsSf https://astral.sh/uv/install.sh | sh
Executing command on worker all: curl -LsSf
https://astral.sh/uv/install.sh | sh
Using ssh batch size of 1. Attempting to SSH into 1 nodes with a total of 4 workers.
SSH: Attempting to connect to worker 0...
SSH: Attempting to connect to worker 1...
SSH: Attempting to connect to worker 2...
SSH: Attempting to connect to worker 3...
downloading uv 0.5.21 x86_64-unknown-linux-gnu
downloading uv 0.5.21 x86_64-unknown-linux-gnu
downloading uv 0.5.21 x86_64-unknown-linux-gnu
downloading uv 0.5.21 x86_64-unknown-linux-gnu
no checksums to verify
no checksums to verify
no checksums to verify
installing to /home/ucsdwanglab/.local/bin
no checksums to verify
installing to /home/ucsdwanglab/.local/bin
  uv
  uvx
everything's installed!
  uv
  uvx
everything's installed!
installing to /home/ucsdwanglab/.local/bin
  uv
  uvx
everything's installed!
installing to /home/ucsdwanglab/.local/bin
  uv
  uvx
everything's installed!
Command completed successfully
```

And verify that `uv` is available:

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

Next, we clone this repo on all hosts:

```sh
uvx eopod run git clone https://github.com/your_username/tpu-pod-tutorial ~/your_folder/tpu-pod-tutorial
```

Make sure we get a `Success` from every host:

```sh
uvx eopod run '[ -d ~/your_folder/tpu-pod-tutorial ] && echo Success || echo Fail'
```

Now, we can run the program. Create `run.sh` with contents

```bash
#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <python_file>"
    exit 1
fi

PYTHON_FILE="$1"

eopod kill-tpu --force
eopod run "
export PATH=$PATH &&
cd ~/your_folder/tpu-pod-tutorial &&
git pull &&
uv run --prerelease allow $PYTHON_FILE
"
```

And run

```sh
bash run.sh hello.py
```

```
$ bash run.sh
⠋ Scanning for TPU processes...Fetching TPU status...
⠙ Scanning for TPU processes...TPU state: READY
Executing command on worker 0: ps aux | grep -E
'python|jax|tensorflow' | grep -v grep | awk '{print $2}' | while read
pid; do   if [ -d /proc/$pid ] && grep -q 'accel' /proc/$pid/maps
2>/dev/null; then     echo $pid;  fi; done
Executing command on worker 1: ps aux | grep -E
'python|jax|tensorflow' | grep -v grep | awk '{print $2}' | while read
pid; do   if [ -d /proc/$pid ] && grep -q 'accel' /proc/$pid/maps
2>/dev/null; then     echo $pid;  fi; done
Executing command on worker 2: ps aux | grep -E
'python|jax|tensorflow' | grep -v grep | awk '{print $2}' | while read
pid; do   if [ -d /proc/$pid ] && grep -q 'accel' /proc/$pid/maps
2>/dev/null; then     echo $pid;  fi; done
Executing command on worker 3: ps aux | grep -E
'python|jax|tensorflow' | grep -v grep | awk '{print $2}' | while read
pid; do   if [ -d /proc/$pid ] && grep -q 'accel' /proc/$pid/maps
2>/dev/null; then     echo $pid;  fi; done
⠏ Scanning for TPU processes...Command completed successfully
Command completed successfully
⠋ Scanning for TPU processes...Command completed successfully
⠙ Scanning for TPU processes...Command completed successfully
No TPU processes found.
⠙ Scanning for TPU processes...
Started at: 2025-01-18 21:00:12
Executing:
export
PATH=/home/ucsdwanglab/.local/bin:/home/ucsdwanglab/.local/bin:/usr/lo
cal/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/
local/games:/snap/bin:/home/ucsdwanglab/.local/bin &&
cd ~/your_folder/tpu-pod-tutorial &&
git checkout run &&
git pull &&
uv run --prerelease allow hello.py

Executing command on worker all:
export
PATH=/home/ucsdwanglab/.local/bin:/home/ucsdwanglab/.local/bin:/usr/lo
cal/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/
local/games:/snap/bin:/home/ucsdwanglab/.local/bin &&
cd ~/your_folder/tpu-pod-tutorial &&
git checkout run &&
git pull &&
uv run --prerelease allow hello.py

Using ssh batch size of 1. Attempting to SSH into 1 nodes with a total of 4 workers.
SSH: Attempting to connect to worker 0...
SSH: Attempting to connect to worker 1...
SSH: Attempting to connect to worker 2...
SSH: Attempting to connect to worker 3...
Already on 'run'
M	README.md
Your branch is up to date with 'origin/run'.
Already on 'run'
Your branch is up to date with 'origin/run'.
Already on 'run'
Your branch is up to date with 'origin/run'.
Already on 'run'
Your branch is up to date with 'origin/run'.
Already up to date.
Already up to date.
Already up to date.
Already up to date.
16
Command completed successfully

Command completed successfully!
Completed at: 2025-01-18 21:00:20
Duration: 0:00:08.020470
```

We see there are 16 detected devices on the `v4-32` pod! 

Let's try running a distributed computation. Create a file `pmap_test.py` in the repo
with the following contents

```python
# The following code snippet will be run on all TPU hosts
import jax

# The total number of TPU cores in the Pod
device_count = jax.device_count()

# The number of TPU cores attached to this host
local_device_count = jax.local_device_count()

# The psum is performed over all mapped devices across the Pod
xs = jax.numpy.ones(jax.local_device_count())
r = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)

# Print from a single host to avoid duplicated output
if jax.process_index() == 0:
    print('global device count:', jax.device_count())
    print('local device count:', jax.local_device_count())
    print('pmap result:', r)
```

Commit and push

```sh
git add pmap.py && git commit -am "Add pmap.py" && git push 
```

And run 

```sh
bash run.sh pmap.py
```

After all the setup output, it should show something like

```
global device count: 16
local device count: 4
pmap result: [16. 16. 16. 16.]
```
