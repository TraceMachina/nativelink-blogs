# Train a machine learning model with Bazel and NativeLink

An example project for training a machine learning model with Bazel.

## ❄️ Setup

1. Either have Bazelisk installed or enter the nix flake from this repository.
2. Create a `user.bazelrc` in this directory and paste the values from the
   `Quickstart` section at <https://app.nativelink.com> into it:

    ```bash
    # user.bazelrc
    build --remote_cache=grpcs://TODO
    build --remote_header=x-nativelink-api-key=TODO
    build --bes_backend=grpcs://TODO
    build --bes_header=x-nativelink-api-key=TODO
    build --bes_results_url=https://app.nativelink.com/TODO
    build --remote_timeout=600
    build --remote_executor=grpcs://TODO
    ```

## 🚄 Training

Run a local invocation:

```bash
bazel run training
```

Now run a remote invocation. The first time you do this it'll be slow (~15min)
as the runner needs to spin up and fetch the various toolchains:

```bash
bazel test training --config=remote --test_output=all
```

Run the above command a second time. Since the runner is now warm it'll
immediately start the test (~5min).

## ✨ Keeping dependencies up-to-date

The dependencies for this project are tracked in `pyproject.toml` and locked in
`requirements_{linux±|macos}.lock` via a Bazel-wrapped `uv` toolchain. To
regenerate the lockfiles:

```bash
# On linux
bazel run //:requirements_linux.update

# On macos
bazel run //:requirements_macos.update
```

On darwin you might have to disable remote execution temporarily for this.

## 🧹 Fixing formatting and linting

Use the flake if you develop on this code. Don't use local ruff installations as
versions other than the one used in the flake might format code differently.

```bash
# Check lints
ruff check

# Format
ruff format
```
