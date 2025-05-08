load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_shell//shell:sh_test.bzl", "sh_test")

exports_files(
    ["train_model.py"],
    visibility=["//visibility:public"],  # Makes it visible to all packages
)

py_binary(
    name="train_model",
    srcs=["train_model.py"],
    deps=[
        "@pypi//torch",
        "@pypi//transformers",
        "@pypi//datasets",
        "@pypi//accelerate",
        "@pypi//psutil",
    ],
    visibility=["//visibility:public"],
)

sh_test(
    name="training_test",
    srcs=["//:python_provider.sh"],  # Use the python provider script from root
    args=["src.training.train_model"],  # Pass the Python module path as an argument
    data=[  # The Python script as a data dependency
        "//src/training:train_model.py",
    ],
    # Optional: Set appropriate test size/timeout
    size="large",
)
