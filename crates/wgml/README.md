# wgml: cross-platform GPU LLM inference

**/!\ This library is still under heavy development and is still missing many features.**

The goal of **wgml** is to provide composable WGSl shaders and kernels for cross-platform GPU LLM inference.

## Running the models

Currently, the `gpt2` and `llama2` models are implemented. They can be loaded from gguf files. Support of quantization
is very limited (tensors are systematically unquantized upon loading) and somewhat untested. A very basic execution
of these LLMs can be run from the examples.

### Running GPT-2

```shell
cargo run -p wgml --example gpt2 -- your_model_file_path.gguf --prompt "How do I bake a cake?"
```

Note that this will run both the gpu version and cpu version of the transformer.

### Running llama-2

```shell
cargo run -p wgml --example llama2 -- your_model_file_path.gguf
```

Note that this will run both the cpu version and gpu version of the transformer.