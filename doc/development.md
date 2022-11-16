# Development (Graphcore)

## Contribution process

To contribute code, please:
 - Open a Pull Request
 - Add a reviewer
 - When you get an LGTM, submitter merges
   - Merge-squash by default
   - Merge when required (e.g. branches-upon-branches)


## Coding guidelines

A few principles that we feel are important:
 - Unless refactoring, keep with the existing pattern of code
 - Keep code simple by:
   - Minimising the need to reason about state (e.g. functional style)
   - Using descriptive, but not overly long, variable names
   - Ensuring methods/classes/etc have a single, easy-to-describe responsibility
 - Let's not worry about basic formatting - use the autoformatters clang-format & black, as configured for everyone
 - Try to avoid dependencies that aren't strictly necessary (both internal & external)
 - Keep trying to improve test coverage, but with as little testing code as possible


## Using VSCode

We strongly recommend taking the time to sort out indexing & autocomplete in your IDE of choice. It should be possible to get suggestions for C++ and Python (including poplar_kge but not libpoplar_kge) using the following.

```
ln -s $POPLAR_SDK_ENABLED third_party/poplar

# Add the following to .vscode/settings.json
{
    "editor.formatOnSave": true,
    "python.defaultInterpreterPath": ".venv/bin/python",
    "C_Cpp.default.includePath": [
        "${workspaceFolder}/src",
        "${workspaceFolder}/third_party/poplar/include",
        "${workspaceFolder}/third_party/pybind11/include",
        "${workspaceFolder}/third_party/catch2/single_include"
    ]
}
```
