# Virtual Environments

When you work with Python projects, you should use a **virtual environment** to isolate the packages installed for each project.

For FastAPI projects, I recommend using [uv](https://docs.astral.sh/uv/) to manage the project, its dependencies, and its virtual environment.

## Create a Project

Install `uv` using the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/), and then create a project:

```console
$ uv init awesome-project --bare
$ cd awesome-project
$ uv add "fastapi[standard]"
```

`uv` creates a virtual environment for the project automatically. You don't need to create or activate one yourself.

Run commands inside the project environment with `uv run`, for example:

```console
$ uv run fastapi dev
```

## Learn More

Read the [Virtual Environments guide](https://tiangolo.com/guides/virtual-environments/) to learn how virtual environments work underneath, including activation and the alternative `python -m venv` and `pip` workflow.
