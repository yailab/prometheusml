[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

----

# Contribute to PrometheusML
Thank you for considering contributing to PrometheusML!

## How can you contribute?
All contributions, code or no-code ones, and ideas are welcome. You can:

- Report a bug
- Improve documentation
- Submit a bug fix
- Propose a new feature or improvement
- Contribute to a new feature or improvement
- Join the AI ethics discussion on our [Discord](https://discord.gg/M6E7msv6)
- Test PrometheusML

### AI Ethics discussion
**_With great power comes great responsibility_**.

At yaiLab we aim to make AI accessible to everyone and believe in **responsible AI use**.
We are super excited about AI’s ability to help humanity with its biggest problems, and
committed to **Ethical** and **Responsible** AI.

The journey to responsible and accessible AI includes
transparency and open inclusive discussion. Join our [Discord channel](https://discord.gg/UH4D8QZv) and be part of
the discussion for the future of responsible AI.

## Security vulnerability disclosure

Report suspected security vulnerabilities in private to
`info@yailab.com`.

WARNING:
Do **NOT** create publicly viewable issues for suspected security vulnerabilities.


## Contribution Checklist
Before starting your contribution to PrometheusML, make sure you do the following:

- Read the [contributing guidelines](CONTRIBUTING.md).
- Read the [Code of Conduct](CODE_OF_CONDUCT.md).
- Ensure you agree to the [Contributor License Agreement](#contributor-license-agreement).
- Ensure you agree to the [AI Ethics Policy](AI_ETHICS_POLICY.md).
- Check if your changes are consistent with the
    [guidelines](#general-guidelines-and-philosophy-for-contribution).
- Changes are consistent with the [Coding Style](#python-coding-style).
- Run the [unit tests](#running-the-test-suite).

### Contributor license agreement
We'd love to accept your contributions! Before we can take them, we have to jump a couple of
legal hurdles (we know... ) .

By submitting code as an individual you agree to the
[individual contributor license agreement](doc/legal/individual_contributor_license_agreement.md).
By submitting code as an entity you agree to the
[corporate contributor license agreement](doc/legal/corporate_contributor_license_agreement.md).

All Documentation content that resides under the [`doc/` directory](/doc) of this
repository is licensed under Creative Commons:
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).


[//]: # (_This notice should stay as the first item in the `CONTRIBUTING.md` file._)

## Contributing code
If you have improvements to PrometheusML, send us your merge requests! For those just getting started,
GitLab has a [how-to](https://docs.gitlab.com/ee/user/project/merge_requests/getting_started.html).

### Where to start?

If you want to contribute, start working through the PrometheusML codebase,
navigate to the [GitLab "issues" tab](https://gitlab.com/yailab/prometheusml/-/issues) and start looking through
interesting issues.

If you are not sure of where to start:

* Check the issues listed under the ["good first issue"](https://gitlab.com/yailab/prometheusml/-/issues/?label_name%5B%5D=good%20first%20issue)
and the ["contributions welcome"](https://gitlab.com/yailab/prometheusml/-/issues/?label_name%5B%5D=status%3A%3Acontributions%20welcome)
labels.
* Increasing our test coverage is another great opportunity to contribute.

If you decide to start on an issue, leave a comment so that other people know that you're working on it.
If you want to help out, but not alone, use the issue comment thread to coordinate.

### Working with the code
Now that you have found an issue you want to fix, enhancement to add, or documentation to improve,
you need to learn how to work with GitLab and the PrometheusML code base.

Here is the general workflow of contributing to PrometheusML:
1. Create a fork of PrometheusML. Check [Gitlab's documentation on forking workflow](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html).
2. Make your changes in your fork.
3. When you're ready, [create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html).
4. In the merge request's description:
   - Ensure you provide complete and accurate information.
   - Review the provided checklist.
5. Assign the merge request (if possible) to, or [mention](../../user/discussions/index.md#mentions),
   one of the [code owners](../../user/project/code_owners.md) for the relevant project,
   and explain that you are ready for review.


### Contribution guidelines and standards
Before sending your merge request for review, make sure your changes are consistent with the
guidelines and follow the PrometheusML coding style.

The only **exception** in the guidelines is **UI**, because it is in the process of **significant refactoring**! If you want to
contribute to UI, please first come to our [Discord channel](https://discord.gg/J699JaEJ) for a quick chat.

#### General guidelines and philosophy for contribution

* We highly recommend that you [open an issue](https://gitlab.com/yailab/prometheusml/-/issues/new), describe your contribution, share all needed information there and
link it to a merge request.
* Include unit tests when you contribute new features, as they help to a)
    prove that the code works correctly, and b) guard against future breaking
    changes to lower the maintenance cost.
* Bug fixes also generally require unit tests, because the presence of bugs
    usually indicates we have an insufficient test coverage.
* The yaiLab team is evaluating merge requests taking into account: code architecture and quality, code style,
comments & docstrings and coverage by tests.


### Creating a development environment

#### Development environment with Docker (recommended)
Instead of manually setting up a development environment, you can use [Docker Compose](https://docs.docker.com/compose/install/)
to automatically create the environment with just a few commands. yaiLab provides a `docker-compose-dev.yml` file
in the root directory to define and run the full PrometheusML development environment.

##### **Docker Compose Commands**

Build the Docker images:
```sh
# Set needed environment variable(s)
# You can built the images by using: gl_username="yailab", or
# by passing your GitLab username to use your own fork
export gl_username=<gitlab_username> prometheus_dir="/home/prometheus/prometheusml"

# Build the various images
docker compose -f docker-compose-dev.yml build
```

Start and run PrometheusML's services:
```sh
# Run the containers and bind your local repo to them
docker compose -f docker-compose-dev.yml up -d --remove-orphans
```

_Even easier, you can integrate Docker with the following IDEs:_

**Visual Studio Code**

You can use the `DockerFile` to launch a remote session with Visual Studio Code, a popular free IDE,
using the `.devcontainer.json` file. See https://code.visualstudio.com/docs/remote/containers for details.

**PyCharm (Professional)**

Enable Docker support and use the Services tool window to build and manage images as well as run and interact with containers. See https://www.jetbrains.com/help/pycharm/docker.html for details.


### Code standards
Writing good code is not just about what you write. It is also about _how_ you write it.

#### Python coding style

Changes to PrometheusML Python code should conform to
[Black Style Guide](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html).
To format a file with `black` against its style definition:

```sh
black <path/to/file>
```

We use `flake8` for code style checks. To check a file with `flake8` against Black's style definition:

```sh
flake8 <path/to/file>
```

#### Type hints
yaiLab strongly encourages the use of [PEP 484](https://peps.python.org/pep-0484/) style type hints.
New development should contain type hints and **merge requests** to **annotate** existing code are accepted as well!

We use `mypy` to statically analyse the code base and type hints. After making any change you can ensure
your type hints are correct by running:

```sh
mypy <path/to/file>
```

**_This part is under development, and we would greatly appreciate your help!_**

#### Running the test suite
We are trying to embrace [test-driven development (TDD)](https://en.wikipedia.org/wiki/Test-driven_development), and are
strongly encouraging contributors too. So, before actually writing any code, you should write your tests.

All tests should go into the `tests` subdirectory of the specific package (i.e. `Prometheus` and `Webapp`).

The tests can then be run directly inside your Git clone by typing:
```sh
pytest prometheus
```

and
```sh
pytest webapp
```

It is often worth running only a subset of tests first around your changes before running the entire
suite. The easiest way to do this is with:
```sh
pytest path/to/test.py -k regex_matching_test_name
```

**_This part is under development, and we would greatly appreciate your help!_**


### Pre-commit
[Continuous Integration](https://docs.gitlab.com/ee/ci/introduction/#continuous-integration) will run code formatting
checks like `black`, `flake8`, `isort`. Any warnings from these checks will cause the `Continuous Integration` to fail;
therefore, it is helpful to run the check yourself before submitting code.

You can run many of these styling checks manually as we have described above.
However, we encourage you to use [pre-commit hooks](https://pre-commit.com/) instead to automatically run them
when you make a git commit. This can be done by installing `pre-commit`:

```sh
pip install pre-commit
```

... and then running on the root directory of PrometheusML:

```sh
pre-commit install
```

If you don't want to use `pre-commit` as part of your workflow, you can run its checks to the files
you have modified with:
```sh
pre-commit run --files <files you have modified>
```
