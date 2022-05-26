<div align="center">

  <picture>
    <source height="200" media="(prefers-color-scheme: dark)" srcset="https://user-images.githubusercontent.com/17221436/171939728-199315cb-e950-4bf1-9a7d-6c83e1866e5c.svg">
    <source height="200" media="(prefers-color-scheme: light)" srcset="https://user-images.githubusercontent.com/17221436/171939288-9d3c8427-d508-48f6-9538-07f35868801a.svg">
    <img height="200" alt="Shows PrometheusML logo" src="https://yailab.com/assets/logo/logo-prometheus-dark-letters.svg">
  </picture>

</div>

-----------------
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![MIT License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

PrometheusML Core is an open-source no-code platform for building machine learning
and deep learning models, developed by [yaiLab](https://yailab.com/).

## What exactly is PrometheusML and PrometheusML Core?

There are two versions of PrometheusML:
- The open-source PrometheusML Core
- The cloud data science assistant PrometheusML

PrometheusML Core allows anyone and everyone to build machine learning models in an
interactive way through a UI, whilst PrometheusML is a cloud data science assistant that helps
users build specialised machine learning models fast in physics-intense fields.

## Is it free?

Yes, PrometheusML Core is completely free. You can use PrometheusML Core for free by following the installation steps
in this [repo](#install-prometheusml).

Alternatively, to try out the cloud data science assistant PrometheusML on a **free** trial go to our website and press on
[Try it now!](https://yailab.com). You can also find more info on the data science assistant
PrometheusML [here](https://yailab.com/product.html).

## How does PrometheusML Core work?

PrometheusML Core will help you build an entire machine learning pipeline without
writing a single of code by guiding you through the entire process.

### 1. Select regression or classification template

<video src="https://user-images.githubusercontent.com/105654866/171970130-fdfd35c1-4f4c-4a47-9160-5b0feb8d7f1e.mp4">
</video>

![](doc/tutorial/videos/template_selection.mp4)


### 2. Data upload and evaluation

<video src="https://user-images.githubusercontent.com/105654866/171970137-ab258ccd-1315-4295-81f1-0d9a7054dd55.mp4">
</video>

![](doc/tutorial/videos/upload_exploration.mp4)


### 3. Feature engineering

<video src="https://user-images.githubusercontent.com/105654866/171970184-6920a11d-8082-4d3b-8303-1bdb9a84a52a.mp4">
</video>

![](doc/tutorial/videos/feature_selection.mp4)


### 4. Algorithm selection

<video src="https://user-images.githubusercontent.com/105654866/171970239-918b0a62-226e-4f46-b3cf-9da782ee0b5e.mp4">
</video>

![](doc/tutorial/videos/algorithm_selection.mp4)


### 5. Model validation and deployment

<video src="https://user-images.githubusercontent.com/105654866/171970276-7f528062-5a1f-4b72-a46f-3dde016c86a8.mp4">
</video>

![](doc/tutorial/videos/model_validation.mp4)


### 6. Making predictions

<video src="https://user-images.githubusercontent.com/105654866/171970344-a64f311d-3999-42d7-ba65-1bb46525cdba.mp4">
</video>

![](doc/tutorial/videos/prediction.mp4)

## Canonical source

The canonical source of PrometheusML where all development takes place is
[hosted on GitLab.com](https://gitlab.com/yailab/prometheusml).

## Table of contents

* [Install PrometheusML](#install-prometheusml)
* [Documentation](#documentation)
* [Discussion on AI ethics](#discussion-on-ai-ethics)
* [Contributing](#contributing-to-prometheusml)
* [License and Copyright](#copyright-and-license)
* [Community](#community)


## Install PrometheusML

Install Prometheus<span style="color: #ff7F2a;">ML</span> with the cloud native Docker Compose tool.

**_Important consideration!_** - The default Docker Compose configuration is not intended for production. It creates
a proof of concept (PoC) implementation where all Prometheus<span style="color: #ff7F2a;">ML</span> services are placed
into a cluster.

Follow the next steps to quickly install and take advantage of Prometheus<span style="color: #ff7F2a;">ML</span>:

### 1. Install Docker Desktop
You need to have Docker Compose **installed** on your computer. You can easily install it by taking advantage of the
Docker Desktop installation.
To install Docker Desktop go to their [website](https://docs.docker.com/compose/install/compose-desktop/).

### 2. Clone repository
Open your:
* **_PowerShell_** for Windows
* **_Terminal_** for Mac or Linux

... and type the following command:
```sh
git clone https://gitlab.com/yailab/prometheusml.git && cd prometheusml
```

### 3. Launch PrometheusML

Type some more commands in your **_PowerShell/Terminal_** as follows:

```sh
# Build the necessary image
docker compose build
# Run the multi-container application
docker compose up -d --remove-orphans
```

**_Note_**: You can stop Prometheus<span style="color: #ff7F2a;">ML</span> by typing into your
_PowerShell/Terminal_ the command `docker compose down`.

### 4. You are done!
You can now access Prometheus<span style="color: #ff7F2a;">ML</span> through the browser of your choice
by typing the address `localhost:5000`.

### 5. Set up your username and password and get going!
You can create **a new user** by going to the registration page of your locally launched
Prometheus<span style="color: #ff7F2a;">ML</span> instance.

The created user comes preloaded
with two **templates**, a general _regression_ and a _classification_ one. You can check out the [tutorials videos](#documentation)
that will guide you through the entire process of building a regression and a classification machine learning model.

**_Enjoy building machine learning models!_**

## Documentation
The documentation of PrometheusML is under
**_active development!_**

You can start with the following **tutorial videos** for a quick introduction:

| Problem    | Description               | Dataset source                                                                            | Tutorial video                          |
|------------|---------------------------|-------------------------------------------------------------------------------------------|-----------------------------------------|
| Regression | Predict concrete strength | [Kaggle](https://www.kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set) | [Youtube](https://youtu.be/x45gEUYpJXM) |


Contributing to the documentation benefits everyone who uses PrometheusML.
We encourage you to help us improve the documentation, and you don’t have to be an expert to do so!
In fact, there are sections of the docs that are worse off after being written by experts. If something in the docs
does not make sense to you, updating the relevant section after you figure it out is a great way to ensure it will help
the next person.


## Discussion on AI ethics
**_With great power comes great responsibility_**.

At yaiLab we aim to make AI accessible to everyone and believe in **responsible AI use**.
We are super excited about AI’s ability to help humanity with its biggest problems, and
committed to **Ethical** and **Responsible** AI.

The journey to responsible and accessible AI includes
transparency and open inclusive discussion. Review our [AI Ethics Policy](AI_ETHICS_POLICY.md) and
join our [Discord channel](https://discord.gg/UH4D8QZv) to be part of
the conversation for the future of responsible AI.

## Contributing to PrometheusML
PrometheusML is an open source project, and we are
very happy to accept community contributions. Please refer to the [Contributing guide](CONTRIBUTING.md)
for more details.

### Code of Conduct

Please treat others with respect and follow the guidelines articulated in the
[Community Code of Conduct](CODE_OF_CONDUCT.md).

### Versioning

This project is maintained under the [Semantic Versioning guidelines](https://semver.org/).

See the [Releases section](https://gitlab.com/yailab/prometheusml/-/releases) of our PrometheusML project for
changelogs for each release version of PrometheusML.


## Copyright and license

Code and documentation copyright 2022 yaiLab, Ltd.

See the [LICENSE](LICENSE) file for licensing information as it pertains the files in this repository.

Code released under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.


## Community

* Follow [@lab_yai](https://twitter.com/lab_yai) on Twitter for the latest yaiLab news, or
sign up for our [newsletter](https://yailab.com/video_newsletter.html).
* If you want to get involved or simply chat, join our [Discord community](https://discord.com/invite/Pb6mnkzGfF)!
