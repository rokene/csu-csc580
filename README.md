# csu-csc580

Applied Neural Net and Machine Learning course. We will primarily be using
windows and python env for the projects. We will also use Anaconda for managing
python environments and versions because these are research and developement
type activities.

## Prequisites and Setup

* Python 3.12+
* Anaconda

Setup `conda` cli.

```bash

# generic
conda init

# powershell specific
conda init powershell
```

## Usage 

For a list of commands: `make`

### Basic Facial Recognition App

One time setup.

```bash
# One time setup
make basic-facial-app-setup
```

Make sure you are in the `faceenv` conda env.

```bash
conda activate faceenv
```
Execute the basic facial app.

```bash
make basic-facial-app
```
