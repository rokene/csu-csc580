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

### Handwritten Digits App

One time setup

```bash
make handwritten-digits-ml-app-setup
```

Make sure you are in the `mnist-tf` conda env.

```bash
conda activate mnist-tf
```

Execute the hand written ml app training

```bash
make handwritten-digits-ml-app-train
```

Execute the hand written ml app inference

```bash
make handwritten-digits-ml-app-infer
```

Execute the learning rate test

```bash
make handwritten-digits-ml-app-lrtest
```

Execute the hidden layers test

```bash
make handwritten-digits-ml-app-ltest
```

Execute the batch size test

```bash
make handwritten-digits-ml-app-btest
```

## Fuel Efficiency

Setup Environment.

```bash
make basic-fuel-efficency-setup
```

Make sure you are in the `fueleff` conda env.

```bash
conda activate fueleff
```

Execute training.

```bash
make basic-fuel-efficency MODE=train
```

Execute trainference.

```bash
make basic-fuel-efficency MODE=infer
```
