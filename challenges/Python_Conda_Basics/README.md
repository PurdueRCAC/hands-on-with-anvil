# Python: Conda Basics

In high-performance computing, [Python](https://www.python.org/) is heavily used to analyze scientific data on the system. 
Various Python installations and scientific packages need to be installed to analyze data for our users. These Python installations can become difficult to manage on an HPC system as the programming environment is complicated.  [Conda](https://conda.io/projects/conda/en/latest/index.html), a package and virtual environment manager from the [Anaconda](https://www.anaconda.com/) distribution, helps alleviate these issues. [Miniforge](https://github.com/conda-forge/miniforge) is an open source version of Miniconda, which is what the OLCF crash course will use to be able to utilize conda environments.

Conda allows users to easily install different versions of binary software packages and any required libraries appropriate for their computing platform.
The versatility of conda allows a user to essentially build their own isolated Python environment, without having to worry about clashing dependencies and other system installations of Python.

This hands-on challenge will introduce a user to installing Conda on Anvil, the basic workflow of using conda environments, as well as providing an example of how to create a conda environment that uses a different version of Python than the base environment uses on Anvil.

&nbsp;

## Installing Miniconda

Currently, Anvil provides a few different ways to manage Python environments, most commonly by way of Anaconda modules. As new releases of Anaconda are available we add them to the modules but do not remove previous ones to not break existing environments users have created from them.

```bash
$ module avail anaconda
```

&nbsp;

## Setting up the environment

First, we will unload all the current modules that you may have previously loaded on Anvil:

```bash
$ module reset
```

Next, we need to load the `anaconda` module:

```bash
$ module load anaconda/2024.02-py311
```

This puts you in the "`base`" conda environment.
You will not be able to install new packages into the `base` environment because it is write protected from users. Instead you will want to create your own environments and install packages into them. 
So, next, we will create a new environment using the `conda create` command:

```bash
$ conda create -n py39-anvil python=3.9
```

The "`-n`" flag specifies the desired name of your new virtual environment.
This will install the environment into your home directory in a specific location. Instead, one can use the `-p <path>` option which will install to some other desired location (like your project directory).

After executing the `conda create` command, you will be prompted to install "the following NEW packages" -- type "y" then hit Enter/Return.
Downloads of the fresh packages will start and eventually you should see something similar to:

```
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate py39-anvil
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

<!-- Please note, more recent releases of the Anaconda distribution for Python, specifically the `conda` package manager, now provide an activation mechanism that lets you invoke `conda activate x` instead of `source activate x`. While this improves the overall experience of most users and allows for a consistent interface across platforms, it is problematic on an HPC cluster. It is problematic because it asks you to first invoke `conda init $SHELL` before it will allow you to call `conda activate` on any of your environments. When you run `conda init` all that happens is it inserts a short snippet of code into your shell profile (e.g., `~/.bashrc`) to automatically activate the base environment every time you log in. This is convenient on your personal machine but problematic on an HPC resource where you will likely need to do things other than use that particular Anaconda or worse, it sources an expensive intialization script for every file transfer. -->

Let's activate our new environment:

```bash
$ conda activate py39-anvil
```

The name of your environment should now be displayed in "( )" at the beginning of your terminal lines, which indicate that you are currently using that specific conda environment.
And if you check with `conda env list` again, you should see that the `*` marker has moved to your newly activated environment:

```
$ conda env list

# conda environments:
#
py39-anvil          *  /home/<user>/.conda/envs/py39-anvil
base                   /apps/.../anaconda/2024.02-py311
```

&nbsp;

## Installing packages

Next, let's install a package ([NumPy](https://numpy.org/)). 
There are a few different approaches.

One way to install packages into your conda environment is to build packages from source using [pip](https://pip.pypa.io/en/stable/).
This approach is useful if a specific package or package version is not available in the conda repository, or if the pre-compiled binaries don't work on the HPC resources (which is common).
However, building from source means you need to take care of some of the dependencies yourself, especially for optimization.
In Anvil's case, this means we need to load the `openblas` module.
Pip is available to use after installing Python into your conda environment, which we have already done.


> NOTE: Because issues can arise when using conda and pip together (see link in [Additional Resources Section](#refs)), it is recommended to do this only if absolutely necessary.


To build a package from source, use `pip install --no-binary=<package_name> <package_name>`:

```bash
$ module load openblas
$ CC=gcc pip install --no-binary=numpy numpy
```

The `CC=gcc` flag will ensure that we are using the proper compiler and wrapper.
Building from source results in a longer installation time for packages, so you may need to wait a few minutes for the install to finish.

Congratulations, you have built NumPy from source in your conda environment!  

We did not link in any additional linear algebra packages, so this version of NumPy is not optimized.
Let's install a more optimized version using a different method instead, but first we must uninstall the pip-installed NumPy:

```bash
$ pip uninstall numpy
$ module unload openblas
```

The traditional, and more basic, approach to installing/uninstalling packages into a conda environment is to use the commands `conda install` and `conda remove`.
Installing packages with this method checks the [Anaconda Distribution Repository](https://docs.anaconda.com/anaconda/packages/pkg-docs/) for pre-built binary packages to install.
Let's do this to install NumPy:

```bash
$ conda install numpy
```

Conda handles dependencies when installing pre-built binaries, so  it will automatically install all of the packages NumPy needs for optimization.   

Congratulations, you have just installed an optimized version of NumPy, now let's test it!

&nbsp;

## Testing your new environment

Let's run a small script to test that things installed properly.
Since we are running a small test, we can do this without having to run on a compute node. 

> NOTE: Remember, at larger scales both your performance and your fellow users' performance will suffer if you do not run on the compute nodes.

It is always highly recommended to run on the compute nodes (through the use of a batch job or interactive batch job).

Make sure you're in the correct directory and execute the example Python script:

```
$ cd ~/hands-on-with-anvil/challenges/Python_Conda_Basics/
$ python3 hello.py

Hello from Python 3.9.18!
You are using NumPy 1.26.0
```

Congratulations, you have just created your own Python environment and ran on one of the fastest computers in the world!


> Note: If you're doing this challenge for the certificate, you can submit your Python environment for completion. See "Exporting (sharing) an environment" tip below of how to export your environment to a file.

&nbsp;

## Additional Tips

* Cloning an environment:

    It is not recommended to try to install new packages into the base environment.
    Instead, you can clone the base environment for yourself and install packages into the clone.
    To clone an environment, you must use the `--clone <env_to_clone>` flag when creating a new conda environment.
    An example for cloning the base environment into your `$HOME` directory on Anvil is provided below:

    ```bash
    $ conda create -n baseclone-anvil --clone base
    $ conda activate baseclone-anvil
    ```

* Deleting an environment:

    If for some reason you need to delete an environment, you can execute the following:

    ```bash
    $ conda env remove -n <name>
    ```

* Exporting (sharing) an environment:

    You may want to share your environment with someone else.
    As mentioned previously, one way to do this is by creating your environment in a shared location where other users can access it.
    A different way (the method described below) is to export a list of all the packages and versions of your environment (an `environment.yml` file).
    If a different user provides conda the list you made, conda will install all the same package versions and recreate your environment for them -- essentially "sharing" your environment.
    To export your environment list:
    
    ```bash
    $ conda activate my_env
    $ conda env export > environment.yml
    ```
    
    You can then email or otherwise provide the `environment.yml` file to the desired person.
    The person would then be able to create the environment like so:
    
    ```bash
    $ conda env create -f environment.yml
    ```

* Adding known environment locations:

    For a conda environment to be callable by a "name", it must be installed in one of the `envs_dirs` directories.
    The list of known directories can be seen by executing:

    ```bash
    $ conda config --show envs_dirs
    ```

    On Anvil, the default location is your `$HOME` directory.
    If you plan to frequently create environments in a different location than the default (such as `/anvil/project/...`), then there is an option to add directories to the `envs_dirs` list.
    To do so, you must execute:

    ```bash
    $ conda config --append envs_dirs /anvil/project/<project>/<user>/conda_envs/anvil
    ```
    
    > Note: On Anvil you can see your allocation with the `myproject` command as well as other locations with the `myquota` command.

    This will create a `.condarc` file in your `$HOME` directory if you do not have one already, which will now contain this new envs_dirs location.
    This will now enable you to use the `--name env_name` flag when using conda commands for environments stored in that specific directory, instead of having to use the `-p /anvil/project/<project>/<user>/conda_envs/env_name` option and specifying the full path to the environment.
    For example, you can do `conda activate py3711-anvil` instead of `conda activate /anvil/project/<project>/<user>/conda_envs/py3711-anvil`.

&nbsp;

## Quick-Reference Commands

* List environments:

    ```bash
    $ conda env list
    ```

* List installed packages in current environment:

    ```bash
    $ conda list
    ```

* Creating an environment with Python version X.Y:

    For a **specific path**:

    ```bash
    $ conda create -p /path/to/your/my_env python=X.Y
    ```

    For a **specific name**:

    ```bash
    $ conda create -n my_env python=X.Y
    ```
       
* Deleting an environment:

    For a **specific path**:

    ```bash
    $ conda env remove -p /path/to/your/my_env
    ```

    For a **specific name**:

    ```bash
    $ conda env remove -n my_env
    ```

* Copying an environment:

    For a **specific path**:

    ```bash
    $ conda create -p /path/to/new_env --clone old_env
    ```

    For a **specific name**:

    ```bash
    $ conda create -n new_env --clone old_env
    ```
       
* Activating/Deactivating an environment:

    ```bash
    $ source activate my_env
    $ source deactivate # deactivates the current environment
    ```

* Installing/Uninstalling packages:

    Using **conda**:

    ```bash
    $ conda install package_name
    $ conda remove package_name
    ```

    Using **pip**:

    ```bash
    $ pip install package_name
    $ pip uninstall package_name
    $ pip install --no-binary=package_name package_name # builds from source
    ```

&nbsp;

## <a name="refs"></a>Additional Resources

* [Conda User Guide](https://conda.io/projects/conda/en/latest/user-guide/index.html)
* [Anaconda Package List](https://docs.anaconda.com/anaconda/packages/pkg-docs/)
* [Pip User Guide](https://pip.pypa.io/en/stable/user_guide/)
* [Using Pip In A Conda Environment](https://www.anaconda.com/blog/using-pip-in-a-conda-environment)
