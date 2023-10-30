# Access Anvil and Clone Repository

Follow the instructions below to login to OLCF's Anvil compute system and grab a copy of the code we'll be using.

<hr>

&nbsp;

## Get started by logging into Anvil using SSH. 
Use the userid and passcode that you setup for Anvil to login. 
```bash
ssh userid@anvil.rcac.purdue.edu
```
&nbsp;

<hr>

&nbsp;
## Clone GitHub Repository for this Event
GitHub is a code-hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere. We use GitHub to develop and host the code for this event. 

You will need to `clone` a copy of our repository to your `home` directory so that you can complete all of our challenges. Use the following commands to complete this step:
```bash
$ cd /ccs/home/userid
$ git clone https://github.com/purduercac/hands-on-with-anvil.git
```

Check that you can list the files in your current directory to see the repository directory: 
```bash
$ ls
hands-on-with-anvil
```

Finally, move into that directory:
```bash
$ cd hands-on-with-anvil
```

&nbsp;

<hr>

&nbsp;
## Congratulations! You've completed your first challenge. 
You can now move on to other [challenges](../). 

``` 
New to Unix? Start here.
```
- [Basic_Unix_Vim](Basic_Unix_Vim)
- [Basic_Workflow](Basic_Workflow)
- [Password_in_a_Haystack](Password_in_a_Haystack)

```
Ready for HPC and Parallel Code?
```
- [Jobs_in_Time_Window](Jobs_in_Time_Window)
- [MPI_Basics](MPI_Basics)
- [OpenMP_Basics](OpenMP_Basics)

```
Learn to use those AMD GPUs!
```
- [OpenMP_Offload](OpenMP_Offload)
- [GPU_Data_Transfers](GPU_Data_Transfers)
- [GPU_Matrix_Multiply](GPU_Matrix_Multiply)
- [GPU_Profiling](GPU_Profiling)

```
Or, Visualize all the data!
```
- [Python_Conda_Basics](Python_Conda_Basics)
- [Python_Parallel_HDF5](Python_Parallel_HDF5)
- [Python_Pytorch_Basics](Python_Pytorch_Basics)
- [Python_Cupy_Basics](Python_Cupy_Basics)

&nbsp;

<hr>

&nbsp;


Not Functioning
- [srun_Job_Launcher](jsrun_Job_Launcher)
- [Find_the_Compiler_Flag](Find_the_Compiler_Flag)
- [Parallel_Scaling_Performance](Parallel_Scaling_Performance)
- [Score-P_and_Vampir_Basics](Score-P_and_Vampir_Basics)



