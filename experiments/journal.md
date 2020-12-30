<!-- - ###################################################### - -->
<!--- 
# [eXXXX] Template

|          |          |
|----------|----------|
|Start Date|2017-07-05|
|End Date  |2017-07-05|
|dataset   |cifar10|
|new       |cont. e0001|

## Results & Deliverables
 
## Interpretation & Conclusion
---> 
<!--- ###################################################### --->


# problem/dataset:
Classification on Cifar10. "Old problem review years later"

Issue: CIFAR10 and LaNet-5

# Description
 ## Background
Both are iconic examples of their kind. But at the moment of creation lots of things
were dictated by computational resourses of Pre-DL era in CV.
Now we can scale the training process even with consumer-grade laptops.

 ## Hypotheses/Research Questions/Ideas
 Interesting thing is to explore how differs optimal training process under current conditions from 
the original environment. What are other limitations for such problem now.

Thou, Cifar10 and LaNet are "solved" problems, they could be viewed from different angles today and, at least
serve as benchmarks for different systems (even weaker machines can handle such training procedure, as well
as mobile devices, FPGA, low-power boards, far 'edge' autonomous devices used in IoT context, etc.) : they all
can tackle this problem, but will met different limitations on their own.

 ## Plan
    + Deploy
    + Debug
    + Achieve saturation-state
    +- Investigate how to acive the same saturation satate in minimal clock-time
    ++++ Observe the irony, as deadline came and no time left...

# Experiemnt jornal

<!--- ###################################################### --->

# [e0013-e0015] Check for optimal traing loss schedule

![e13_15](fig/13_15.gif?ra=true "13_15")

| experiments|     diff       
|------------|-----------------------------------------   
|e0013      | initLR=0.01, update:10, gamma:0.1, momentum:0.9
|e0014      | initLR=0.05, update:10, gamma:0.1, momentum:0.9
|e0015      | initLR=0.02, update:5, gamma:0.3, momentum:0.9

e0014: too large initial LR, after decrease still cannot converge.
e0015: managed to reproduce e0013 success, should advance in this direction!

## Conclusion

Experiments should continue, but problem is manageble to solve in 2-3 minutes by finetuning each batch.
& use extreamly large LR on first iteration to shock-saturate the weights...

# [e0010-e0013] Search for the best learning rate init and schedule

## Interpretation

![e13_15](fig/10_13.gif?ra=true "13_15")


BS - batch size
it - iterations

| experiments|     diff                                              |
|------------|-------------------------------------------------------|        
|e0010      | BS=512, cosnt initLR = 1e-3,     
|e0011      | BS=2048, const initLR = 1e-3,         
|e0012      | BS=2048, initLR=0.1, update:10, gamma:0.1, momentum:0.9 
|e0013      | BS=2048, initLR=0.01, update:10, gamma:0.1, momentum:0.9

## Conclusion

e10, e11, e12 severely undeftit, due to less iteraitons upate with larger bathces
e13 succed to overcome this underfit issue!. Next is to do it faster

# [e0009] Check if saturation confirmes on different batch sizes

![e9](fig/e0009.gif?ra=true "e9")

## Conclusion

Saturation confirmed, on furher iterations result is not improved.

# [e0008] Select best scale for fetching data for CUDA (on given system).

## Motivation

Processing more data in single batch mean better accuracy, representativeness of data.
Iteration speed is the stopping factor. (As well as the amount of updates performed).

## Interpretation

![e8](fig/e0008.gif?ra=true "e8")

We observe that iterations over the dataset happen even faster, but up to 40th iter, we still have underfitting issue.

## Conclusion

We have to increase LearningRate

# [e0005-e0007] Baseline computed on CPU.

## Motivation

e0005, 0006: Collect baseline data and time for computation.
e0007: try to run computations on CUDA, compare if performance and problems remains.

## Interpretation

BS - batch size
it - iterations

| experiments |     diff           |  Training time  |  TestAcc best/final  | val_acc (max)|
|-------------|--------------------|-----------------|----------------------|-------------
|e0005        |BS=16, it=20      |    13m43s       |      65.9/63.77      | 78.15       |
|e0006        |BS=32, it=30      |    18m59s       |      63.2/64.79      | 83.86       |
|e0007        |BS=32, it=30, CUDA|    12m51s       |      64.0/65.03      | 84.04       |

![5_6_7](fig/e0005_e0006_e0007.gif?ra=true "5_6_7")

Used 40 iterations for e0006, as BS=32 learns obviously slower

## Conclusion

e0005, e0006:
We observe that model achieves saturation and does not learn further. Only validation accuracy grows-- overfittg.

e0007:
We see that we can perform learning on CUDA without losing accuracy compared to baseline, but only improving execution speed.
All following trials will be performed using CUDA.

# [e0004] Deploy2. Rewrite code, prepare metrics and visualisations.

Iterating with experiments requires data track through the training process, so we need to define the framework in advance, to save the costs on transforming it later.

## Deliverables

- [x] Wrap eval procedures, init routines into dedicated functions
- [x] Chained calls to provide integral training pipeline

(All relates to `/notebooks/practiceModel.ipynb`)
- [x] Following the advices from assignment added more logging levels:
  - Loss/Epoch
  - TestAccuracy/Epoch & TrainAccuracy/Epoch: combined in a single file.
- [x] Replace tracking of per-batch loss by running loss
- [X] Log-names replaced with Human-readable timestamps

- [x] Replaced bare logging with tqdm visualization, add custom tags and descriptions (Saves space for the larger number of iterations)
![progressbars](fig/e0004_progressbars.gif?ra=true "progressbars")
(something went wrong with visualized scale, but with all libraries around it, this is a minor issue)

## Conclusion
With updated model we now can shchedule experiments and iterate faset.

# [e0003] \#Debug2 + code/model revision

Code model can have potential bugs, so we need to revise it rigorously before continuing in order to secure our
future work.

# Observation & Changes:

Found out that model visualization has loops in computation graphs:

![Initial model with loops](fig/e0001_2_graph.gif?ra=true "Initial model with loops") ==> 
![Initial model without loops](fig/e0002_fixed_graph.gif?ra=true "Initial model without loops")

MaxPool layer is reused in model declaration. It does not contain trainable parameter, but
we consider that it is better to transform it into canonical view.

Instead of one 'pool' layer we add 'pool1' and 'pool2'. Graph became acyclic.

### Performance
Training time [cpu]: ~ 4 min. (5 epochs)

Accuracy [val]:    61.0 %

## Deliverables

- [x] updated notebooks
  - `/notebooks/practiceModel.ipynb` - for each max pooling operation add dedicated layer

## Interpretation & Conclusion 

| experiments |     diff           | accuracy     |
|-------------|--------------------|--------------|
|e0002        |#'pool' and 'pool'  |    61.0 %    |
|e0003        |#'pool1' and 'pool2'|    61.02%    |

Our changes preseved model properties, nothing is broken. Code is prepearing for future experiments iterations.

# [e0002] \#Dubug

We see that loss drops dramatically, source of bug is pretty obvious: it is conserned with loss sheduler.

## Motivation

Find the bug in current setting to continue work

## Description

### Performance
Training time [cpu]: ~ 4 min. (5 epochs)

Accuracy [val]:    61.0 %

## Deliverables

- [x] updated notebooks
  - `/notebooks/practiceModel.ipynb` - fixed the issue with incorrect loss sceduler

## Interpretation

As we have similar notebooks, models seem valid, parameter altering does not give results, - then one with problem has some bug.

![Loss vs Epoch](fig/e0002_loss.gif?ra=true "Loss vs Epoch")

# [e0001] \#Deploy
<!-- Training LeNet-like model on cifar-10 dataset -->

## Motivation

Deploy model training on local machine, adapt familiar examples to newly created environment.
(Expected behavior is observed on Google Colab, we have to make the same)

The goal here is to train simple model on cifar10 dataset without gpu support.

## Description

Training time [cpu]: ~ 5 min. (5 epochs)

Accuracy [val]: 9.27 % (worse than random guess)

## Deliverables

- [x] added notebooks
  - `/notebooks/cifar10_tutorial.ipynb` works and produce meaningfull results, leave commented code to deploy missing libs
  - `/notebooks/practiceModel.ipynb` launches, but loss function does not fall as expected

## Interpretation

As we have similar notebooks, models seem valid, parameter altering does not give results, - then one with problem has some bug.

![Loss vs Epoch](fig/e0001Loss.gif?ra=true "Loss vs Epoch")


## Conclusion
We succesfully reproduce Colab notebook behavior from course practice task: it is both broken.
Need to find the source of problem.

<!--- ###################################################### ---> 

# Summary

We've got practice of systematic working on DeepLearning for computer vision task.
Using systematic approach helps greatly to tackle problem with unintuitive properties.

Sequence of Hypothesis-Experiment-Validation is good for long lasting problems, 
but here it seems that it was not the best, as it slows down progress for cases where
iterations are fast. Still it would shine on the later stages of this project, when 
we shoul select and combine outcomes from sellected approaches and "evolve" solution based
on best former examples.

Considering stated question of the moder systems limitations for Pre-DL resourse-bound problems:
 - Amount of GPU memory is not an issue at all.
 - Bottleneck was at data fetching stage: iterations were performed faster then 
 data were supplied.
 - The main problem limiting the optimisation procedure was small amound of data (sic!),
 issue is in that fact that we can fit all the data in few batches, and number of updates
 will be too small to achieve good progress in learning, so we have to pay much more
 attention to the procedure of LR scheduling.
 - Problem statement becomes closer to One-shol learning problem, or similar.