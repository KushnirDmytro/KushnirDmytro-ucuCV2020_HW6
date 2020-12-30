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
 задача має бути Computer Vision related
(images or video), яка вирiшується методами Deep Learning.
наприклад - classification/cifar10

# possible experiments:
- Training XXXXX model on YYYYY dataset
- Experimenting with different initialization methods
- Experimenting with different optimization algorithms
- Data augmentation
- Experimenting with different pre-trained models
- Experimenting with new layers

# Description [± 1 page]
 ## Background
 ## Hypotheses/Research Questions/Ideas
 ## Plan
    ### deploy
    ### debug
    ### PROFIT

# Experiemnt jornal
[3-8 pages]

# Summary
[1 page]

<!--- ###################################################### --->

# [e0039, e0040] Adam (second try)

|          |          |
|----------|----------|
|Start Date|2017-07-25|
|End Date  |2017-07-26|
|dataset   |cifar10|

## Motivation

Try adam again (with best settings from the previous experiment [e0038])

 
## Interpretation & Conclusion

All experiments share setting from [e0038]

| experiments |     diff    | best val. acc. |
|-------------|-------------|----------------|
|e0039        |  with scheduler |     89.17  |
|e0040        |  no scheduler   |     88.88  |

![](fig/e0038,e0039,e0040_val_acc_epoch.png)


Observations:

* adam is worse than sgd
* scheduler helps in this case (wat unexpected) 

<!--- ###################################################### --->

# [e0033-e0034, e0036-e0038] make best of VGG

|          |          |
|----------|----------|
|Start Date|2017-07-22|
|End Date  |2017-07-24|
|dataset   |cifar10|

## Motivation

To get as much as possible in terms of accuracy on vgg net, 
playing with learning schedule (optim.lr_scheduler.StepLR) and other parameters.


## Results & Deliverables & Observation

[time per epoch]: ~ 3 min
[accuracy]: 91.24 %
  
 
## Interpretation & Conclusion

All experiments share the following:
* NET MODEL: VGG_BN_3x32x32
* optim.lr_scheduler.StepLR
* lr_schedule_gamma: 0.1

| experiments |     diff    | best val. acc. |
|-------------|-------------|----------------|
|e0033        |  lr: 0.1 |                                                                     57.62 |
|e0034        |  lr: 0.01; weights: experiments/logs/e0031_model_best.pth  |                   88.85 |
|e0036        |  Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201];  |      89.52 |
|e0037        |  the same setting as e0036; weight_decay: 0.0005 |                             90.17 |
|e0038        |  the same setting as e0037; train: augment_simple |                            91.24 |

![](fig/e0033,e0034,e0038_train_loss_iter.png)
![](fig/e0034,e0036,e0037,e0038_train_acc_epoch.png) 
![](fig/e0034,e0036,e0037,e0038_val_acc_epoch.png) 

Observations:

* high init lr has bad consequences on training (e0033);
* normalization coefficients are important;
* weight_decay plays a role;
* simple augmentation is better than more complex one;

<!--- ###################################################### --->

# [e0002-e0004] LeNet-like model with different number of epochs

|          |          |
|----------|----------|
|Start Date|2017-07-05|
|End Date  |2017-07-05|
|dataset   |cifar10|
|new       |cont. e0001|

## Motivation

* To extend the training period to larger epochs.
* To 'repeat' the experiment with the same parameters (as in e0001) 
and compare.
* Test load from checkpoint functionality.

This *baseline* can serve as a starting point for further investigations 
(different learning rates, different initialization, different optimizations)
on this model.   

## Results & Deliverables

Accuracy [val]: 66.22 %

## Interpretation & Conclusion 

| experiments |     diff |
|-------------|----------|
|e0001        |#epochs = 30|
|e0002        |#epochs = 50|
|e0003        |#epochs = 80|
|e0004        |#epochs = 80, start from e0002 checkpoint|

![Acc vs Epoch](fig/e0001,e0002,e0003,e0004_acc_epoch.png?ra=true "Acc vs Epoch")
![Loss vs Epoch](fig/e0001,e0002,e0003,e0004_loss_epoch.png?ra=true "Acc vs Epoch")

**observations:**

* validation accuracy starts to saturate around 40 epoch.
* validation loss starts to degradate around 50 epoch.
At this point there is a clear sign of over-fitting. 
* different experiments are consistent.
* load from checkpoint works fine.

For further investigation the #epoch = 50 seems the most reasonable. 

**weirdness:**:

* After 50 epoch the validation loss increase, but the validation accuracy is the same.

<!--- ###################################################### --->


# [e0002] \#Deploy2. Advance: add gpu support.

# [e0002] \#Debug

We see that loss drops dramatically, source of bug is pretty obvious: it is conserned with loss sheduler
Difference with orignal tuto

## Motivation

Find the bug in current setting to continue work

## Description

### Performance
Training time [cpu]: ~ 4 min. (5 epochs)

Accuracy [val]:    61.0 %

## Deliverables

- [x] updated notebooks
  - `/notebooks/practiceModel.ipynb` - fixed the issue with incorrect loss sceduler
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

<!-- As we can see on the plot,  -->

<!-- ![Acc vs Epoch](fig/e0001_acc_epoch.png?ra=true "Acc vs Epoch") -->

<!-- validation accuracy starts to saturate around 22 epoch,  -->
<!-- suggesting that we have exploited all capacity of this simple model.   -->

<!-- On the other hand the loss is still decreasing,  -->
<!-- suggesting to run other experiment with this model for more epoch. -->

![Loss vs Epoch](fig/e0001Loss.gif?ra=true "Loss vs Epoch")


## Conclusion
We succesfully reproduce Colab notebook behavior from course practice task: it is both broken.
Need to find the source of problem.

<!-- It seems that more powerful model is needed.  -->

<!--- ###################################################### ---> 
