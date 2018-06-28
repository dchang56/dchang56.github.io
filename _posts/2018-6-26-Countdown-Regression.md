---
published: true
title: Countdown Regression: sharp and calibrated survival predictions
use_math: true
category: Survival Analysis
layout: default
---

Personalized probabilistic forecasts of time to event

Inspired by ideas from the meteorology literature

paradigm of maximizing sharpness of prediction distributions, subject to calibration

In regression, it's been shown that optimizing the continuous ranked probability score (CRPS) instead of maximum likelihood 
leads to sharper prediction distributions while maintaining calibration.

Introducing: the Survival-CRPS, a generalization of the CRPS to the time to event setting

Evaluation: the Survival-AUPRC evaluation metric (analog to area under precision-recall curve)

Method: Build RNN for mortality prediction using EHR for millions of patients

Main contribution is the benefit of training by the Survival-CRPS objective instead of maximum likelihood.

Survival prediction: predicting time to event by estimating a distribution over future time
Traditional survival analysis models such as the Cox proportional hazards model or accelerated failure time model are capable
of handling data with censored observations.

But, 1. traditional models typically make strong assumptions; 2. challenges of low prevalence often arise when these methods 
are applied to large-scale
observational datasets with heavy censoring, i.e. EHR; 3. these SA methods are typically evaluated as point estimates of risk, 
such as 10-year
probabilities of events, rather than holistic measures of quality of the predicted distributions. 

Common metrics of evaluation include the C-statistic, log-l1 loss, and mean-squared-error. While useful for the purposes of 
relative risk stratification,
model comparisons made using point estimates leaves the quality of uncertainty in predicted distributions left unmeasured.
If a point prediction
is way off, it is penalized by the same amount whether the model was confident or not (whether the predicted distribution 
had low or high variance)

In contrast, forecasts in the field of meteorology are typically made as full prediction distributions over all weather
conditions given past and current observations. Evaluation of predictive performance is assessed by the paradigm of maximizing
the sharpness of the predictive distribution, subject to calibration. The intuition behind is that probabilities have to be calibrated
in order to be correct. The usefulness of a prediction distribution lies in its sharpness, or how well its mass concentrates
In summary, uncalibrated predictions (sharp or not) are useless; calibrated but not-sharp predictions are correct but less useful,
and calibrated and sharp distributions are most useful.

To improve the sharpness of prediction distributions in the survival setting, propose use of proper scoring rules beyond
maximum likelihood as the training objective.
Proper scoring rules are known to measure calibration, and any model trained with proper scoring rule will tend to maintain calibration
Continuous ranked probability score (CRPS); generalized to survival setting: Survival-CRPS, with right-censored and interval-censored extensions

This is the first time any scoring rule other than ML has been successfully applied to a large-scale survival prediction task

*Summary of contributions*.
(1) introduce the proper scoring rule Survival-CRPS as an objective in survival prediction
(2) a new metric Survival-AUPRC (inspired by paradigm of maximizing sharpness subject to calibration, to holistically measure the
quality of a prediction distribution with respect to a possibly censored outcome
(3) practical recommendations for mortality prediction task: use log-normal parameterization and interval censoring when training
(4) demonstrate their efficacy by training a deep RNN for accurate survival prediction of patient mortality using EHR data


# 2 Countdown Regression

Parametric survival prediction methods model the time to event of interest with a family of probability distributions,
uniquely identified by the distribution parameters.
The survival function, denoted $S(t) : [0,\infty) \to [0,1]$, is a monotonically decreasing function over the positive reals with 
S(0)=1 and $\lim_{t\to\infty} S(t)=0$


## 2.1 Survival-CRPS: proper scoring rules as training objectives

A scoring rule is a measure of the quality of a probabilistic forecast. A forecast over a continuous outcome is a probability density function over all possible outcomes, $\hat{f} with corresponding cumulative density function \hat{F}.$ In reality, we observe some actual outcome, y. 
A scoring rule S takes a predicted distribution and an actual outcome, and returns a loss $S(\hat{F},y)$.

It is considered a *proper scoring rule* if for all possible distributions G,
$$\mathbb{E}_{y\sim\hat{F}}\big[S(\hat{F},y)] \leq \mathbb{E}_{y\sim\hat{F}}\big[S(G,y)]$$, and strictly proper when equality holds iff $\hat{F} = G$

A proper scoring rule is one in which the expected score is minimized by the distribution with respect to which the expectation is taken. Intuitively, it encourages a model for being honest by predicting what it actually believes. It naturally forced the model to output calibrated probabilities.

There are many commonly used proper scoring rules. Most widely used is the logarithmic scoring rule, equivalent to the maximum likelihood objective:

$$S_{MLE}(\hat{F},y) = -log \hat{f}(y)$$

With censorship, we maximize the density for observed outcomes, and tail or interval mass for censored outcomes

$$S_{MLE-RIGHT}(\hat{F},(y,c))) = -log\big((1-c)\hat{f}(y) + c\hat{S}(y)\big)$$
$$S_{MLE-INTVL}(\hat{F},(y,c,T)) = -log\big((1-c)\hat{f}(y) + c(\hat{F}(T)-\hat{F}(y))\big)$$

where c is the censorship indicator.

However, the log scoring rule is asymmetric, and harshly penalizes predictions that are wrong yet confident. This results in the training process becoming sensitive to outliers, and in general conservative in prediction-making (i.e. hesitant to make sharp predictions).

![alt text][fig
