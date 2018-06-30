---
published: true
title: Countdown Regression
use_math: true
category: Literature Review - DLEHR
layout: default
---

# Table of Contents

* TOC
{:toc}

# Introduction

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
The survival function, denoted $S(t) : \[0,\infty) \to \[0,1\]$, is a monotonically decreasing function over the positive reals with 
S(0)=1 and $\lim_{t\to\infty} S(t)=0$


## 2.1 Survival-CRPS: proper scoring rules as training objectives

A scoring rule is a measure of the quality of a probabilistic forecast. A forecast over a continuous outcome is a probability density function over all possible outcomes, $\hat{f}$ with corresponding cumulative density function $\hat{F}.$ In reality, we observe some actual outcome, y. 
A scoring rule S takes a predicted distribution and an actual outcome, and returns a loss $S(\hat{F},y)$.

It is considered a *proper scoring rule* if for all possible distributions G,
$$\mathbb{E}_{y\sim\hat{F}} \big[S(\hat{F},y)] \leq \mathbb{E}_{y\sim\hat{F}}\big[S(G,y)]$$ , and strictly proper when equality holds iff $\hat{F} = G$

A proper scoring rule is one in which the expected score is minimized by the distribution with respect to which the expectation is taken. Intuitively, it encourages a model for being honest by predicting what it actually believes. It naturally forced the model to output calibrated probabilities.

There are many commonly used proper scoring rules. Most widely used is the **logarithmic** scoring rule, equivalent to the maximum likelihood objective:

$$S_{MLE}(\hat{F},y) = -log \hat{f}(y)$$

With censorship, we maximize the density for observed outcomes, and tail or interval mass for censored outcomes

$$S_{MLE-RIGHT}(\hat{F},(y,c))) = -log\big((1-c)\hat{f}(y) + c\hat{S}(y)\big)$$
$$S_{MLE-INTVL}(\hat{F},(y,c,T)) = -log\big((1-c)\hat{f}(y) + c(\hat{F}(T)-\hat{F}(y))\big)$$

where c is the censorship indicator.

However, the log scoring rule is asymmetric, and harshly penalizes predictions that are wrong yet confident. This results in the training process becoming sensitive to outliers, and in general conservative in prediction-making (i.e. hesitant to make sharp predictions).

![figure2][fig2]

[fig2]: {{ site.url }}/assets/fig2.png

Figure 2: graphical intuition for Survival-CRPS scoring rule. For uncensored observations, we minimize mass before and after the observed time of event. For right-censored, we minimize mass before observed time of censoring. For interval-censored, we minimize mass before observed time of censoring, and mass after the time by which event must have occurred. 

An alternative proper scoring rule for forecasts over continuous outcomes is the CRPS, defined as

$$ S_{CRPS}(\hat{F},y) = \int_{-\infty}^{\infty} \big(\hat{F}(z) - 1\{z\geq{y}\}\big)^{2}dz$$

$$ = \int_{-\infty}^{y} \hat{F}(z)^{2}dz + \int_{y}^{\infty} (1-\hat{F}(z))^{2}dz $$


CRPS has been used in regrssion as an objective function that yields sharper predicted distributions compared to ML, while maintaining calibration. 
Note the two integral terms in the latter epression; they correspond to the two shaded regions in Fig2a. CRPS score is reduced to 0 when the predicted distribution places all the mass on the point of true outcome (when shaded region vanishes).

In the context of time to event predictions, they propose the **Survival-CRPS**, which accounts for the possibility of right-censored or interval-censored data

$$ S_{CRPS-RIGHT}(\hat{F},(y,c)) = \int_{0}^{\infty} \big(\hat{F}(z)1\{z\leq{y}\cup c=0\} - 1\{z\geq{y}\cap c=0\}\big)^{2}dz$$

$$ = \int_{0}^{y} \hat{F}(z)^{2}dz + (1-c)\int_{y}^{\infty} (1-\hat{F}(z))^{2}dz $$

$$ S_{CRPS-INTVL}(\hat{F},(y,c,T)) = \int_{0}^{\infty} \big(\hat{F}(z)1\{\{z\leq{y}\cup c=0\}\cup z\geq{T}\} - 1\{\{z\geq{y}\cap c=0\}\cup z\geq{T}\}\big)^{2}dz$$

$$ = \int_{0}^{y} \hat{F}(z)^{2}dz + (1-c)\int_{y}^{T} (1-\hat{F}(z))^{2}dz + \int_{T}^{\infty} (1-\hat{F}(z))^{2}dz $$

Note: when c=0 (i.e. uncensored), both of them are equivalent to CRPS. Again, intuition is better understood by looking at second expression: each integral maps to corresponding shaded region in Fig2b and c. 

Survival-CRPS is identical to original CRPS when time of event is uncensored.
For censored outcomes, it penalizes the predicted mass that occurs before the time of censoring (and for interval censored, mass after time by which the event must have occurred). Both variants are proper scoring rules. 

## 2.2 Evaluation by sharpness subject to calibration

Calibration assesses how well forecasted event probabilities match up to observed event probabilities. 
There is no widely accepted method for evaluating the calibration of a set of entire prediction distributions, over multiple time frames, in the survival setting. 

(D-calibration: recently proposed method for holistic evaluation, but relies on assuming true times of death are uniformly distributed past times of censoring, which means when censored obvs outnumber uncensored obvs, this can lead to overly optimistic assessments of calibration)

There is also Kaplan-Meier estimate, but this is also limited in heavily censored settings (quantiles in the tail of predicted dist have few uncensored obvs)

Proposal to measure calibration:
compare predicted cumulative densities against observed event frequencies, evaluated at quantiles of predicted cumulative density. Right-censored obvs are removed from consideration in quantiles that correspond to times after their points of censoring. Interval-censored obvs are removed from consideration in quantiles that correspond to times after censoring, but reintroduced in quantiles that correspond to times past the time by which event must have occurred. 

Subject to calibration, we want prediction distributions that are sharp (concentrated).There are several ways to measure sharpness, such as variance or entropy. They use the coefficient of variation (CoV) as a reasonable measure of sharpness, defined as the ratio of one sd to the mean

$$CoV(\hat{F}) = \frac{ \sqrt{Var[\hat{F}]} } { \mathbb{E}[\hat{F}]}$$

## 2.3 Survival-AUPRC - holistic evaluation of a time to event prediction distribution

A metric that measures how concentrated the mass of the prediction distribution is around the true outcome, robust to miscalibration.

Uncensored case:
as an analog to precision, we consider intervals relative to the true time of event, defined by ratios. 
  i.e. a region of precision 0.9 around an event at time y is the interval $[0.9y, y/0.9]$
the analogy to recall is the mass assigned by the predicted distribution over this interval, $\hat{F}(y/0.9) - \hat{F}(0.9y)$.

By exploring the full range of precision from 0 to 1, we get the Survival Precision Recall Curve. The area under this curve measures how quickly predicted mass concentrates around the true outcome as we expand the precision window.

$$Survival-AUPRC_{UNCENSORED}(\hat{F},y) = \int_{0}^{1} (\hat{F}(y/t) - \hat{F}(yt))dt$$

Max is 1, lowest is 0 (infinitely dispersed).The mean of all Survival-AUPRC scores across examples provides an overall measure of the quality of the predictions.
This only applies to uncensored outcomes.

Censored case:
Same idea, but with the right end of precision intervals defined wrt the time by which the event must have occurred in the interval-censored case (infinity in right-censored case).

$$Survival-AUPRC_{RIGHT}(\hat{F},y) = \int_{0}^{1} (1-\hat{F}(yt))dt$$

$$Survival-AUPRC_{INTVL}(\hat{F},y,T) = \int_{0}^{1} (\hat{F}(T/t) - \hat{F}(yt)dt$$

## 2.4 Recurrent neural network model

Input: sequence of features (patient info from EHR)
Want to predict parameters of a parametric prob dist $\hat{F}$ over time to death at each timestep.
The distributions that are ouput in each timestep are used to construct an overall loss.

$$\mathcal{L}_{RIGHT} = \sum_{i=1}^{N}\sum_{t=1}^{T^{(i)}} S_{RIGHT} \big(\hat{F}_{RNN_{\theta}\{x_{1:t}^{(i)}\}}, (y_{t}^{(i)},c^{(i)})\big) $$

$$\mathcal{L}_{INTVL} = \sum_{i=1}^{N}\sum_{t=1}^{T^{(i)}} S_{INTVL} \big(\hat{F}_{RNN_{\theta}\{x_{1:t}^{(i)}\}}, (y_{t}^{(i)},c^{(i)},T_{t}^{(i)})\big) $$

where N is the total number of patients, T is the sequence length for the patient, and $\hat{F}_{RNN_{\theta}}$ is the distribution parametrized by the output of the RNN. 

Called Countdown Regression because it is sequential and the predicted times to event are monotonically decreasing.

## 2.5 Choice of log-normal noise distribution

Common parametric distributions over time to event used in traditional SA models: Weibull, log-normal, log-logistic, gamma (in order to be sufficiently expressive in model space, we seek distributions with at least two parameters)

Log-normal:doesn't involve Beta function or the pattern $(y/p_{1})^{p_{2}}$ (which make them highly sensitive to inputs and numerical instability issues).

# 3 Experiments

Four different training objectives: ML (right and interval), Survival-CRPS right and interval
Max lifespan = 120 years

The input at each timestep consists of both real valued (i.e. age) and discrete valued (ICD codes) data.
Discrete data is embedded into a trainable real-valued vector space, and vectors corresponding to the codes recorded at a given timestep are combined into a weighted mean by a soft self-attention mechanism.
Then, all real valued inputs are appended to the averaged embedding vector
Use Swish activation function and layer normalization at every layer. 

After the recurrent layers, the network has multple branches, one per parameter of the survival distribution (lognormal has sigma and mu). The final layer in each branch has scalar output. Bernoulli dropout at all fully connected layers, Variational RNN dropout in recurrent layers. Adam optimizer.

## 3.1 Data

## 3.2 Results

All models are reasonably well-calibrated. Survival-CRPS with interval censoring yields the sharpest prediction distributions.
Inspecting the mass past 120 years of age shows that a naively trained prediction model with maximum likelihood can assign more than 75% of the mas to unreasonable regions, largely due to low prevalence of uncensored examples, which is typical in real world EHR data sets. 

By predicting an entire distribution over time to death, the same model can be used to make classification predictions at various time points. (figure 4)

# 4 Related Work

# 5 Conclusion

Should explore objectives beyond maximum likelihood and evaluation metrics that assess the holistic quality of predicted distrubitions, instead of point estimates.

For evaluation: Survival-AUPRC metric captures the degree to which a prediction distribution concentrates around the observed time of event

By predicting an entire distribution for time-to-event, circumvent issues associated with binary classification.
Still yields accurate predictions when evaluated as dichotomous outcomes at specific times.

This is really awesome because it addresses a lot of challenges associated with survival analysis and time-to-event predictions. Not only does it estimate an entire distribution for each patient, but it's also capable of making dichotomous classifaction at specific time points, which is very useful and impressive.



