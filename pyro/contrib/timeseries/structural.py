# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

import pyro
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, init_to_sample
from pyro.nn.module import PyroModule, PyroSample
from pyro.optim import ClippedAdam


class GlobalTrend(PyroModule):
    def __init__(self, timescale, min_timescale):
        self.timescale = PyroSample()


class Seasonal(PyroModule):
    def __init__(self, seasonality):
        self.seasonality = seasonality

    def forward(self, covariates):
        raise NotImplementedError("TODO")


class Regression(PyroModule):
    def __init__(self, regressor):
        self.regressor = regressor

    def forward(self, covariates):
        raise NotImplementedError("TODO")


class StructuredModel(PyroModule):
    def __init__(self, signal, noise):
        self.signal = signal
        self.noise = noise

    def forward(self, data, covariates):
        assert len(data) == len(covariates)
        signal = self.signal(covariates)
        noise = data - signal
        noise_dist = self.noise._get_dist()
        pyro.sample("obs_noise", noise_dist, obs=noise)

    def forecast(self, data, covariates):
        assert len(data) < len(covariates)
        raise NotImplementedError("TODO")
        noise_dist = self.noise._get_dist()
        return pyro.sample("forecast_noise", noise_dist)


class VariationalForecaster(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.guide = AutoDiagonalNormal(model, init_fn=init_to_sample)

    def fit(self, data, covariates, learning_rate=0.01, num_steps=1000):
        elbo = Trace_ELBO()
        optim = ClippedAdam({"lr": learning_rate, "betas": (0.9, 0.99),
                             "lrd": 0.1 ** (1 / num_steps)})
        svi = SVI(self.model, self.guide, optim, elbo)
        losses = []
        for _ in range(num_steps):
            loss = svi.step(data, covariates) / data.numel()
            losses.append(loss)
        return losses

    def forecast(self, data, covariates):
        with poutine.trace() as tr:
            self.guide(data, covariates)
        with poutine.replay(trace=tr.trace):
            return self.model.forecast(data, covariates)
