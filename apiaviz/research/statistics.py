"""Paired benchmark inference: routes are clusters, seeds are crossed repeats."""
import itertools
import numpy as np


def sign_flip_pvalue(route_effects):
    values=np.asarray(route_effects,dtype=float)
    if values.ndim!=1 or len(values)==0 or len(values)>20 or not np.isfinite(values).all():
        raise ValueError("Expected 1--20 finite route-level effects")
    signs=np.asarray(list(itertools.product((-1.,1.),repeat=len(values))))
    null=np.abs((signs*values).mean(axis=1))
    return float(np.mean(null>=abs(values.mean())-1e-12))


def holm_adjust(pvalues):
    p=np.asarray(pvalues,dtype=float)
    if p.ndim!=1 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError("Expected finite p-values in [0,1]")
    order=np.argsort(p,kind="stable")
    corrected=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    result=np.empty_like(p);result[order]=corrected
    return result.tolist()


def paired_effect(differences,repeats=20000,seed=20260929):
    """Positive differences favour ApiaViz. Input is ant x matched wiring seed.

    P-value conditions on the five chosen wiring seeds. Route-only bootstrap
    does too. Crossed bootstrap additionally resamples seed columns and may be
    conservative; it is a sensitivity analysis with only five seeds.
    """
    values=np.asarray(differences,dtype=float)
    if values.ndim!=2 or min(values.shape)<2 or not np.isfinite(values).all():
        raise ValueError("Expected a complete finite route-by-seed matrix")
    rng=np.random.default_rng(seed)
    ri=rng.integers(values.shape[0],size=(repeats,values.shape[0]))
    ci=rng.integers(values.shape[1],size=(repeats,values.shape[1]))
    route_mean=values.mean(1)
    route_boot=route_mean[ri].mean(1)
    crossed=values[ri[:,:,None],ci[:,None,:]].mean((1,2))
    return dict(effect=float(values.mean()),
                route_ci95=np.quantile(route_boot,[.025,.975]).tolist(),
                crossed_ci95=np.quantile(crossed,[.025,.975]).tolist(),
                p_raw=sign_flip_pvalue(route_mean),route_effects=route_mean.tolist(),
                seed_effects=values.mean(0).tolist(),routes_favouring_apia=int((route_mean>0).sum()),
                seeds_favouring_apia=int((values.mean(0)>0).sum()))
