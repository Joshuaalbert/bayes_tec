
import tensorflow as tf
from gpflow.models import SVGP
from gpflow.decors import params_as_tensors, autoflow
from gpflow import settings
from gpflow.params import DataHolder
from gpflow.params import Minibatch
float_type = settings.float_type

from ..logging import logging

class HeteroscedasticPhaseOnlySVGP(SVGP):
    def __init__(self,Y_var,freqs,*args, **kwargs):
        minibatch_size = kwargs.get('minibatch_size',None)
        if minibatch_size is None:
            Y_var = DataHolder(Y_var)
            freqs = DataHolder(freqs)
        else:
            Y_var = Minibatch(Y_var, batch_size=minibatch_size, seed=0)     
            freqs = Minibatch(freqs, batch_size=minibatch_size, seed=0)     

        super(HeteroscedasticPhaseOnlySVGP, self).__init__(*args, **kwargs)

        self.Y_var = Y_var
        self.freqs = freqs
        
    @params_as_tensors
    def _build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean, fvar = self._build_predict(self.X, full_cov=False, full_output_cov=False)
        

#        cov = self.kern.K(self.X, full_output_cov=False)#P,N,N
#        tf.summary.image('Kxx',cov[..., None])


        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y, self.Y_var, self.freqs)

#        var_exp = var_exp * self.weights

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.float_type) / tf.cast(tf.shape(self.X)[0], settings.float_type)

        return tf.reduce_sum(var_exp) * scale - KL

    @autoflow((float_type, [None,None]))
    def predict_dtec(self, Xnew):
        """
        Draws the predictive mean and variance of dTEC at the points `Xnew`
        X should be [N,D] and this returns [N,num_latent], [N,num_latent]
        """
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False)
        return Fmean*self.likelihood.tec_scale, Fvar*self.likelihood.tec_scale**2

    @autoflow((float_type, [None,None]))
    def predict_phase(self, Xnew):
        """
        Draws the predictive mean and variance of dTEC at the points `Xnew`
        X should be [N,D] and this returns [N,num_latent], [N,num_latent]
        eval_freq : float the eval freq for phase
        """
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False)
        ### BROKEN need to predict Y_var to get predictive phase variance, though conditional mean is fine
        Y_var = None
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

    @autoflow((float_type, [None, None]), (float_type, [None, None]), (float_type, [None, None]), (float_type, [None, None]))
    def predict_density(self, Xnew, Ynew, Yvarnew, freq):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, full_output_cov=False)
        # Get variational expectations.

        # tile to matach Y
        l = self.likelihood.predict_density(Fmean, Fvar, Ynew, Yvarnew, freq)
        return l 
