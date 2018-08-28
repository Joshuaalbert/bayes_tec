import tensorflow as tf
from gpflow.models import SVGP
from gpflow.decors import params_as_tensors, autoflow
from gpflow import settings
float_type = settings.float_type

class HomoscedasticPhaseOnlySVGP(SVGP):
    def __init__(self,P,*args, **kwargs):
        super(HomoscedasticPhaseOnlySVGP, self).__init__(*args, **kwargs)
        self.P = P

    @params_as_tensors
    def _build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean, fvar = self._build_predict(self.X, full_cov=False, full_output_cov=False)

        # Get variational expectations.
        freqs = self.Y[:,-1:] # N, 1
        # tile to matach Y
        freqs = tf.tile(freqs,(1, self.P))# N, num_latent
        Y = self.Y[:,:self.P] #N, num_latent
        var_exp = self.likelihood.variational_expectations(fmean, fvar, Y, freqs = freqs)

        weights = self.Y[:,self.P:2*self.P]
        var_exp = var_exp * weights

        # re-scale for minibatch sizenum_gauss_hermite_points
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

    @autoflow((float_type, [None,None]), (float_type, []))
    def predict_phase(self, Xnew, eval_freq=140e6):
        """
        Draws the predictive mean and variance of dTEC at the points `Xnew`
        X should be [N,D] and this returns [N,num_latent], [N,num_latent]
        eval_freq : float the eval freq for phase
        """
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False)
        self.likelihood.eval_freq = eval_freq
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

    @autoflow((float_type, [None, None]), (float_type, [None, None]))
    def predict_density(self, Xnew, Ynew):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, full_output_cov=False)
        # Get variational expectations.
        freqs = Ynew[:,-1:] # N, 1

        # tile to matach Y
        freqs = tf.tile(freqs,(1, self.P))# N, num_latent
        l = self.likelihood.predict_density(Fmean, Fvar, Ynew[:,:self.P], freqs=freqs)
        return l 
