from gpflow.models import SVGP
from gpflow.decors import params_as_tensors, autoflow
from gpflow import settings
float_type = settings.float_type

class HomoscedasticPhaseOnlySVGP(SVGP):
    def __init__(self,*args, **kwargs):
        super(HomoscedasticPhaseOnlySVGP, self).__init__(*args, **kwargs)

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
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        weights = self.Y[:,-1:]
        var_exp = var_exp * weights

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
        """
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False)
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

    @autoflow((float_type, [None, None]), (float_type, [None, None]))
    def predict_density(self, Xnew, Ynew):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False)
        l = self.likelihood.predict_density(Fmean, Fvar, Ynew)
        log_num_samples = tf.log(tf.cast(num_samples, float_type))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)


