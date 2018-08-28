import gpflow as gp
import numpy as np
import tensorflow as tf
from gpflow import params_as_tensors
from gpflow import transforms
from gpflow.params import Parameter
from gpflow.likelihoods import Likelihood
from gpflow import settings
from gpflow.quadrature import ndiagquad

try:
    @tf.RegisterGradient('WrapGrad')
    def _wrap_grad(op,grad):
        phi = op.inputs[0]
        return tf.ones_like(phi)*grad
except:
    pass#already defined

def wrap(phi):
    out = tf.atan2(tf.sin(phi),tf.cos(phi))
    with tf.get_default_graph().gradient_override_map({'Identity': 'WrapGrad'}):
        return tf.identity(out)


class WrappedPhaseGaussianEncoded(Likelihood):
    def __init__(self, tec_scale=0.01, freq=140e6, num_gauss_hermite_points=20, variance=1.0, name=None):
        super().__init__(name=name)
        self.variance = Parameter(
            variance, transform=transforms.positive, dtype=settings.float_type)
        self.tec_scale = tec_scale
        self.num_gauss_hermite_points = num_gauss_hermite_points
        self.freq = tf.convert_to_tensor(freq,dtype=settings.float_type,name='test_freq') # frequency the phase is calculated at for the predictive distribution
        self.tec_conversion = tf.convert_to_tensor(tec_scale * -8.4480e9,dtype=settings.float_type,name='tec_conversion') # rad Hz/ tecu
        self.tec2phase = tf.convert_to_tensor(self.tec_conversion / self.freq,dtype=settings.float_type,name='tec2phase')
        
    @params_as_tensors
    def logp(self, F, Y, freqs, **kwargs):
        """The log-likelihood function."""
        assert freqs is not None
        #freqs = Y[:,-1:]
        #Y = Y[:,:self.num_latent]
        # N,1
        tec2phase = self.tec_conversion/freqs
        phase = F*tec2phase
        dphase = wrap(phase) - wrap(Y) # Ito theorem
        
        arg = tf.stack([-0.5*tf.square(dphase + 2*np.pi*k)/self.variance - 0.5 * tf.log((2*np.pi) * self.variance) \
                for k in range(-2,3,1)],axis=-1)
        return tf.reduce_logsumexp(arg,axis=-1)
        
    @params_as_tensors
    def conditional_mean(self, F, eval_freq=None):  # pylint: disable=R0201
        """The mean of the likelihood conditioned on latent."""
        eval_freq = self.freq if eval_freq is None else eval_freq
        tec2phase = self.tec_conversion/eval_freq
        phase = F*tec2phase
        return phase

    @params_as_tensors
    def conditional_variance(self, F):
        return tf.fill(tf.shape(F),tf.cast(self.variance,gp.settings.float_type))

    def predict_mean_and_var(self, Fmu, Fvar, **kwargs):
        r"""
        Given a Normal distribution for the latent function,
        return the mean of Y
        if
            q(f) = N(Fmu, Fvar)
        and this object represents
            p(y|f)
        then this method computes the predictive mean
           \int\int y p(y|f)q(f) df dy
        and the predictive variance
           \int\int y^2 p(y|f)q(f) df dy  - [ \int\int y^2 p(y|f)q(f) df dy ]^2
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (e.g. Gaussian) will implement specific cases.
        """
        integrand2 = lambda *X, **kwargs: self.conditional_variance(*X, **kwargs) + tf.square(self.conditional_mean(*X, **kwargs))
        E_y, E_y2 = ndiagquad([self.conditional_mean, integrand2],
                              self.num_gauss_hermite_points,
                              Fmu, Fvar, **kwargs)
        V_y = E_y2 - tf.square(E_y)
        return E_y, V_y

    def predict_density(self, Fmu, Fvar, Y, **kwargs):
        r"""
        Given a Normal distribution for the latent function, and a datum Y,
        compute the log predictive density of Y.
        i.e. if
            q(f) = N(Fmu, Fvar)
        and this object represents
            p(y|f)
        then this method computes the predictive density
            \log \int p(y=Y|f)q(f) df
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        return ndiagquad(self.logp,
                         self.num_gauss_hermite_points,
                         Fmu, Fvar, logspace=True, Y=Y, **kwargs)

    def variational_expectations(self, Fmu, Fvar, Y, **kwargs):
        r"""
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values.
        if
            q(f) = N(Fmu, Fvar)
        and this object represents
            p(y|f)
        then this method computes
           \int (\log p(y|f)) q(f) df.
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        return ndiagquad(self.logp,
                         self.num_gauss_hermite_points,
                         Fmu, Fvar, Y=Y, **kwargs)

class WrappedPhaseGaussianMulti(Likelihood):
    """This is an efficient version of the encoded likelihood."""
    
    def __init__(self, tec_scale=0.001, freqs=None, num_gauss_hermite_points=20, variance=1.0, name=None):
        super().__init__(name=name)
        self.variance = Parameter(
            variance, transform=transforms.positive, dtype=settings.float_type)
        self.tec_scale = tec_scale
        self.num_gauss_hermite_points = num_gauss_hermite_points
        self.freqs = tf.convert_to_tensor(freqs,dtype=settings.float_type,name='freqs') # freqs of data
        self.tec_conversion = tf.convert_to_tensor(tec_scale * -8.4480e9,dtype=settings.float_type,name='tec_conversion') # rad Hz/ tecu
        # Nf
        self.tec2phase = self.tec_conversion / self.freqs
        
    @params_as_tensors
    def logp(self, F, Y, **kwargs):
        """The log-likelihood function."""
        #..., Nf
        phase = F[..., None]*self.tec2phase
        dphase = wrap(phase) - wrap(Y) # Ito theorem
        
        arg = tf.stack([-0.5*tf.square(dphase + 2*np.pi*k)/self.variance - 0.5 * tf.log((2*np.pi) * self.variance) \
                for k in range(-2,3,1)],axis=-1)
        return tf.reduce_logsumexp(arg,axis=-1)
        
    @params_as_tensors
    def conditional_mean(self, F):  # pylint: disable=R0201
        """The mean of the likelihood conditioned on latent."""
        # ..., Nf
        phase = F[..., None]*self.tec2phase
        return phase

    @params_as_tensors
    def conditional_variance(self, F):
        return tf.fill(tf.concat([tf.shape(F),tf.shape(self.freqs)],axis=0),
                tf.cast(self.variance,gp.settings.float_type))

    def predict_mean_and_var(self, Fmu, Fvar, **kwargs):
        r"""
        Given a Normal distribution for the latent function,
        return the mean of Y
        if
            q(f) = N(Fmu, Fvar)
        and this object represents
            p(y|f)
        then this method computes the predictive mean
           \int\int y p(y|f)q(f) df dy
        and the predictive variance
           \int\int y^2 p(y|f)q(f) df dy  - [ \int\int y^2 p(y|f)q(f) df dy ]^2
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (e.g. Gaussian) will implement specific cases.
        """
        integrand2 = lambda *X, **kwargs: self.conditional_variance(*X, **kwargs) + tf.square(self.conditional_mean(*X, **kwargs))
        E_y, E_y2 = ndiagquad([self.conditional_mean, integrand2],
                              self.num_gauss_hermite_points,
                              Fmu, Fvar, **kwargs)
        V_y = E_y2 - tf.square(E_y)
        return E_y, V_y

    def predict_density(self, Fmu, Fvar, Y, **kwargs):
        r"""
        Given a Normal distribution for the latent function, and a datum Y,
        compute the log predictive density of Y.
        i.e. if
            q(f) = N(Fmu, Fvar)
        and this object represents
            p(y|f)
        then this method computes the predictive density
            \log \int p(y=Y|f)q(f) df
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        return ndiagquad(self.logp,
                         self.num_gauss_hermite_points,
                         Fmu, Fvar, logspace=True, Y=Y, **kwargs)

    def variational_expectations(self, Fmu, Fvar, Y, **kwargs):
        r"""
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values.
        if
            q(f) = N(Fmu, Fvar)
        and this object represents
            p(y|f)
        then this method computes
           \int (\log p(y|f)) q(f) df.
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        return ndiagquad(self.logp,
                         self.num_gauss_hermite_points,
                         Fmu, Fvar, Y=Y, **kwargs)


