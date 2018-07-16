import gpflow as gp
import numpy as np
import tensorflow as tf

from gpflow.likelihoods import Gaussian

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


class WrappedPhaseGaussian(Gaussian):
    def __init__(self, tec_scale=0.01, freq=140e6, name=None):
        super().__init__(name=name)
        self.tec_scale = tec_scale
        self.num_gauss_hermite_points = 20
        self.freq = tf.convert_to_tensor(freq,dtype=settings.float_type,name='test_freq') # frequency the phase is calculated at for the predictive distribution
        self.tec_conversion = tf.convert_to_tensor(tec_scale * -8.4480e9,dtype=settings.float_type,name='tec_conversion') # rad Hz/ tecu
        self.tec2phase = tf.convert_to_tensor(self.tec_conversion / self.freq,dtype=settings.float_type,name='tec2phase')
        
    @params_as_tensors
    def logp(self, F, Y):
        """The log-likelihood function."""
        freqs = Y[:,-2:-1]
        Y = Y[:,:-1]
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


