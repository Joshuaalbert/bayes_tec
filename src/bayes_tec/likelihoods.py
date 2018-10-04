import gpflow as gp
import numpy as np
import tensorflow as tf
from gpflow import params_as_tensors
from gpflow import transforms
from gpflow.params import Parameter
from gpflow.likelihoods import Likelihood
from gpflow import settings
from gpflow.quadrature import ndiagquad, ndiag_mc, mvnquad
from gpflow import logdensities

float_type = settings.float_type

try:
    @tf.RegisterGradient('WrapGrad')
    def _wrap_grad(op,grad):
        phi = op.inputs[0]
        return tf.ones_like(phi)*grad

    def wrap(phi):
        out = tf.atan2(tf.sin(phi),tf.cos(phi))
        with tf.get_default_graph().gradient_override_map({'Identity': 'WrapGrad'}):
            return tf.identity(out)

except:
    pass#already defined



class WrappedPhaseGaussianEncoded(Likelihood):
    def __init__(self, tec_scale=0.01, num_gauss_hermite_points=20, variance=1.0, name=None):
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
        self.Nf = len(freqs)
        self.freqs = tf.convert_to_tensor(freqs,dtype=settings.float_type,name='freqs') # freqs of data
        self.tec_conversion = tf.convert_to_tensor(tec_scale * -8.4480e9,dtype=settings.float_type,name='tec_conversion') # rad Hz/ tecu
        # Nf
        self.tec2phase = self.tec_conversion / self.freqs
        
    @params_as_tensors
    def logp(self, F, **kwargs):
        """The log-likelihood function."""
        #..., Nf
        Y = tf.stack([kwargs["Y_{}".format(i)] for i in range(self.Nf)],axis=2)
        
        phase = F[..., None]*self.tec2phase
        dphase = wrap(phase) - wrap(Y) # Ito theorem
        
        arg = tf.stack([-0.5*(tf.square(dphase + 2*np.pi*k)/self.variance + tf.cast(tf.log(2*np.pi), settings.float_type) + tf.log(self.variance)) \
                for k in range(-2,3,1)],axis=0)
        if kwargs.get("W_0") is not None:
            W = tf.stack([kwargs["W_{}".format(i)] for i in range(self.Nf)],axis=2)
            return tf.reduce_mean(W*tf.reduce_logsumexp(arg,axis=0), axis=-1)
        else:
            return tf.reduce_mean(tf.reduce_logsumexp(arg,axis=0),axis=-1)

        
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
        Y_burst = {"Y_{}".format(i): Y[:,:,i] for i in range(self.Nf)}

        return ndiagquad(self.logp,
                         self.num_gauss_hermite_points,
                         Fmu, Fvar, logspace=True, **Y_burst, **kwargs)

    def variational_expectations(self, Fmu, Fvar, Y, weights, **kwargs):
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
        Y_burst = {"Y_{}".format(i): Y[:,:,i] for i in range(self.Nf)}
        weights_burst = {"W_{}".format(i): weights[:,:,i] for i in range(self.Nf)}
        return ndiagquad(self.logp,
                         self.num_gauss_hermite_points,
                         Fmu, Fvar, **Y_burst, **weights_burst, **kwargs)


class WrappedPhaseGaussianEncodedHetero(Likelihood):
    """This is an efficient version of the encoded likelihood."""
    
    def __init__(self, tec_scale=0.005, num_gauss_hermite_points=20, num_mc_samples=1, variance=1.0, K=2, name=None):
        super().__init__(name=name)
        self.K = K
        self.variance = Parameter(
            variance, transform=transforms.positive, dtype=settings.float_type)
        self.tec_scale = tec_scale
        self.num_gauss_hermite_points = num_gauss_hermite_points
        self.num_mc_samples = num_mc_samples
        self.tec_conversion = tf.convert_to_tensor(tec_scale * -8.4480e9,dtype=settings.float_type, name='tec_conversion') # rad Hz/ tecu
        
    @params_as_tensors
    def logp(self, F, Y, Y_var, freq, **kwargs):
        """
        The log-likelihood function.
        F is ..., P
        Y is ..., P
        Y_var ..., P
        freq ..., P
        Returns:
        tensor ..., P
        """
        #..., Nf       
        phase = self.tec_conversion * (F / freq)
#        dphase = wrap(phase) - wrap(Y) # Ito theorem

        log_prob = tf.stack([tf.distributions.Normal(phase + tf.convert_to_tensor(k*2*np.pi,float_type), 
                                          tf.sqrt(Y_var)).log_prob(wrap(Y)) for k in range(-self.K,self.K+1,1)], axis=0)
        log_prob = tf.reduce_logsumexp(log_prob, axis=0) #..., P

        return log_prob

        
    @params_as_tensors
    def conditional_mean(self, F, freq, **kwargs):  # pylint: disable=R0201
        """The mean of the likelihood conditioned on latent."""
        # ..., Nf
        phase = self.tec_conversion * (F/freq)
        return phase

    @params_as_tensors
    def conditional_variance(self, Y_var, **kwargs):
        return Y_var + self.variance

    def predict_mean_and_var(self, Fmu, Fvar, Y_var, freq):
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
                              Fmu, Fvar, Y_var=Y_var, freq=freq)
        V_y = E_y2 - tf.square(E_y)
        return E_y, V_y

    def predict_density(self, Fmu, Fvar, Y, Y_var, freq):
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
                         Fmu, Fvar, logspace=True, Y=Y, Y_var=Y_var, freq=freq)

    def variational_expectations(self, Fmu, Fvar, Y, Y_var, freq, mc=False, mvn=False):
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
        if mvn:
            assert len(Fvar.shape) == 3

        if not mvn:
            if not mc:
                return ndiagquad(self.logp,
                                 self.num_gauss_hermite_points,
                                 Fmu, Fvar, Y=Y, Y_var=Y_var, freq=freq)
            else:
                return ndiag_mc(self.logp, self.num_mc_samples , Fmu, Fvar, Y=Y, Y_var=Y_var, freq=freq)
        else:
            if not mc:
                raise ValueError("Too slow to do this")
            else:
                return mvn_mc(self.logp, self.num_mc_samples , Fmu, Fvar, Y=Y, Y_var=Y_var, freq=freq)


class GaussianTecHetero(Likelihood):
    def __init__(self, tec_scale=0.005,  **kwargs):
        super().__init__(**kwargs)
        self.tec_scale = tf.convert_to_tensor(tec_scale, dtype=float_type)

    @params_as_tensors
    def logp(self, F, Y, Y_var):
        tec = F*self.tec_scale
        return logdensities.gaussian(Y, tec, Y_var)

    @params_as_tensors
    def predict_mean_and_var(self, Fmu, Fvar, Y_var):
        return tf.identity(Fmu)*self.tec_scale, Fvar*self.tec_scale**2 + Y_var

    @params_as_tensors
    def predict_density(self, Fmu, Fvar, Y, Y_var):
        return logdensities.gaussian(Y, Fmu*self.tec_scale, Fvar*self.tec_scale**2 + Y_var)

    @params_as_tensors
    def variational_expectations(self, Fmu, Fvar, Y, Y_var):
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(Y_var) \
               - 0.5 * (tf.square(Y - Fmu*self.tec_scale) + Fvar*self.tec_scale**2) / Y_var

class WrappedPhaseGaussianEncodedHeteroDirectionalOutliers(Likelihood):
    """This is an efficient version of the encoded likelihood."""
    
    def __init__(self,  tec_scale=0.005, num_gauss_hermite_points=20, num_mc_samples=1, variance=1.0, K=2, directional_var_matrix=None, name=None):
        super().__init__(name=name)
        self.K = K
        self.variance = Parameter(
            variance, transform=transforms.positive, dtype=settings.float_type)
        assert directionla_var_matrix is not None
        self.directional_var_matrix = Parameter(directional_var_matrix, transform=transforms.positive, dtype=settings.float_type)
        self.tec_scale = tec_scale
        self.num_gauss_hermite_points = num_gauss_hermite_points
        self.num_mc_samples = num_mc_samples
        self.tec_conversion = tf.convert_to_tensor(tec_scale * -8.4480e9,dtype=settings.float_type, name='tec_conversion') # rad Hz/ tecu
        
    @params_as_tensors
    def logp(self, F, Y, Y_var, freq, dir_idx, **kwargs):
        """
        The log-likelihood function.
        F is ..., P
        Y is ..., P
        Y_var ..., P
        freq ..., P
        Returns:
        tensor ..., P
        """
        #..., Nf       
        phase = wrap(self.tec_conversion * (F / freq))
#        dphase = wrap(phase) - wrap(Y) # Ito theorem

        dir_var = tf.gather(self.directional_var_matrix, dir_idx)

        log_prob = tf.stack([tf.distributions.Normal(phase + tf.convert_to_tensor(k*2*np.pi,float_type), 
                                          tf.sqrt(Y_var)).log_prob(wrap(Y)) for k in range(-self.K,self.K+1,1)], axis=0)
        log_prob = tf.reduce_logsumexp(log_prob, axis=0) #..., P

        return log_prob

        
    @params_as_tensors
    def conditional_mean(self, F, freq, **kwargs):  # pylint: disable=R0201
        """The mean of the likelihood conditioned on latent."""
        # ..., Nf
        phase = self.tec_conversion * (F/freq)
        return phase

    @params_as_tensors
    def conditional_variance(self, Y_var, **kwargs):
        return Y_var + self.variance

    def predict_mean_and_var(self, Fmu, Fvar, Y_var, freq):
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
                              Fmu, Fvar, Y_var=Y_var, freq=freq)
        V_y = E_y2 - tf.square(E_y)
        return E_y, V_y

    def predict_density(self, Fmu, Fvar, Y, Y_var, freq):
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
                         Fmu, Fvar, logspace=True, Y=Y, Y_var=Y_var, freq=freq)

    def variational_expectations(self, Fmu, Fvar, Y, Y_var, freq, mc=False, mvn=False):
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
        if mvn:
            assert len(Fvar.shape) == 3

        if not mvn:
            if not mc:
                return ndiagquad(self.logp,
                                 self.num_gauss_hermite_points,
                                 Fmu, Fvar, Y=Y, Y_var=Y_var, freq=freq)
            else:
                return ndiag_mc(self.logp, self.num_mc_samples , Fmu, Fvar, Y=Y, Y_var=Y_var, freq=freq)
        else:
            if not mc:
                raise ValueError("Too slow to do this")
            else:
                return mvn_mc(self.logp, self.num_mc_samples , Fmu, Fvar, Y=Y, Y_var=Y_var, freq=freq)


