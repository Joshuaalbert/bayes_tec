import tensorflow as tf
import numpy as np
import pylab as plt
plt.style.use('ggplot')

from gpflow import transforms
from gpflow import settings

from gpflow.params import Parameter, Parameterized, ParamList
from gpflow.decors import params_as_tensors, autoflow
from gpflow.kernels import Kernel

from bayes_tec.utils.stat_utils import log_normal_solve
from gpflow.priors import LogNormal

from .logging import logging

class SpectralMixture(Kernel):
    """
    SM(t) = sum_q w_q exp[-2 pi**2 |t**2/v_q| ] prod_p cos[2 pi |t(p)/mu_q(p)|]
    """
    def __init__(self, dims, w, v, mu, active_dims=None, name=None):
        super().__init__(dims, active_dims, name=name)
        self.Q, self.P = v.shape
        self.w = Parameter(w,#Q
                dtype=settings.float_type,
                trainable=True,
                transform = transforms.positive)
        self.v = Parameter(v,#Q,P
                dtype=settings.float_type,
                trainable=True,
                transform=transforms.positive)#inverse lengthscale**2
        self.mu = Parameter(mu,#Q,P
                dtype=settings.float_type,
                trainable=True)# inverse periods (symmetry means make it positive

    @params_as_tensors
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        se_scale = (2*np.pi)*self.v#Q, P
        se_X  = se_scale[:, None, :] * X[None, :, :]#Q, N, P
        cos_scale = (2*np.pi)*self.mu#Q, P
        cos_X  = tf.tensordot(cos_scale, X, axes=[[1],[1]])#Q, N
        se_X2 = None
        cos_X2 = cos_X
        if X2 is not None:
            se_X2 = se_scale[:, None, :] * X2[None, :, :]#Q, M, P
            cos_X2 = tf.tensordot(cos_scale, X2, axes=[[1],[1]])#Q, M

        r2 = self.scaled_square_dist_batched(se_X,se_X2)#Q, N, M
        # tau = X_NP * mu_QP - X2_MP * mu_QP
        #Q N M
        tau = cos_X[:, :, None] -  cos_X2[:,None, :]
        k =  tf.reduce_sum(self.w[:, None, None] * tf.exp(-0.5*r2) * tf.cos(tau), axis=0)
        return k

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.reduce_sum(self.w))

    @params_as_tensors    
    def scaled_square_dist_batched(self,X, X2):
        """
        X: tensor B, N, D
        X2: tensor B, M, D (or 1, M, D) and will be broadcast to B, M ,D
        Return:
        tensor B, N, M
        """
        # Clipping around the (single) float precision which is ~1e-45.
#        X = X / lengthscales
        Xs = tf.reduce_sum(tf.square(X), axis=2)#B,N

        if X2 is None:
            dist = -2.*tf.matmul(X,X,transpose_b=True)
            dist += Xs[:,:,None] + Xs[:,None,:]
            return tf.maximum(dist, 1e-40)

#        X2 = X2 / lengthscales# B (1), M, D
        X2s = tf.reduce_sum(tf.square(X2), axis=2)# B (1), M 
        dist = -2 * tf.matmul(X, X2, transpose_b=True)
        dist += Xs[:,:,None] + X2s[:,None,:]
        return tf.maximum(dist, 1e-40)

    def plot_spectrum(self, save_fig=None, show=True, dim = 0,  yscale='log'):
        max_freq = np.max(self.mu.value[:,dim]) + 10.*np.max(np.sqrt(self.v.value[:,dim]))
        freqs = np.linspace(0., max_freq, 1000)
        S = np.zeros(1000)

        for (w,v,mu) in zip(self.w.value, self.v.value[:,dim], self.mu.value[:,dim]): 
            S += w*np.exp(-(freqs - mu)**2/2./v) / np.sqrt(2*np.pi*v)

        plt.plot(freqs, S)
        plt.xlabel('freq [1/sample]')
        plt.ylabel('spectrum [1]')
        plt.yscale(yscale)
        if save_fig is not None:
            plt.savefig(save_fig)
            plt.close('all')
        else:
            plt.show()


class ThinLayer(Kernel):
    """
    Thin layer kernel see paper.
    input dims are (east, north, up, cos zenith_angle)
    """
    def __init__(self, x0, a=400.,b=20.,l=10., tec_scale=1e-3,
                 active_dims=None, name=None):
        super().__init__(4, active_dims, name=name)
        self.tec_scale = tec_scale
        # b**2 exp(2 g) sec1 sec2 K(f(x),f(x'))

        self.x0 = Parameter(x0, 
                dtype=settings.float_type,
                trainable=False)


#        g_prior = log_normal_solve(1.,np.log(100.))
#        self.expg = Parameter(1.,
#                transforms.Exp(),
#                dtype=settings.float_type, 
#                prior=LogNormal(g_prior[0],g_prior[1]**2),
#                name='thinlayer_expg')#per 10^10

        kern_sigma = 0.005/tec_scale
        v_prior = log_normal_solve(kern_sigma**2,0.1*kern_sigma**2)

#        v_prior = log_normal_solve(,0.5) 
        self.variance = Parameter(np.exp(v_prior[0]), 
                transform=transforms.positive,
                dtype=settings.float_type, 
                prior=LogNormal(v_prior[0],v_prior[1]**2),
                name='thinlayer_var')

        l_prior = log_normal_solve(10.,20.)
        self.lengthscales = Parameter(l, 
                transform=transforms.Rescale(10.)(transforms.positive),
                dtype=settings.float_type,
                prior=LogNormal(l_prior[0],l_prior[1]**2), 
                name='thinlayer_l')

        a_scale = 400. # 300 km scale 
        a_prior = log_normal_solve(400., 200.)
        self.a = Parameter(a, 
                transform=transforms.Rescale(a_scale)(transforms.positive),
                dtype=settings.float_type,
                prior=LogNormal(a_prior[0],a_prior[1]**2),
                name='thinlayer_a')
        
#        b_scale = 20 # 10km scale
#        b_prior = log_normal_solve(20,20)
#        self.b = Parameter(b, 
#                transform=transforms.Rescale(b_scale)(transforms.positive),
#                dtype=settings.float_type,
#                prior=LogNormal(b_prior[0],b_prior[1]**2),
#                name='thinlayer_b')

    @params_as_tensors
    def f(self,X):
        kz = X[:,0]
        xi = X[:,1:4]
        #N
        h = (self.a + (self.x0[2] - xi[:,2])) * tf.sqrt(1. - tf.square(kz)) * (3. / (1. + 2.*kz))
        #N,2
        x_ = xi[:,:2] - h[:,None]
        return x_

    def _K_M52(self,r2):
        
        return tf.exp(-0.5 * r2)
        r = tf.sqrt(r2)
        return (1.0 + np.sqrt(5.) * r + (5. / 3.) * r2) * tf.exp(-np.sqrt(5.) * r)

        
    @params_as_tensors
    def _K(self, X, X2=None):
        X = self.f(X)
        if X2 is not None:
            X2 = self.f(X2)
        r2 = tf.maximum(self.scaled_square_dist(X, X2), 1e-40)
        return self.variance * self._K_M52(r2)

    @params_as_tensors
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        
        if X2 is None:

            kz = X[:,0]
            X0 = tf.tile(self.x0[None,:],
                        tf.concat([tf.shape(X)[0:1], tf.constant([1])],axis=0))
            X0 = tf.concat([kz[:,None],X0],axis=1)
            Ksym = self._K(X,X0)
            Ksym = Ksym + tf.transpose(Ksym)

            # smoothed at 60 deg
            sec = 1./kz
#            return (1e-6/self.tec_scale**2)*(self.b * self.expg)**2 *
            return (self._K(X,X2) + self._K(X0,X0) - Ksym) * sec[:,None]*sec[None,:]

        else:
            kz = X[:,0]
            kz2 = X2[:,0]
            X0i = tf.tile(self.x0[None,:],
                        tf.concat([tf.shape(X)[0:1], tf.constant([1])],axis=0))
            X0i = tf.concat([kz[:,None],X0i],axis=1)

            X0j = tf.tile(self.x0[None,:],
                        tf.concat([tf.shape(X2)[0:1], tf.constant([1])],axis=0))
            X0j = tf.concat([kz2[:,None],X0j],axis=1)

            Ksym = self._K(X,X0j) + self._K(X0i,X2)

            # smoothed at 60 deg
            sec = 1. / kz
            sec2 = 1. / kz2
            return (self._K(X,X2) + self._K(X0i,X0j) - Ksym) * sec[:,None] *sec2[None,:]
#            return (1e-6/self.tec_scale**2)*(self.b * self.expg)**2 *(self._K(X,X2) + self._K(X0i,X0j) - Ksym) * sec[:,None] *sec2[None,:]

    @params_as_tensors
    def scaled_square_dist(self, X, X2):
        """
        Returns ((X - X2?)/lengthscales)².
        Due to the implementation and floating-point imprecision, the
        result may actually be very slightly negative for entries very
        close to each other.
        """
        X = X / self.lengthscales
        Xs = tf.reduce_sum(tf.square(X), axis=1)

        if X2 is None:
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += tf.reshape(Xs, (-1, 1))  + tf.reshape(Xs, (1, -1))
            return dist

        X2 = X2 / self.lengthscales
        X2s = tf.reduce_sum(tf.square(X2), axis=1)
        dist = -2 * tf.matmul(X, X2, transpose_b=True)
        dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))
        return tf.maximum(dist, 1e-40)

    def scaled_euclid_dist(self, X, X2):
        """
        Returns |(X - X2?)/lengthscales| (L2-norm).
        """
        r2 = self.scaled_square_dist(X, X2)
        # Clipping around the (single) float precision which is ~1e-45.
        return tf.sqrt(r2)

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        if not presliced:
            X, _ = self._slice(X, None)

        kz = X[:,0]
        sec = 1./kz
        
        tanphi = tf.sqrt(1. - tf.square(kz)) * sec
        xi = X[:,1:4]
        dx = xi - self.x0[None,:]
        dx = dx[:,0:2] + dx[:,2:3]*tanphi[:,None]
        dx = dx / self.lengthscales
        r2 = tf.maximum(tf.reduce_sum(tf.square(dx),axis=1),1e-40)
        Ksym = self._K_M52(r2)
        
        return tf.maximum(2.*tf.square(sec) * tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance) )*(1. - Ksym), 1e-40)
#        return 2.*(1e-6/self.tec_scale**2) * tf.square(self.b * self.expg* sec) * tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance) )*(1. - Ksym)
