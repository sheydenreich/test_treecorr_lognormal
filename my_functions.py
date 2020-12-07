from numpy import sin,cos,exp,pi,sinh,tanh,cosh,arctan,sqrt,arccos,tan,arctan2
from scipy.special import jv,jn_zeros
import numpy as np
from FyeldGenerator import generate_field
import treecorr
import multiprocessing.managers
from multiprocessing import Process, Pool
from functools import partial

class besselintegrator:
    """Input Parameters: n_dim_bessel: Order of Bessel function
    prec_h: step width
    prec_k: max. root of bessel function being considered in the integral
    written after Ogata et al. (2005)"""
    def psi(self,t):
        return t*tanh(pi*sinh(t)/2)
    def psip(self,t):
        zahler = sinh(pi*sinh(t))+pi*t*cosh(t)
        nenner = cosh(pi*sinh(t))+1
        return zahler/nenner
    
    def __init__(self,n_dim_bessel,prec_h,prec_k):
        self.n_dim_bessel = n_dim_bessel
        self.prec_h = prec_h
        self.prec_k = int(prec_k/prec_h)
        self.bessel_zeros = jn_zeros(n_dim_bessel,self.prec_k)
        self.pi_bessel_zeros = self.bessel_zeros/pi
        self.psiarr = pi*self.psi(self.pi_bessel_zeros*self.prec_h)/self.prec_h
        self.besselarr = jv(self.n_dim_bessel,self.psiarr)
        self.psiparr = self.psip(self.prec_h*self.pi_bessel_zeros)
        self.warr = 2/(pi*self.bessel_zeros*jv(n_dim_bessel+1,self.bessel_zeros)**2)

    def integrate(self,function,R):
        """Computes the Integral int_0^infty f(k)J(kR) dk"""
        return pi/R*np.sum(self.warr*function(self.psiarr/R)*self.besselarr*self.psiparr)


def distrib(shape):
    power_field = np.random.normal(0,1,shape)+1.0j*np.random.normal(0,1,shape)
    return power_field

def create_gaussian_random_field(power_spectrum, n_pix=4096,sigma=1.,pixsize=None,return_scale=False):
    """creates gaussian random field from given power spectrum, with mean 0 and variance sigma"""
    shape = (n_pix,n_pix)
    if pixsize is None:
        pixsize = 1./n_pix
    field = generate_field(distrib,power_spectrum,shape,unit_length = pixsize)
    fieldscale = np.sqrt(np.var(field))
    if(return_scale):
        return sigma*field/fieldscale,fieldscale
    else:
        return sigma*field/fieldscale


def power_spectrum(ell,ell_0=100):
    func = 1/(ell+ell_0)**2
    return func

def power_spectrum_ell(ell,ell_0=100):
    return ell*power_spectrum(ell,ell_0)


bi = besselintegrator(0,0.001,6.)
def correlation_function(x,n_pix,mod=1.):
    x = x*2*np.pi/n_pix
    factor = (2*np.pi)**2/np.pi/n_pix**4/mod**2
    if(hasattr(x,'__len__')):
        lx = len(x)
        result = np.zeros(lx)
        for i,xi in enumerate(x):
            result[i] = bi.integrate(power_spectrum_ell,xi)
        return factor*result
    else:
        return factor*bi.integrate(power_spectrum_ell,x)
    
def correlation_function_lognormal(x,alpha,n_pix,sigma=1,mod=1):
    c = np.exp(alpha**2/2)
    A = sigma/np.sqrt(c**2-1)
    return A**2*(np.exp(alpha**2*correlation_function(x,n_pix,mod=mod))-1)

def kappa_3pcf_lognormal(x12,x13,x23,alpha,n_pix,sigma=1,mod=1):
    c = np.exp(alpha**2/2)
    A = sigma/np.sqrt(c**2-1)
    xi12 = correlation_function_lognormal(x12,alpha,n_pix,sigma=sigma,mod=mod)
    xi13 = correlation_function_lognormal(x13,alpha,n_pix,sigma=sigma,mod=mod)
    xi23 = correlation_function_lognormal(x23,alpha,n_pix,sigma=sigma,mod=mod)
    term1 = xi12*xi13*xi23
    term2 = xi12*xi13+xi12*xi23+xi13*xi23
    return term1/A**3+term2/A

def extract_power_spectrum(field,n_bins,max_pix,n_pix):
    fourier_image = np.fft.fftn(field)
    fourier_amplitudes = np.abs(fourier_image)**2
    kfreq = np.fft.fftfreq(n_pix) * n_pix
    kfreq2D = np.meshgrid(kfreq, kfreq)
      
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    kbins = np.linspace(0, max_pix, n_bins)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                     statistic = "mean",
                                     bins = kbins)
#     Abins *= 4. * np.pi / 3. * (kbins[1:]**3 - kbins[:-1]**3)
    return Abins/2,kvals


def create_lognormal_random_field(power_spectrum_of_gaussian_random_field,alpha,sigma=1,npix=4096,pixsize=None):
    """Returns a lognormal field with non-gaussianity alpha"""
    c = np.exp(alpha**2/2)
    new_field_prefactor = sigma/(c*np.sqrt(c**2-1))
    gaussian_random_field = create_gaussian_random_field(power_spectrum_of_gaussian_random_field,npix,sigma)
    new_field = new_field_prefactor*(np.exp(alpha*gaussian_random_field)-c)
    return new_field

def progressBar(name, value, endvalue, bar_length = 25, width = 20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent*bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\r{0: <{1}} : [{2}]{3}%".format(name, width, arrow + spaces, int(round(percent*100))))
    sys.stdout.flush()
    if value == endvalue:        
         sys.stdout.write('\n\n')

def compute_2pcf_of_gaussian_random_field(power_spectrum, n_pix=4096, pixsize=None):
    field = create_gaussian_random_field(power_spectrum,n_pix,pixsize=pixsize)
    idx,idy = np.indices(field.shape)
    field = field.reshape(n_pix**2)
    idx = idx.reshape(n_pix**2)
    idy = idy.reshape(n_pix**2)
    cat = treecorr.Catalog(x=idx,y=idy,k=field)
    kk = treecorr.KKCorrelation(nbins=20,min_sep = 1, max_sep = n_pix/3)
    kk.process(cat)
    return kk

def compute_2pcf_of_lognormal_random_field(power_spectrum_of_gaussian_random_field, alpha, sigma=1, n_pix=4096,pixsize=None):
    field = create_lognormal_random_field(power_spectrum_of_gaussian_random_field,alpha,sigma,n_pix,pixsize)
    idx,idy = np.indices(field.shape)
    field = field.reshape(n_pix**2)
    idx = idx.reshape(n_pix**2)
    idy = idy.reshape(n_pix**2)
    cat = treecorr.Catalog(x=idx,y=idy,k=field)
    kk = treecorr.KKCorrelation(nbins=10,min_sep = 1, max_sep = n_pix/3)
    kk.process(cat)
    return kk

def compute_kappa_3pcf_of_lognormal_random_field(power_spectrum_of_gaussian_random_field, alpha, sigma=1, n_pix=4096,fraction_of_pixels = 0.1):
    field = create_lognormal_random_field(power_spectrum_of_gaussian_random_field,alpha,sigma,n_pix)
    idx,idy = np.indices(field.shape)
    field = field.reshape(n_pix**2)
    idx = idx.reshape(n_pix**2)
    idy = idy.reshape(n_pix**2)
    #sort out fraction of pixels
    rands = np.random.uniform(size=n_pix**2)
    mask = (rands<fraction_of_pixels)
    idx = idx[mask]
    idy = idy[mask]
    field = field[mask]
    #create catalogue
    cat = treecorr.Catalog(x=idx,y=idy,k=field)
    kkk = treecorr.KKKCorrelation(nbins=10,min_sep = 0.99, max_sep = n_pix/3,nubins=10,min_u=0,nvbins=11,min_v=0,max_v=1,num_threads=250)
    kkk.process(cat)
    return kkk

def compute_gamma_3pcf_of_lognormal_random_field(power_spectrum_of_gaussian_random_field, alpha, sigma=1, n_pix=4096,fraction_of_pixels = 0.1):
    field = create_lognormal_random_field(power_spectrum_of_gaussian_random_field,alpha,sigma,n_pix)
    idx,idy = np.indices(field.shape)
    gamma_field = create_gamma_field(field)
    idx = idx.reshape(n_pix**2)
    idy = idy.reshape(n_pix**2)
    gamma_field = gamma_field.reshape(n_pix**2)
    #sort out fraction of pixels
    rands = np.random.uniform(size=n_pix**2)
    mask = (rands<fraction_of_pixels)
    idx = idx[mask]
    idy = idy[mask]
    gamma_field = gamma_field[mask]
    cat = treecorr.Catalog(x=idx,y=idy,g1=gamma_field.real,g2=gamma_field.imag)
    ggg = treecorr.GGGCorrelation(nbins=10,min_sep = 0.99, max_sep = n_pix/3,nubins=10,min_u=0,nvbins=11,min_v=0,max_v=1,num_threads=250)
    ggg.process(cat)
    return ggg

def compute_gamma_3pcf_of_gaussian_random_field(power_spectrum,sigma=1,n_pix=4096,fraction_of_pixels=0.1):
    field = create_gaussian_random_field(power_spectrum,n_pix,sigma)
    idx,idy = np.indices(field.shape)
    gamma_field = create_gamma_field(field)
    idx = idx.reshape(n_pix**2)
    idy = idy.reshape(n_pix**2)
    gamma_field = gamma_field.reshape(n_pix**2)

    rands = np.random.uniform(size=n_pix**2)
    mask = (rands<fraction_of_pixels)

    idx = idx[mask]
    idy = idy[mask]
    gamma_field = gamma_field[mask]
    cat = treecorr.Catalog(x=idx,y=idy,g1=gamma_field.real,g2=gamma_field.imag)
    ggg = treecorr.GGGCorrelation(nbins=10,min_sep = 0.99, max_sep = n_pix/3,nubins=10,min_u=0,nvbins=11,min_v=0,max_v=1,num_threads=250)
    ggg.process(cat)
    return ggg

def Dhat_func(npix = 4096,pixsize = 1.):
    xs1,xs2 = np.indices((npix,npix))
    xs1 = (xs1 - npix/2)*pixsize
    xs2 = (xs2 - npix/2)*pixsize
    with np.errstate(divide="ignore",invalid="ignore"):
        a = (xs1**2-xs2**2+2.j*xs1*xs2)/(xs1**2+xs2**2)
    a[(xs1**2+xs2**2==0)] = 0
    return np.pi*a

def create_gamma_field(kappa_field,Dhat=None):
    if Dhat is None:
        Dhat = Dhat_func()
    fieldhat = np.fft.fftshift(np.fft.fft2(kappa_field))
    gammahat = fieldhat*Dhat
    gamma = np.fft.ifft2(np.fft.ifftshift(gammahat))
    return gamma

def compute_bruteforce_3pcf(field,r1,r2,r3):
    x1 = np.array([0,0])
    x2 = np.array([0,int(np.round(r1))])
    y = (r2**2+r1**2-r3**2)/(2*r1)
    x = np.sqrt(r2**2-y**2)
    x = int(np.round(x))
    y = int(np.round(y))
    x3 = [x,y]
    fieldshape = field.shape
    result = 0
    for i in range(fieldshape[0]-x):
        for j in range(fieldshape[1]-int(np.round(r1))):
            result += field[i,j]*field[i,j+x2[1]]*field[i+x3[0],j+x3[1]]
    result /= ((fieldshape[0]-x)*(fieldshape[1]-int(np.round(r1))))
    return result


class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)

def call_bruteforce_computation(result_array,u_array,v_array,r_array,gamma,alpha,i):
    np.random.seed((243502345*i**3+i)%2**32)
    field = create_lognormal_random_field(power_spectrum,alpha,npix=4096)
    if(gamma):
        gamma_field = create_gamma_field(field)
        field = gamma_field
    for j,r2 in enumerate(r_array):
        r3 = u_array[j]*r2
        r1 = v_array[j]*r3+r2
        result_array[i,j] = compute_bruteforce_3pcf(field,r1,r2,r3)

def compute_triangles_bruteforce(n_eval,n_procs,u_ind,v_ind,alpha,gamma=False,n_pix=4096):
    if(gamma):
        ggg = treecorr.GGGCorrelation(nbins=10,min_sep = 0.99, max_sep = n_pix/3,nubins=10,min_u=0,nvbins=11,min_v=0,max_v=1,num_threads=n_procs)
        ggg.read('results/gamma_lognormal_0.dat')
        r_array = ggg.meand1[:,u_ind,v_ind]
        u_array = ggg.meanu[:,u_ind,v_ind]
        v_array = ggg.meanu[:,u_ind,v_ind]
    else:
        kkk = treecorr.KKKCorrelation(nbins=10,min_sep = 0.99, max_sep = n_pix/3,nubins=10,min_u=0,nvbins=11,min_v=0,max_v=1,num_threads=n_procs)
        kkk.read('results/kappa_lognormal_0.dat')
        r_array = kkk.meand1[:,u_ind,v_ind]
        u_array = kkk.meanu[:,u_ind,v_ind]
        v_array = kkk.meanu[:,u_ind,v_ind]

    m = MyManager()
    m.start()
    if(gamma):
        results = m.np_zeros((n_eval,len(r_array)),dtype=complex)
    else:
        results = m.np_zeros((n_eval,len(r_array)))
    pool = Pool(n_procs)
    run_list = range(n_eval)
    func = partial(call_bruteforce_computation,results,u_array,v_array,r_array,gamma,alpha)
    list_of_results = pool.map(func, run_list)
    if(gamma):
        interm_string = "gamma"
    else:
        interm_string = "kappa"

    np.save("results/bruteforce_"+interm_string+"_u_ind_"+str(u_ind)+"_v_ind_"+str(v_ind),results)