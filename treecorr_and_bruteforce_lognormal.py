from tqdm import tqdm,trange

from my_functions import *


n_eval = 1000 #number of fields for brute-force computation
n_procs = 250 #number of processes for parallel computation
alpha = 1. #degree of non-gaussianity
n_pix = 4096 #number of pixels in the map

"""
WARNING! Running this takes a long time. With these settings (on a server that actually has 250 cores), this will take ~5 days.
"""

print("Computing gamma 3pcf with treecorr")
for i in trange(64):
	kkk = compute_gamma_3pcf_of_lognormal_random_field(power_spectrum,alpha,n_pix=n_pix)
	kkk.write('results/gamma_lognormal_'+str(i)+'.dat')

print("Computing kappa 3pcf with treecorr")
for i in trange(64):
	kkk = compute_kappa_3pcf_of_lognormal_random_field(power_spectrum,alpha,n_pix=n_pix)
	kkk.write('results/kappa_lognormal_'+str(i)+'.dat')

u_and_v_indices = tqdm([[9,11],[9,21],[0,11],[4,16],[4,17]])
print("Computing kappa 3pcf bruteforce with ",n_eval,"iterations.")
for u_ind,v_ind in u_and_v_indices:
    u_and_v_indices.set_description("Processing u:"+str(u_ind)+", v:"+str(v_ind))
    compute_triangles_bruteforce(n_eval,n_procs,u_ind,v_ind,alpha)

print("Computing gamma 3pcf bruteforce with ",n_eval,"iterations.")
for u_ind,v_ind in u_and_v_indices:
    u_and_v_indices.set_description("Processing u:"+str(u_ind)+", v:"+str(v_ind))
    compute_triangles_bruteforce(n_eval,n_procs,u_ind,v_ind,alpha,gamma=True)

print("Computing gamma 3pcf of gaussian random field")
for i in range(32):
    print("Starting ",i)
    ggg = compute_gamma_3pcf_of_gaussian_random_field(power_spectrum,sigma=1,n_pix=n_pix,fraction_of_pixels=0.1)
    ggg.write('results/gamma_lognormal_'+str(i)+'.dat')
