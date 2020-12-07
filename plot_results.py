from my_functions import *
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon

import os

def create_triangle(r1,r2,r3,offset = [0,0],yscale = 1.):
    """
    overwrite the my_functions definition that creates integer-valued triangles
    """
    x1 = np.array([0,0])
    x2 = np.array([r1,0])
    y = (r2**2+r1**2-r3**2)/(2*r1)
    x = np.sqrt(r2**2-y**2)
    x3 = np.array([y,x])
    offset = np.array(offset)
    x1 = x1 + offset
    x2 = x2 + offset
    x3 = x3 + offset
    result = np.array([x1,x2,x3])
    result[:,1]*=yscale
    return result

def plot_triangle(ax,u,v,xpos=0.6,ypos=0.6,scale=0.3):
    ax2 = ax.twiny()
    ax2.set_xlim(0,1)
    ax2.axis('off')
    ylims = ax2.get_ylim()
    dylims = ylims[1]-ylims[0]
    ax2.set_xlim(ylims)
        
    r2 = 1
    r3 = u*r2
    r1 = v*r3+r2
    
    r2 = scale*r2/r1*dylims
    r3 = scale*r3/r1*dylims
    r1 = scale*dylims
    tri = create_triangle(r1,r2,r3,offset = [xpos*ylims[1]+(1-xpos)*ylims[0],ypos*ylims[1]+(1-ypos)*ylims[0]])
    tri2 = np.array([tri[0],tri[2]])
    p = Polygon(tri,ec='black',fill=False,closed=False)
    p2 = Polygon(tri2,ec='red',fill=False)
    ax2.add_patch(p)
    ax2.add_patch(p2)

"""
comparing kkk-correlation
"""

bruteforce_results = {}
for u_ind,v_ind in [[9,11],[9,21],[0,11],[4,16],[4,17]]:
    key = str(u_ind)+'_'+str(v_ind)
    val = np.load('results/bruteforce_kappa_u_ind_'+str(u_ind)+'_v_ind_'+str(v_ind)+'.npy')
    bruteforce_results[key] = val

result_kkk = []
n_pix=4096
for i in range(64):
    kkk = treecorr.KKKCorrelation(nbins=10,min_sep = 0.99, max_sep = n_pix/3,nubins=10,min_u=0,nvbins=11,min_v=0,max_v=1,num_threads=250)
    kkk.read('results/kappa_lognormal_'+str(i)+'.dat')
    result_kkk.append(kkk.zeta)
mean_kkk = np.mean(result_kkk,axis=0)
var_kkk = np.var(result_kkk,axis=0)/64

_,mod = create_gaussian_random_field(power_spectrum,n_pix=4096,return_scale=True)


titlesize = 9
alpha=1.

# plot all triangles
fig,ax = plt.subplots(10,11,figsize=(22*0.8,20*0.8),sharey=True,sharex=True)
plt.subplots_adjust(wspace=0,hspace=0)
for u_ind_plot in range(10):
    for v_ind_plot in range(11):
        v_ind_plot_treecorr = 11+v_ind_plot
        u_array = np.array(kkk.meanu[:,u_ind_plot,v_ind_plot_treecorr])
        v_array = np.array(kkk.meanv[:,u_ind_plot,v_ind_plot_treecorr])
        r2_array = np.array(kkk.meand1[:,u_ind_plot,v_ind_plot_treecorr])
        r3_array = u_array*r2_array
        r1_array = v_array*r3_array+r2_array
        
        ax[u_ind_plot,v_ind_plot].errorbar(kkk.meand1[:,u_ind_plot,v_ind_plot_treecorr],r2_array**2*mean_kkk[:,u_ind_plot,v_ind_plot_treecorr],r2_array**2*np.sqrt(var_kkk[:,u_ind_plot,v_ind_plot_treecorr]), 
                                           label='$\\zeta$ (treecorr)')
        ax[u_ind_plot,v_ind_plot].plot(r2_array,r2_array**2*kappa_3pcf_lognormal(r1_array,r2_array,r3_array,alpha,4096,mod=mod), 
                                       label='$\\zeta$ (analytic)')
    
        if(u_ind_plot==9):
            ax[u_ind_plot,v_ind_plot].set_xlabel('$r$')
        
        try:
            key = str(u_ind_plot)+'_'+str(v_ind_plot_treecorr)
            mean_bruteforce = np.mean(bruteforce_results[key],axis=0)
            std_bruteforce = np.sqrt(np.var(bruteforce_results[key],axis=0)/1000)
            ax[u_ind_plot,v_ind_plot].errorbar(r2_array,r2_array**2*mean_bruteforce,r2_array**2*std_bruteforce,
                                               label='$\\zeta$ (bruteforce)')
        except:
            pass
        ax[u_ind_plot,v_ind_plot].set_xscale('log')
        ax[u_ind_plot,v_ind_plot].set_title('$u='+str(round(kkk.meanu[5,u_ind_plot,v_ind_plot_treecorr],2))+',\\, v='+str(round(kkk.meanv[5,u_ind_plot,v_ind_plot_treecorr],2))+'$'
                                            ,fontsize=titlesize,y=0.82)
        ax[u_ind_plot,v_ind_plot].set_ylim(-1,7)
        plot_triangle(ax[u_ind_plot,v_ind_plot],np.mean(u_array),np.mean(v_array))
        ax[u_ind_plot,v_ind_plot].set_xticks([10,100,1000])
            
lgd = ax[0,0].legend(bbox_to_anchor=(-1.8,0), loc='upper left',prop={'size': 12})
plt.savefig('figures/treecorr_kappa.pdf',transparent=True,bbox_extra_artists=(lgd,),
           bbox_inches='tight')
plt.close()


# plot just the triangles where also bruteforce results are available

fig,ax = plt.subplots(2,3,figsize=(12,8),sharex=True)
titlesize = 12

counter = 0
for u_ind,v_ind in [[9,11],[9,21],[0,11],[4,16],[4,17]]:
    ax_x = counter//3
    ax_y = counter%3
    
    u_array = np.array(kkk.meanu[:,u_ind,v_ind])
    v_array = np.array(kkk.meanv[:,u_ind,v_ind])
    r2_array = np.array(kkk.meand1[:,u_ind,v_ind])
    r3_array = u_array*r2_array
    r1_array = v_array*r3_array+r2_array
    r_array = r2_array
    vis_mod = r_array**2

    key = str(u_ind)+'_'+str(v_ind)
    mean_bruteforce = np.mean(bruteforce_results[key],axis=0)
    std_bruteforce = np.sqrt(np.var(bruteforce_results[key],axis=0)/1000)


    ax[ax_x,ax_y].errorbar(r2_array,vis_mod*mean_kkk[:,u_ind,v_ind],vis_mod*np.sqrt(var_kkk[:,u_ind,v_ind]),label='treecorr')
    ax[ax_x,ax_y].plot(r2_array,vis_mod*kappa_3pcf_lognormal(r1_array,r2_array,r3_array,alpha,4096,mod=mod),label='analytic')
    ylims_now = ax[ax_x,ax_y].get_ylim()
    ax[ax_x,ax_y].errorbar(r2_array,vis_mod*mean_bruteforce,vis_mod*std_bruteforce,label='bruteforce')
    ax[ax_x,ax_y].set_xscale('log')
    ax[ax_x,ax_y].set_title('$u='+str(round(kkk.meanu[5,u_ind,v_ind],2))+',\\, v='+str(round(kkk.meanv[5,u_ind,v_ind],2))+'$'
                                    ,fontsize=titlesize,y=0.92)
    if(ax_x==1):
        ax[ax_x,ax_y].set_xlabel('$r$ [pix]')
    if(ax_y==0):
        ax[ax_x,ax_y].set_ylabel('$r^2\\langle\\kappa\\kappa\\kappa\\rangle$')
    ax[ax_x,ax_y].set_ylim(ylims_now)
    plot_triangle(ax[ax_x,ax_y],np.mean(u_array),np.mean(v_array))
    counter += 1
     
ax[1,2].errorbar([],[],[],label='treecorr')
ax[1,2].plot([],label='analytic')
ax[1,2].errorbar([],[],[],label='bruteforce')
ax[1,2].axis('off')
ax[1,2].legend()

plt.tight_layout()
plt.savefig('figures/treecorr_vs_bruteforce_lognormal.pdf')
plt.close()


"""
comparing ggg-correlation
"""

bruteforce_results_gamma = {}
for u_ind,v_ind in [[9,11],[9,21],[0,11],[4,16],[4,17]]:
    key = str(u_ind)+'_'+str(v_ind)
    val = np.load('results/bruteforce_gamma_u_ind_'+str(u_ind)+'_v_ind_'+str(v_ind)+'.npy')
    bruteforce_results_gamma[key] = val

result_gggr = []
result_gggi = []

n_pix=4096
for i in range(64):
    ggg = treecorr.GGGCorrelation(nbins=10,min_sep = 0.99, max_sep = n_pix/3,nubins=10,min_u=0,nvbins=11,min_v=0,max_v=1,num_threads=250)
    ggg.read('results/gamma_lognormal_'+str(i)+'.dat')
    result_gggr.append(ggg.gam0r)
    result_gggi.append(ggg.gam0i)
    
mean_ggg = np.sqrt(np.mean(result_gggr,axis=0)**2+np.mean(result_gggi,axis=0)**2)
var_ggg = (np.var(result_gggr,axis=0)**2+np.var(result_gggi,axis=0)**2)/64


fig,ax = plt.subplots(2,3,figsize=(12,8),sharex=True)
titlesize = 12

counter = 0
for u_ind,v_ind in [[9,11],[9,21],[0,11],[4,16],[4,17]]:
    ax_x = counter//3
    ax_y = counter%3
    
    u_array = np.array(ggg.meanu[:,u_ind,v_ind])
    v_array = np.array(ggg.meanv[:,u_ind,v_ind])
    r2_array = np.array(ggg.meand1[:,u_ind,v_ind])
    r3_array = u_array*r2_array
    r1_array = v_array*r3_array+r2_array
    r_array = r2_array
    vis_mod = r_array**2

    key = str(u_ind)+'_'+str(v_ind)
    mean_bruteforce = np.abs(np.mean(bruteforce_results_gamma[key],axis=0))
    std_bruteforce = np.sqrt(np.abs(np.var(bruteforce_results_gamma[key],axis=0)/1000))


    ax[ax_x,ax_y].errorbar(r2_array,vis_mod*mean_ggg[:,u_ind,v_ind],vis_mod*np.sqrt(var_ggg[:,u_ind,v_ind]),label='treecorr')
    ylims_now = ax[ax_x,ax_y].get_ylim()
    ax[ax_x,ax_y].plot([])
    ax[ax_x,ax_y].errorbar(r2_array,vis_mod*mean_bruteforce,vis_mod*std_bruteforce,label='bruteforce')
    ax[ax_x,ax_y].set_xscale('log')
    ax[ax_x,ax_y].set_title('$u='+str(round(ggg.meanu[5,u_ind,v_ind],2))+',\\, v='+str(round(ggg.meanv[5,u_ind,v_ind],2))+'$'
                                    ,fontsize=titlesize,y=0.92)
    if(ax_x==1):
        ax[ax_x,ax_y].set_xlabel('$r$ [pix]')
    if(ax_y==0):
        ax[ax_x,ax_y].set_ylabel('$r^2\\langle\\gamma\\gamma\\gamma\\rangle$')
    ax[ax_x,ax_y].set_ylim((ylims_now[0],ylims_now[1]*1.2))
    plot_triangle(ax[ax_x,ax_y],np.mean(u_array),np.mean(v_array))
    counter += 1
     
ax[1,2].errorbar([],[],[],label='treecorr')
ax[1,2].plot([])
ax[1,2].errorbar([],[],[],label='bruteforce')
ax[1,2].axis('off')
ax[1,2].legend()

plt.tight_layout()
plt.savefig('figures/treecorr_vs_bruteforce_gamma_lognormal.pdf')
plt.close()


"""
plotting the results for gaussian random fields
"""

final_results_gaussian = []

n_pix = 4096
los=0
while(os.path.exists('results/gamma_gaussian_'+str(los)+'.dat')):
    print('Reading ',los)
    ggg = treecorr.GGGCorrelation(nbins=10,min_sep = 0.99, max_sep = n_pix/3,nubins=10,min_u=0,nvbins=11,min_v=0,max_v=1,num_threads=250)
    ggg.read('results/gamma_gaussian_'+str(los)+'.dat')
    final_results_gaussian.append(ggg)
    los = los+1
    
final_g0rs_gaussian = []
final_g0is_gaussian = []
for ggg in final_results_gaussian:
    final_g0rs_gaussian.append(ggg.gam0r)
    final_g0is_gaussian.append(ggg.gam0i)


resultr_gaussian = np.mean(np.array(final_g0rs_gaussian),axis=0)
varr_gaussian = np.var(np.array(final_g0rs_gaussian),axis=0)/los
resulti_gaussian = np.mean(np.array(final_g0is_gaussian),axis=0)
vari_gaussian = np.var(np.array(final_g0is_gaussian),axis=0)/los



titlesize = 9

ggg_gauss = final_results_gaussian[0]
fig,ax = plt.subplots(10,11,figsize=(22*0.8,20*0.8),sharey=True,sharex=True)
plt.subplots_adjust(wspace=0,hspace=0)
for u_ind_plot in range(10):
    for v_ind_plot in range(11):
        v_ind_plot_treecorr = 11+v_ind_plot

        r_array = ggg_gauss.meand1[:,u_ind_plot,v_ind_plot_treecorr]
        u_array = np.array(ggg_gauss.meanu[:,u_ind_plot,v_ind_plot_treecorr])
        v_array = np.array(ggg_gauss.meanv[:,u_ind_plot,v_ind_plot_treecorr])

        ax[u_ind_plot,v_ind_plot].errorbar(r_array,r_array*resultr_gaussian[:,u_ind_plot,v_ind_plot_treecorr],r_array*np.sqrt(varr_gaussian[:,u_ind_plot,v_ind_plot_treecorr]), label='$\\Re\\Gamma^0$ (millenium)')
        ax[u_ind_plot,v_ind_plot].errorbar(r_array,r_array*resulti_gaussian[:,u_ind_plot,v_ind_plot_treecorr],r_array*np.sqrt(vari_gaussian[:,u_ind_plot,v_ind_plot_treecorr]), label='$\\Im\\Gamma^0$ (millenium)')
       
        
        if(u_ind_plot==9):
            ax[u_ind_plot,v_ind_plot].set_xlabel('$r$')
        
        ax[u_ind_plot,v_ind_plot].set_xscale('log')
        ax[u_ind_plot,v_ind_plot].set_title('$u='+str(round(ggg_gauss.meanu[5,u_ind_plot,v_ind_plot_treecorr],2))+',\\, v='+str(round(ggg_gauss.meanv[5,u_ind_plot,v_ind_plot_treecorr],2))+'$'
                                            ,fontsize=titlesize,y=0.82)
        plot_triangle(ax[u_ind_plot,v_ind_plot],np.mean(u_array),np.mean(v_array))
        ax[u_ind_plot,v_ind_plot].set_xticks([10,100,1000])

  
            
lgd = ax[0,0].legend(bbox_to_anchor=(-1.8,0), loc='upper left',prop={'size': 12})
plt.savefig('figures/gamma0_gaussian.pdf',transparent=True,bbox_extra_artists=(lgd,),
           bbox_inches='tight')
plt.close()
