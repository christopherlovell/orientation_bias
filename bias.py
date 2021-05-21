import numpy as np

from scipy.stats import truncnorm, binned_statistic
from scipy.integrate import quad

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


class orientation_bias():

    def __init__(self, lambda_obs=250, volume=50):

        self.lambda_obs = lambda_obs
        self.volume = volume
        self.dim_params = [2.40195197e-10, -5.57233507e-06, 6.84216592e-03, 5.27412031e-01]


    def sample_S(self,_model,**kwargs):
        # self.S = 10**model.sample(volume=self.volume, D_lowlim=0, inf_lim=3)
        self.S = 10**model.sample(volume=self.volume, **kwargs)


    def trunc_gaussian(self,mean=2, std=1.0, low_lim=0, upp_lim=5):
        _a, _b = (low_lim - mean) / std, (upp_lim - mean) / std
        return truncnorm.rvs(_a, _b, loc=mean, scale=std, size=len(self.S))


    def sample_z(self, _method, **kwargs):
        ## sample from redshift distribution
        self.redshift = _method(**kwargs)
        self.lambda_rest = self.lambda_obs / (1+self.redshift)


    def dimming_distribution(self,x):#,p1,p2,p3,p4):
        p1=self.dim_params[0]
        p2=self.dim_params[1]
        p3=self.dim_params[2]
        p4=self.dim_params[3]
        return p1*x**3 + p2*x**2 + p3*x + p4


    def sample_dimming(self, _method, max_dim = 0.9, *_args):

        self.a = 10**_method(self.lambda_rest, *_args)
        # a = 10**f(wl_rest, *self.dim_params)

        dimming = np.random.exponential(scale=1/self.a, size=len(self.a))
        # max_dim = 0.9
        while np.sum(dimming > max_dim) > 0:
            dimming[dimming > max_dim] = \
                    np.random.exponential(scale=1/self.a[dimming > max_dim], 
                                          size=np.sum(dimming > max_dim))

        self.dimming = dimming

        self.new_S = self.S * (1-self.dimming)


    def plot_dimming_redshift(self):
        ## plot dimming as a function of redshift
        fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(5,12))
        plt.subplots_adjust(hspace=0.)
        
        binlims = np.linspace(0,5,10)
        bins = binlims[:-1] + np.diff(binlims)[0]/2
        dim_dist = [binned_statistic(self.redshift, self.dimming,
                                     statistic=lambda y: np.percentile(y, p), bins=binlims)[0] \
                                          for p in [16,50,84]]
        
        _im = ax2.hexbin(self.redshift, self.dimming, gridsize=(30,30), cmap='Blues')
        cax = fig.add_axes([0.15, 0.45, 0.02, 0.14])
        fig.colorbar(_im, cax=cax, label='$N$')
        ax2.errorbar(bins, dim_dist[1], yerr=[dim_dist[1] - dim_dist[0],dim_dist[2] - dim_dist[1]],
                     linestyle='none', color='grey', marker='o',
                     markeredgewidth=1, markeredgecolor='black')
        
        ax2.set_xlabel('$z$'); ax2.set_ylabel('$D \;[\mathrm{Dimming}]$'); ax2.set_ylim(0,0.33)
        ax2.set_xticklabels([])
        
        ## plot redshift distribution for different selections
        slim = 30
        binlims = np.linspace(0,5,30); bins = binlims[:-1] + np.diff(binlims)[0]/2
        ax1.hist(self.redshift, histtype='step', density=True, bins=binlims,
                label='$\mathrm{All}$')
        ax1.hist(self.redshift[self.S > slim], histtype='step', density=True, 
                 label='$S > S_{\mathrm{lim}}$', bins=binlims)
        ax1.hist(self.redshift[self.new_S > slim], histtype='step', density=True,
                 label='$S_{\mathrm{dimmed}} > S_{\mathrm{lim}}$', bins=binlims)
        
        ax1.legend()
        ax1.xaxis.tick_top(); ax1.set_xlabel('$z$'); ax1.xaxis.set_label_position('top')
        ax1.set_ylabel('$N(\mathrm{normalised})$')
        ax1.text(0.73, 0.64, '$S_{\mathrm{lim}} = %.1f$'%(slim), transform=ax1.transAxes)
        
        for ax in [ax1,ax2,ax3]: ax.grid(alpha=0.4)
        for ax in [ax1,ax2,ax3]: ax.set_xlim(0,5)
        
        ## plot completeness as a function of redshift
        slims = np.array([4,8,16,32,64])
        cm = plt.get_cmap('cividis')
        cNorm  = matplotlib.colors.Normalize(vmin=slims.min(), vmax=slims.max())
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
        
        for slim in slims:
            binlims = np.linspace(0,5,10); bins = binlims[:-1] + np.diff(binlims)[0]/2
            _dist = binned_statistic(self.redshift,(self.new_S,self.S), 
                                     statistic=lambda y: np.sum(y > slim), bins=binlims)[0]

            ax3.plot(bins,_dist[0]/_dist[1],label='$S_{lim} = %.1f$'%slim, color=scalarMap.to_rgba(slim))
        
        ax3.set_ylim(0.3,0.999)
        ax3.set_xlabel('z'); ax3.set_ylabel('Completeness')
        cax2 = fig.add_axes([0.25, 0.13, 0.5, 0.015])
        cbar = fig.colorbar(scalarMap, label='$S_{\mathrm{lim}}$', cax=cax2, orientation='horizontal')
        cax2.xaxis.set_label_position('top')
        cax2.xaxis.set_ticks_position('top')
        
        return fig


    def plot_dimming_distribution(self):
        Ns = 5
        Slim_array = np.array([4,8,16,32,64])
        binlimits = np.linspace(0,1.0,21)
        bins = binlimits[1:] - ((binlimits[1] - binlimits[0])/2)
        
        cmap = plt.cm.get_cmap('cividis', len(Slim_array))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=Slim_array.min(), 
                                   vmax=Slim_array.max()))
        sm.set_clim(0,Ns)
        
        fig, ax = plt.subplots(1,1,figsize=(6,5))
        
        for i, slim in enumerate(Slim_array):
            _N = list(np.histogram(self.dimming[self.S > slim], bins=binlimits)[0])
            _NB = list(np.histogram(self.dimming[self.new_S > slim], bins=binlimits)[0])
        
            _N.append(_N[-1])
            _NB.append(_NB[-1])
        
            ax.step(binlimits, np.array(_NB) / np.array(_N), color=cmap(i/Ns), where='post')
        
        
        ax.set_xlim(0,1.0)
        ax.set_ylim(0,1)
        ax.set_xlabel('$D$ [Dimming]')
        ax.set_ylabel('$N_{\mathrm{dim}} \,/\, N$')
        ax.grid(alpha=0.2)
        cbar = fig.colorbar(sm)
        cbar.set_ticks(np.arange(Ns)+0.5)
        cbar.set_ticklabels(Slim_array)
        cbar.set_label('$S_{\mathrm{lim}}$')
        return fig
       

    def calc_phi(self,S,volume,binlimits=None):
        if binlimits is None:
            _n,binlims = np.histogram(S)
        else:
            _n,binlims = np.histogram(S,bins=binlimits)

        bins = binlims[:-1] + (binlims[1:] - binlims[:-1])/2
        _y = (_n/volume)/(binlims[1] - binlims[0])
        return _y, bins


    def plot_number_counts(self):
        fig, ax = plt.subplots(1,1, figsize=(6,5))

        cmap = plt.cm.get_cmap('viridis')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0,1))
        
        binlimits = np.linspace(-0.3,2.3,180)
        _phi,bins = self.calc_phi(np.log10(self.new_S), self.volume, binlimits=binlimits)
        ax.plot(10**bins, np.log10(_phi), label='un-dimmed', color='black', linestyle='dashed')
        
        _phi,bins = self.calc_phi(np.log10(self.S), self.volume, binlimits=binlimits)
        ax.plot(10**bins, np.log10(_phi), label='un-dimmed', color='red', linestyle='dashed')
        
        ax.set_xscale('log')
        
        slim = binlimits[100] # [95] # [80]
        _N = np.histogram(np.log10(self.S[self.new_S > 10**slim]), bins=binlimits[binlimits>=slim])[0]
        _NT = np.histogram(np.log10(self.S[self.S > 10**slim]), bins=binlimits[binlimits>=slim])[0]
        
        for _c,_bl in zip(_N/_NT,binlimits[binlimits>=slim]):
            _x = np.linspace(_bl,_bl+np.diff(binlimits)[0],2)
            _bins = _x[1:] - ((_x[1] - _x[0])/2)
            _y = np.log10(self.calc_phi(np.log10(self.new_S),self.volume,binlimits=_x)[0])
            ax.fill_between(10**_x, np.hstack([_y,_y]), y2=-1, color=cmap(_c))
        
        y_upp = 4.3 #4.6 # 4.3
        ax.set_ylim(2.6,y_upp) # (1,y_upp)
        ax.set_xlim(10,80) # (5,90) # (8,80)
        
        _x_completeness = 10**binlimits[binlimits>=slim][np.min(np.where((_N/_NT) > 0.95))]
        ax.text(_x_completeness*1.05, y_upp*0.91, '$S_{95} = %.2f \, \mathrm{mJy}$'%(_x_completeness))
        ax.vlines(_x_completeness, -1, y_upp*0.94, linestyle='dotted', color='black')
        ax.text(10**slim * 1.05, y_upp * 0.96, '$S_{\mathrm{lim}} = %.2f \, \mathrm{mJy}$'%10**slim)
        ax.vlines(10**slim, -1, y_upp, linestyle='-.', color='black')
        
        cbar = fig.colorbar(sm)
        cbar.set_label('Completeness')
        
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_formatter(formatter)
        
        ax.set_xlabel('$\mathrm{log_{10}}(S \,/\, \mathrm{mJy})$')
        ax.set_ylabel('$\phi \,/\, (\mathrm{deg^{-2} \; dex^{-1}})$')
        ax.text(0.7,0.9,'$\lambda_{\mathrm{obs}} = %.1f \, \mathrm{\mu m}$'%self.lambda_obs,
                transform=ax.transAxes)

        return fig


class Schechter():

    def __init__(self, Dstar=1e-2, alpha=-1.4, log10phistar=4):
        self.sp = {}
        self.sp['D*'] = Dstar
        self.sp['alpha'] = alpha
        self.sp['log10phistar'] = log10phistar


    def _integ(self, x,a,D):
        return 10**((a+1)*(x-D)) * np.exp(-10**(x-D))


    def binPhi(self, D1, D2):
        args = (self.sp['alpha'],self.sp['D*'])
        gamma = quad(self._integ, D1, D2, args=args)[0]
        return gamma * 10**self.sp['log10phistar'] * np.log(10)


    def _CDF(self, D_lowlim, normed = True, inf_lim=30):
        log10Ls = np.arange(self.sp['D*']+5.,D_lowlim-0.01,-0.01)
        CDF = np.array([self.binPhi(log10L,inf_lim) for log10L in log10Ls])
        if normed: CDF /= CDF[-1]

        return log10Ls, CDF


    def sample(self, volume, D_lowlim, inf_lim=100):
        D, cdf = self._CDF(D_lowlim, normed=False, inf_lim=inf_lim)
        n = np.random.poisson(volume * cdf[-1])
        ncdf = cdf/cdf[-1]
        D_sample = np.interp(np.random.random(n), ncdf, D)

        return D_sample



if __name__ == "__main__":
    bias = orientation_bias()
    
    # first define our number counts model
    model = Schechter(Dstar=1.50, alpha=-1.91, log10phistar=3.56)  # 250 mu-metre

    # then sample from this model to get `S`, an array of flux densities
    bias.sample_S(model, D_lowlim=0, inf_lim=3)
    
    print("S:", bias.S)

    # sample from a redshift distribution (here a gaussian, with mean z = 1.2)
    bias.sample_z(bias.trunc_gaussian, mean=1.2)

    print("Redshift:", bias.redshift)

    # sample from the dimming distribution for each source
    bias.sample_dimming(bias.dimming_distribution)
    
    print("Dimming:", bias.dimming)

    fig = bias.plot_dimming_redshift()
    plt.show()

    fig = bias.plot_dimming_distribution()
    plt.show()

    fig = bias.plot_number_counts()
    plt.show()



