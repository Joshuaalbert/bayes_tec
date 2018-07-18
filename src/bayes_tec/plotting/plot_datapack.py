import pylab as plt
import numpy as np
import os
from concurrent import futures
from ..datapack import DataPack
from ..frames import UVW
import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
from scipy.spatial import ConvexHull, cKDTree
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors
import matplotlib
import time


try:
    import cmocean
    phase_cmap = cmocean.cm.phase
except ImportError:
    phase_cmap = plt.cm.hsv




class DatapackPlotter(object):
    def __init__(self,datapack):
        if isinstance(datapack,str):
            datapack = DataPack(filename=datapack,readonly=True)
        self.datapack = datapack
    
    def _create_polygon_plot(self,points, values=None, N = None,ax=None,cmap=plt.cm.bone,overlay_points=True,title=None,polygon_labels=None,reverse_x=False):
        # get nearest points (without odd voronoi extra regions)
        k = cKDTree(points)
        dx = np.max(points[:,0]) - np.min(points[:,0])
        dy = np.max(points[:,1]) - np.min(points[:,1])
        N = N or int(min(max(100,points.shape[0]*2),500))
        x = np.linspace(np.min(points[:,0])-0.1*dx,np.max(points[:,0])+0.1*dx,N)
        y = np.linspace(np.min(points[:,1])-0.1*dy,np.max(points[:,1])+0.1*dy,N)
        X,Y = np.meshgrid(x,y,indexing='ij')
        # interior points population
        points_i = np.array([X.flatten(),Y.flatten()]).T
        # The match per input point
        dist,i = k.query(points_i,k=1)
        # the polygons are now created using convex hulls
        # order is by point order
        patches = []
        for group in range(points.shape[0]):
            points_g = points_i[i==group,:]
            hull = ConvexHull(points_g)
            nodes = points_g[hull.vertices,:]
            poly = Polygon(nodes,closed=False)
            patches.append(poly)
        if ax is None:
            fig,ax = plt.subplots()
            print("Making new plot")
        if values is None:
            values = np.zeros(len(patches))#random.uniform(size=len(patches))
        p = PatchCollection(patches,cmap=cmap)
        p.set_array(values)
        ax.add_collection(p)
        #plt.colorbar(p)
        if overlay_points:
            ax.scatter(points[:,0],points[:,1],marker='+',c='black')
        if reverse_x:
            ax.set_xlim([np.max(points_i[:,0]),np.min(points_i[:,0])])
        else:
            ax.set_xlim([np.min(points_i[:,0]),np.max(points_i[:,0])])
        ax.set_ylim([np.min(points_i[:,1]),np.max(points_i[:,1])])
        ax.set_facecolor('black')
        if title is not None:
            if reverse_x:
                ax.text(np.max(points_i[:,0])-0.05*dx,np.max(points_i[:,1])-0.05*dy,title,ha='left',va='top',backgroundcolor=(1.,1.,1., 0.5))
            else:
                ax.text(np.min(points_i[:,0])+0.05*dx,np.max(points_i[:,1])-0.05*dy,title,ha='left',va='top',backgroundcolor=(1.,1.,1., 0.5))
#            Rectangle((x, y), 0.5, 0.5,
#    alpha=0.1,facecolor='red',label='Label'))
#            ax.annotate(title,xy=(0.8,0.8),xycoords='axes fraction')
        return ax, p
    
    def _create_image_plot(self,points, values=None, N = None,ax=None,cmap=plt.cm.bone,overlay_points=True,title=None,polygon_labels=None,reverse_x=False):
        '''
        Create initial plot, with image data instead of polygons.
        points: the locations of values, if values is None assume points are squared.
        values: array [n, m] or None, assumes (dec, ra) ordering ie (y,x)
        '''
        dx = np.max(points[:,0]) - np.min(points[:,0])
        dy = np.max(points[:,1]) - np.min(points[:,1])
        if values is not None:
            n,m = values.shape
        else:
            n=m=int(np.sqrt(points.shape[0]))
            assert n**2 == points.shape[0]
        if ax is None:
            fig,ax = plt.subplots()
            print("Making new plot")
        if values is None:
            values = np.zeros([n,m])
        x = np.linspace(np.min(points[:,0]),np.max(points[:,0]),m)
        y = np.linspace(np.min(points[:,1]),np.max(points[:,1]),n)
        img = ax.imshow(values,origin='lower',extent=(x[0],x[-1],y[0],y[-1]))
        if overlay_points:
            ax.scatter(points[:,0],points[:,1],marker='+',c='black')
        if reverse_x:
            ax.set_xlim([np.max(points_i[:,0]),np.min(points_i[:,0])])
        else:
            ax.set_xlim([np.min(points_i[:,0]),np.max(points_i[:,0])])
        ax.set_ylim([np.min(points_i[:,1]),np.max(points_i[:,1])])
        ax.set_facecolor('black')
        if title is not None:
            if reverse_x:
                ax.text(np.max(points[:,0])-0.05*dx,np.max(points[:,1])-0.05*dy,title,ha='left',va='top',backgroundcolor=(1.,1.,1., 0.5))
            else:
                ax.text(np.min(points[:,0])+0.05*dx,np.max(points[:,1])-0.05*dy,title,ha='left',va='top',backgroundcolor=(1.,1.,1., 0.5))
        return ax, img


    def plot(self, ant=None,time=None,freq=None,dir=None,pol=None, fignames=None, vmin=None,vmax=None,mode='perantenna',observable='phase',phase_wrap=True, log_scale=False, plot_crosses=True,plot_facet_idx=False,plot_patchnames=False,labels_in_radec=False,show=False, plot_arrays=False):
        """
        Plot datapack with given parameters.
        """
        SUPPORTED = ['perantenna']
        assert mode in SUPPORTED, "only 'perantenna' supported currently".format(SUPPORTED)
        if fignames is None:
            save_fig = False
            show = True
        else:
            save_fig = True
            show = show and True #False
        if plot_patchnames:
            plot_facet_idx = False
        if plot_patchnames or plot_facet_idx:
            plot_crosses = False
        if not show:
            print('turning off display')
            matplotlib.use('Agg')

        ###
        # Set up plotting

        with self.datapack:
            self.datapack.select(ant=ant,time=time,freq=freq,dir=dir,pol=pol)
            if observable == 'phase':
                obs,axes = self.datapack.phase
            elif observable == 'variance_phase':
                phase_wrap=False
                obs,axes = self.datapack.variance_phase
            elif observable == 'std':
                phase_wrap = False
                obs,axes = self.datapack.variance_phase
                obs = np.sqrt(obs)
            if 'pol' in axes.keys():
                # plot only first pol selected
                obs = obs[0,...]
            #obs is dir, ant, freq, time
            antenna_labels, antennas = self.datapack.get_antennas(axes['ant'])
            patch_names, directions = self.datapack.get_sources(axes['dir'])
            timestamps, times = self.datapack.get_times(axes['time'])
            freq_labels, freqs = self.datapack.get_freqs(axes['freq'])

        
            if phase_wrap:
                obs = np.angle(np.exp(1j*obs))
                vmin = -np.pi
                vmax = np.pi
                cmap = phase_cmap
            else:
                vmin = vmin or np.percentile(obs.flatten(),1)
                vmax = vmax or np.percentile(obs.flatten(),99)
                cmap = plt.cm.bone
            if log_scale:
                obs = np.log10(obs)

            Na = len(antennas)
            Nt = len(times)
            Nd = len(directions)
            Nf = len(freqs)
            fixfreq = Nf >> 1
            
            _, antennas_ = self.datapack.get_antennas([self.datapack.ref_ant])        
            #ants_uvw = antennas.transform_to(uvw)

            ref_dist = np.sqrt((antennas.x - antennas_.x)**2 + (antennas.y - antennas_.y)**2 + (antennas.z - antennas_.z)**2).to(au.km).value
            if labels_in_radec:
                ra = directions.ra.deg
                dec = directions.dec.deg
                points = np.array([ra,dec]).T
            else:
                fixtime = times[0]
                phase_center = self.datapack.pointing_center
                array_center = self.datapack.array_center
                uvw = UVW(location = array_center.earth_location,obstime = fixtime,phase = phase_center)
                dirs_uvw = directions.transform_to(uvw)
                u_rad = np.arctan2(dirs_uvw.u.value,dirs_uvw.w.value)
                v_rad = np.arctan2(dirs_uvw.v.value,dirs_uvw.w.value)
                points = np.array([u_rad,v_rad]).T

        if fignames is not None:
            if not isinstance(fignames,(tuple,list)):
                fignames = [fignames]
        if fignames is not None:
            assert Nt == len(fignames)


        if mode == 'perantenna':
            
            M = int(np.ceil(np.sqrt(Na)))
            fig,axs = plt.subplots(nrows=M,ncols=M,sharex='col',sharey='row',squeeze=False, \
                    figsize=(4*M,4*M))
            fig.subplots_adjust(wspace=0., hspace=0.)
            axes_patches = []
            c = 0
            for row in range(M):
                for col in range(M):
                    ax = axs[row,col]
                    if col == 0:
                        ax.set_ylabel("Projected North (radians)" if not labels_in_radec else "DEC (deg)")
                            
                    if row == M - 1:
                        ax.set_xlabel("Projected East (radians)" if not labels_in_radec else "RA (deg)")
                    
                    if c >= Na:
                        continue
                    try:
                        title = antenna_labels[c].decode()
                    except:
                        title = antenna_labels[c]
                    _, p = self._create_polygon_plot(points, values=None, N = None,
                            ax=ax,cmap=cmap,overlay_points=plot_crosses,
                            title="{} {:.1f}km".format(title, ref_dist[c]),
                            reverse_x=labels_in_radec)
                    p.set_clim(vmin,vmax)
                    axes_patches.append(p)
                    c += 1
            
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(p, cax=cbar_ax, orientation='vertical')
            if show:
                plt.ion()
                plt.show()
            for j in range(Nt):
                print("Plotting {}".format(timestamps[j]))
                for i in range(Na):
                    axes_patches[i].set_array(obs[:,i,fixfreq,j])
                axs[0,0].set_title("{} {} : {}".format(observable, freq_labels[fixfreq], timestamps[j]))
                fig.canvas.draw()
                if save_fig:
                    plt.savefig(fignames[j])

            if show:
#                plt.close(fig)
                plt.ioff()

def _parallel_plot(arg):
    datapack,time_slice,kwargs,output_folder=arg
    dp = DatapackPlotter(datapack=datapack)
    _,axes = dp.datapack.phase
    times = axes['time']
    fignames = [os.path.join(output_folder,"fig-{:04d}.png".format(j)) for j in range(len(times))[time_slice]]
    dp.plot(time=time_slice,fignames=fignames,**kwargs)
    return fignames
    
def animate_datapack(datapack,output_folder,num_processes,**kwargs):
    """
    Plot the datapack in parallel, then stitch into movie.
    datapack: str the datapack filename
    output_folder: str, folder to store figs in
    num_processes: int number of parallel plotting processes to run
    **kwargs: keywords to pass to DatapackPlotter.plot function.
    """
    try:
        os.makedirs(output_folder)
    except:
        pass

    with DataPack(datapack) as datapack:
        datapack.add_antennas(DataPack.lofar_array)

    args = []
    for i in range(num_processes):
        args.append((datapack,slice(i,None,num_processes),kwargs,output_folder))
    with futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        jobs = executor.map(_parallel_plot,args)
        results = list(jobs)
    make_animation(output_folder,prefix='fig',fps=4)

def make_animation(datafolder,prefix='fig',fps=4):
    '''Given a datafolder with figures of format `prefix`-%04d.png create a 
    video at framerate `fps`.
    Output is datafolder/animation.mp4'''
    if os.system('ffmpeg -framerate {} -i {}/{}-%04d.png -vf scale="trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -profile:v high -pix_fmt yuv420p -g 30 -r 30 {}/animation.mp4'.format(fps,datafolder,prefix,datafolder)):
        print("{}/animation.mp4 exists already".format(datafolder))    



def test_vornoi():
    from scipy.spatial import Voronoi, voronoi_plot_2d
    import pylab as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import numpy as np

    points = np.random.uniform(size=[10,2])
    v = Voronoi(points)
    nodes = v.vertices
    regions = v.regions

    ax = plt.subplot()
    patches = []
    for reg in regions:
        if len(reg) < 3:
            continue
        poly = Polygon(np.array([nodes[i] for i in reg]),closed=False)
        patches.append(poly)
    p = PatchCollection(patches)
    p.set_array(np.random.uniform(size=len(patches)))
    ax.add_collection(p)
    #plt.colorbar(p)
    ax.scatter(points[:,0],points[:,1])
    ax.set_xlim([np.min(points[:,0]),np.max(points[:,0])])
    ax.set_ylim([np.min(points[:,1]),np.max(points[:,1])])
    plt.show()

def test_nearest():
    from scipy.spatial import ConvexHull, cKDTree
    import pylab as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import numpy as np

    points = np.random.uniform(size=[42,2])
    k = cKDTree(points)
    dx = np.max(points[:,0]) - np.min(points[:,0])
    dy = np.max(points[:,1]) - np.min(points[:,1])
    N = int(min(max(100,points.shape[0]*2),500))
    x = np.linspace(np.min(points[:,0])-0.1*dx,np.max(points[:,0])+0.1*dx,N)
    y = np.linspace(np.min(points[:,1])-0.1*dy,np.max(points[:,1])+0.1*dy,N)
    X,Y = np.meshgrid(x,y,indexing='ij')
    points_i = np.array([X.flatten(),Y.flatten()]).T
    dist,i = k.query(points_i,k=1)
    patches = []
    for group in range(points.shape[0]):
        points_g = points_i[i==group,:]
        hull = ConvexHull(points_g)
        nodes = points_g[hull.vertices,:]
        poly = Polygon(nodes,closed=False)
        patches.append(poly)
    ax = plt.subplot()
    p = PatchCollection(patches)
    p.set_array(np.random.uniform(size=len(patches)))
    ax.add_collection(p)
    #plt.colorbar(p)
    ax.scatter(points[:,0],points[:,1])
    ax.set_xlim([np.min(points_i[:,0]),np.max(points_i[:,0])])
    ax.set_ylim([np.min(points_i[:,1]),np.max(points_i[:,1])])
    ax.set_facecolor('black')
    plt.show()

def test():
    from ionotomo.astro.real_data import generate_example_datapack
    datapack = generate_example_datapack(Ndir=10,Nant=10,Ntime=20)
    datapack.phase = np.random.uniform(size=datapack.phase.shape)
    dp = DatapackPlotter(datapack='../data/rvw_datapack_full_phase_dec27_smooth.hdf5')
    dp.plot(ant_idx=[50],dir_idx=-1,time_idx=[0],labels_in_radec=True,show=True)

#    animate_datapack('../data/rvw_datapack_full_phase_dec27_smooth.hdf5',
#            'test_output',num_processes=1,observable='phase',labels_in_radec=True,show=True)




if __name__=='__main__':
    test()
