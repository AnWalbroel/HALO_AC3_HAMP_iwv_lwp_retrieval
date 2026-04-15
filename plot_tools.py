import numpy as np
import matplotlib as mpl
from cmcrameri import cm

def change_colormap_len(
    cmap, 
    n_new,
    to_listed_cmap=True):

    """
    Changes the number of colours of a given colourmap (cmap) to the number given by n_new.

    Parameters:
    -----------
    cmap : matplotlib.colors element
        Colourmap used by matplotlib.
    n_new : int
        Number of colours the new colourmap should have ('length of colourmap').
    to_listed_cmap : bool
        If true, returns a mpl.colors.ListedColormap object. If False, returns an array of 
        RGB-alpha values.
    """

    len_cmap = cmap.shape[0]    # length of colourmap
    n_rgba = cmap.shape[1]      # rgb + alpha

    cmap_new = np.zeros((n_new, n_rgba))
    for m in range(n_rgba):
        cmap_new[:,m] = np.interp(np.linspace(0, 1, n_new), np.linspace(0, 1, len_cmap), cmap[:,m])

    if to_listed_cmap: cmap_new = mpl.colors.ListedColormap(cmap_new)

    return cmap_new


def get_cm_cmap(name: str, to_listed_cmap=False):
    
    cmap = cm.__dict__[name](range(len(cm.__dict__[name].colors)))
    if to_listed_cmap:
        return mpl.colors.ListedColormap(cmap)
    else:
        return cmap


def stack_cm_cmaps(bounds: np.ndarray, cmap_names: list, breakpoints=np.array([])):
    
    """
    Merge two or more cmaps into one by stacking them. Breakpoints indicating
    at which points to switch from one to the next colourmap can be provided.
    """
    
    cmap_tuple = tuple([get_cm_cmap(cmap_name) for cmap_name in cmap_names])
    n_breaks = len(breakpoints)
    if n_breaks > 0:
        bounds_broken = list()
        for k, breakpoint in enumerate(breakpoints):
            if k == 0:
                bounds_broken.append(bounds[bounds < breakpoint])
            else:
                bounds_broken.append(bounds[(bounds >= breakpoints[k-1]) & (bounds < breakpoint)])
            
            if k == n_breaks-1:
                bounds_broken.append(bounds[bounds >= breakpoint])
        cmap_tuple = tuple([change_colormap_len(cmap, len(br_bound), False) for 
                             cmap, br_bound in zip(cmap_tuple, bounds_broken)])
    
    cmap_merged = np.vstack(cmap_tuple)
    cmap_merged = change_colormap_len(cmap_merged, len(bounds))
    
    return cmap_merged


def get_colourmap_kwargs(data_lims: np.ndarray, ddata: float, cmap_data, discrete=False):

    if discrete:
        data_lvls = np.arange(data_lims[0], data_lims[1]+ddata, ddata)
        n_lvls = len(data_lvls)
        norm_data = mpl.colors.BoundaryNorm(data_lvls, n_lvls)
        cmap_data = change_colormap_len(cmap_data, n_lvls, to_listed_cmap=True)
        
        kwargs = {'norm': norm_data, 'cmap': cmap_data}
    else:
        kwargs = {'vmin': data_lims[0], 'vmax': data_lims[1], 
                  'cmap': mpl.colors.ListedColormap(cmap_data)}
        
    return kwargs


def create_colourbar(
    fig: mpl.figure.Figure, 
    axis: mpl.axes.Axes, 
    image, 
    cb_label: str, 
    xpad=0.0, 
    cbwidth=0.03, 
    cb_kwargs=dict()):
    
    x0, y0, width, height = axis.get_position().bounds
    cax = fig.add_axes([x0+width+xpad, y0, cbwidth, height])
    cb = fig.colorbar(mappable=image, cax=cax, orientation='vertical', **cb_kwargs)
    cb.set_label(cb_label)
    
    return fig, axis
