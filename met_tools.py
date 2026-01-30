import numpy as np
import xarray as xr
import pdb
import copy



# constants:
R_d = 287.0597  # gas constant of dry air, in J kg-1 K-1
R_v = 461.5     # gas constant of water vapour, in J kg-1 K-1
R_ = 8.314462618    # universal gas constant, in J mol-1 K-1, https://physics.nist.gov/cgi-bin/cuu/Value?r
M_dv = R_d / R_v # molar mass ratio , in ()
m_mol_air = 0.0289647       # molar mass of dry air, in kg mol-1, https://www.engineeringtoolbox.com/molecular-mass-air-d_679.html
mw_h2o = 0.01802  # h2o molar mass in kg mol-1
e_0 = 611       # saturation water vapour pressure at freezing point (273.15 K), in Pa
T0 = 273.15     # freezing temperature, in K
g = 9.80665     # gravitation acceleration, in m s^-2 (from https://doi.org/10.6028/NIST.SP.330-2019 )
c_pd = 1005.7   # specific heat capacity of dry air at constant pressure, in J kg-1 K-1
c_vd = 719.0    # specific heat capacity of dry air at constant volume, in J kg-1 K-1
c_h2o = 4187.0  # specific heat capacity of water at 15 deg C; in J kg-1 K-1
L_v = 2.501e+06 # latent heat of vaporization, in J kg-1
omega_earth = 2*np.pi / 86164.09    # earth's angular velocity: World Book Encyclopedia Vol 6. Illinois: World Book Inc.: 1984: 12.
R_e = 6371000   # volumetric radius of earth in m, https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html


def compute_IWV(
    rho_v,
    z,
    nan_threshold=0.0,
    scheme='balanced'):

    """
    Compute Integrated Water Vapour (also known as precipitable water content)
    out of absolute humidity (in kg m^-3) and height (in m).
    The moisture data may contain certain number gaps (up to nan_threshold*n_levels) but
    the height variable must be free of gaps.

    Parameters:
    -----------
    rho_v : array of floats
        One dimensional array of absolute humidity in kg m^-3.
    z : array of floats
        One dimensional array of sorted height axis (ascending order) in m.
    nan_threshold : float, optional
        Threshold describing the fraction of nan values of the total height level
        number that is still permitted for computation.
    scheme : str, optional
        Chose the scheme 'balanced' or 'top_weighted'. They differ in the way the altitude
        levels are used to compute IWV. Recommendation and default: 'balanced'
    """

    # Check if the height axis is sorted in ascending order:
    if np.any(np.diff(z) < 0):
        print("Warning! Height axis must be in ascending order to compute the integrated" +
            " water vapour.")

        # if the pressure data is okay until 300 hPa, compute IWV nonetheless and truncate the
        # profile beyond:
        where_broken = np.where(np.diff(z) < 0)[0]      # when where_broken == 152, then z[153] - z[152] is broken
        if z[where_broken[0]] < 9000.0: # then, sufficient altitude doesn't have valid data valid data, return IWV=nan
            return IWV

    # truncate data to non nan height or pressure levels:
    non_nan_idx = np.where(~np.isnan(z))[0]
    rho_v = rho_v[non_nan_idx[0]:non_nan_idx[-1]+1]
    z = z[non_nan_idx[0]:non_nan_idx[-1]+1]

    # check if height axis is free of gaps:
    if np.any(np.isnan(np.diff(z))): 
        print("Height axis contains gaps. Aborted IWV computation.")
        return IWV


    n_height = len(z)
    # Check if rho_v has got any gaps:
    n_nans_rho_v = np.count_nonzero(np.isnan(rho_v))


    # If no nans exist, the computation is simpler. If some nans exist IWV will still be
    # computed but needs to look for the next non-nan value. If too many nans exist IWV
    # won't be computed.
    if scheme == 'balanced':
        if (n_nans_rho_v == 0):

            IWV = 0.0
            for k in range(n_height):
                if k == 0:      # bottom of grid
                    dz = 0.5*(z[k+1] - z[k])        # just a half of a level difference

                elif k == n_height-1:   # top of grid
                    dz = 0.5*(z[k] - z[k-1])        # the other half level difference

                else:           # mid of grid
                    dz = 0.5*(z[k+1] - z[k-1])

                IWV = IWV + rho_v[k]*dz

        elif n_nans_rho_v / n_height < nan_threshold:

            # Loop through height grid:
            IWV = 0.0
            k = 0
            prev_nonnan_idx = -1
            while k < n_height:
                
                # check if hum on current level is nan:
                # if so search for the next non-nan level:
                if np.isnan(rho_v[k]):
                    next_nonnan_idx = np.where(~np.isnan(rho_v[k:]))[0]

                    if (len(next_nonnan_idx) > 0) and (prev_nonnan_idx >= 0):   # mid or near top of height grid
                        next_nonnan_idx = next_nonnan_idx[0] + k    # plus k because searched over part of rho_v
                        IWV += 0.25*(rho_v[next_nonnan_idx] + rho_v[prev_nonnan_idx])*(z[k+1] - z[k-1])
                    
                    elif (len(next_nonnan_idx) > 0) and (prev_nonnan_idx < 0):  # bottom of height grid
                        next_nonnan_idx = next_nonnan_idx[0] + k    # plus k because searched over part of rho_v

                        # fixing height grid variable in case only the lowest measurement doesn't exist:
                        if np.isnan(z[0]) and not (np.isnan(z[1]+z[2])):
                            IWV += 0.5*rho_v[next_nonnan_idx]*(z[2] - z[1])
                        else:
                            IWV += 0.5*rho_v[next_nonnan_idx]*(z[k+1] - z[k])
                        

                    else: # reached top of grid
                        IWV += 0.0

                else:
                    prev_nonnan_idx = k

                    if k == 0:          # bottom of grid
                        IWV += 0.5*rho_v[k]*(z[k+1] - z[k])
                    elif k == 1 and np.isnan(z[k-1]):   # next to bottom of grid
                        IWV += 0.5*rho_v[k]*(z[k+1] - z[k])
                    elif (k > 0) and (k < n_height-1):  # mid of grid
                        IWV += 0.5*rho_v[k]*(z[k+1] - z[k-1])
                    else:               # top of grid
                        IWV += 0.5*rho_v[k]*(z[-1] - z[-2])

                k += 1      

        else:
            IWV = np.nan


    elif scheme == 'top_weighted':
        if (n_nans_rho_v == 0):

            IWV = 0.0
            for k in range(n_height):
                if k < n_height-2:      # bottom or mid of grid
                    dz = z[k+1] - z[k]

                else:   # top and next to top of grid
                    dz = 0.5*(z[-1] - z[-2])        # half the height for top two levels

                IWV = IWV + rho_v[k]*dz

        elif n_nans_rho_v / n_height < nan_threshold:

            # Loop through height grid:
            IWV = 0.0
            k = 0
            prev_nonnan_idx = -1
            while k < n_height:
                
                # check if hum on current level is nan:
                # if so search for the next non-nan level:
                if np.isnan(rho_v[k]):
                    next_nonnan_idx = np.where(~np.isnan(rho_v[k:]))[0]

                    if (len(next_nonnan_idx) > 0) and (prev_nonnan_idx >= 0):   # mid of height grid
                        next_nonnan_idx = next_nonnan_idx[0] + k    # plus k because searched over part of rho_v

                        if k+1 != n_height-1:
                            IWV += 0.5*(rho_v[next_nonnan_idx] + rho_v[prev_nonnan_idx])*(z[k+1] - z[k])
                        else:   # near top of grid
                            IWV += 0.25*(rho_v[next_nonnan_idx] + rho_v[prev_nonnan_idx])*(z[k+1] - z[k])
                    
                    elif (len(next_nonnan_idx) > 0) and (prev_nonnan_idx < 0):  # bottom of height grid
                        next_nonnan_idx = next_nonnan_idx[0] + k    # plus k because searched over part of rho_v

                        if np.isnan(z[0]) and not (np.isnan(z[1]+z[2])):
                            IWV += rho_v[next_nonnan_idx]*(z[2] - z[1])
                        else:
                            IWV += rho_v[next_nonnan_idx]*(z[k+1] - z[k])
                        

                    else: # reached top of grid
                        IWV += 0.0

                else:
                    prev_nonnan_idx = k

                    if k < n_height-2:  # bottom or mid of grid
                        IWV += rho_v[k]*(z[k+1] - z[k])
                    else:               # top of grid
                        IWV += 0.5*rho_v[k]*(z[-1] - z[-2])

                k += 1      

        else:
            IWV = np.nan
        
    return IWV


def compute_IWV_q(
    q,
    press,
    nan_threshold=0.0,
    scheme='balanced'):

    """
    Compute Integrated Water Vapour (also known as precipitable water content)
    out of specific humidity (in kg kg^-1), gravitational constant and air pressure (in Pa).
    The moisture data may contain certain number gaps (up to nan_threshold*n_levels) but
    the height variable must be free of gaps.

    Parameters:
    -----------
    q : array of floats
        One dimensional array of specific humidity in kg kg^-1.
    press : array of floats
        One dimensional array of pressure in Pa.
    nan_threshold : float, optional
        Threshold describing the fraction of nan values of the total height level
        number that is still permitted for computation.
    scheme : str, optional
        Chose the scheme 'balanced' or 'top_weighted'. They differ in the way the altitude
        levels are used to compute IWV. Recommendation and default: 'balanced'
    """

    IWV = np.nan

    # Check if the Pressure axis is sorted in descending order:
    if np.any(np.diff(press) > 0):
        print("Warning! Height axis must be in ascending order (pressure in descending) to compute the integrated" +
            " water vapour.")

        # if the pressure data is okay until 300 hPa, compute IWV nonetheless and truncate the
        # profile beyond:
        where_broken = np.where(np.diff(press) > 0)[0]      # when where_broken == 152, then press[153] - press[152] is broken
        if press[where_broken[0]] > 30000.0:    # then, sufficient altitude doesn't have valid data valid data, return IWV=nan
            return IWV

    # truncate data to non nan height or pressure levels:
    non_nan_idx = np.where(~np.isnan(press))[0]
    q = q[non_nan_idx[0]:non_nan_idx[-1]+1]
    press = press[non_nan_idx[0]:non_nan_idx[-1]+1]

    # check if height axis is free of gaps:
    if np.any(np.isnan(np.diff(press))): 
        print("Height axis contains gaps. Aborted IWV computation.")
        return IWV


    n_height = len(press)
    # Check if q has got any gaps:
    n_nans = np.count_nonzero(np.isnan(q))


    # If no nans exist, the computation is simpler. If some nans exist IWV will still be
    # computed but needs to look for the next non-nan value. If too many nans exist IWV
    # won't be computed.
    if scheme == 'balanced':
        if (n_nans == 0):

            IWV = 0.0
            for k in range(n_height):
                if k == 0:      # bottom of grid
                    dp = 0.5*(press[k+1] - press[k])        # just a half of a level difference

                elif k == n_height-1:   # top of grid
                    dp = 0.5*(press[k] - press[k-1])        # the other half level difference

                else:           # mid of grid
                    dp = 0.5*(press[k+1] - press[k-1])

                IWV = IWV - q[k]*dp

        elif n_nans / n_height < nan_threshold:

            # Loop through height grid:
            IWV = 0.0
            k = 0
            prev_nonnan_idx = -1
            while k < n_height:

                # check if hum on current level is nan:
                # if so search for the next non-nan level:
                if np.isnan(q[k]):
                    next_nonnan_idx = np.where(~np.isnan(q[k:]))[0]

                    if (len(next_nonnan_idx) > 0) and (prev_nonnan_idx >= 0):   # mid or near top of height grid
                        next_nonnan_idx = next_nonnan_idx[0] + k    # plus k because searched over part of rho_v
                        IWV -= 0.25*(q[next_nonnan_idx] + q[prev_nonnan_idx])*(press[k+1] - press[k-1])
                    
                    elif (len(next_nonnan_idx) > 0) and (prev_nonnan_idx < 0):  # bottom of height grid
                        next_nonnan_idx = next_nonnan_idx[0] + k    # plus k because searched over part of q

                        # fixing height grid variable in case only the lowest measurement doesn't exist:
                        if np.isnan(press[0]) and not (np.isnan(press[1]+press[2])):
                            IWV -= 0.5*q[next_nonnan_idx]*(press[2] - press[1])
                        else:
                            IWV -= 0.5*q[next_nonnan_idx]*(press[k+1] - press[k])

                    else: # reached top of grid
                        IWV += 0.0

                else:
                    prev_nonnan_idx = k

                    if k == 0:          # bottom of grid
                        IWV -= 0.5*q[k]*(press[k+1] - press[k])
                    elif k == 1 and np.isnan(press[k-1]):       # next to bottom of grid
                        IWV -= 0.5*q[k]*(press[k+1] - press[k])
                    elif (k > 0) and (k < n_height-1):  # mid of grid
                        IWV -= 0.5*q[k]*(press[k+1] - press[k-1])
                    else:               # top of grid
                        IWV -= 0.5*q[k]*(press[-1] - press[-2])

                k += 1

        else:
            IWV = np.nan


    elif scheme == 'top_weighted':
        if (n_nans == 0):

            IWV = 0.0
            for k in range(n_height):
                if k < n_height-2:      # bottom or mid of grid
                    dp = press[k+1] - press[k]

                else:   # top and next to top of grid
                    dp = 0.5*(press[-1] - press[-2])        # half the height for top two levels

                IWV = IWV - q[k]*dp

        elif n_nans / n_height < nan_threshold:

            # Loop through height grid:
            IWV = 0.0
            k = 0
            prev_nonnan_idx = -1
            while k < n_height:
                
                # check if hum on current level is nan:
                # if so search for the next non-nan level:
                if np.isnan(q[k]):
                    next_nonnan_idx = np.where(~np.isnan(q[k:]))[0]

                    if (len(next_nonnan_idx) > 0) and (prev_nonnan_idx >= 0):   # mid of height grid
                        next_nonnan_idx = next_nonnan_idx[0] + k    # plus k because searched over part of q

                        if k+1 != n_height-1:
                            IWV -= 0.5*(q[next_nonnan_idx] + q[prev_nonnan_idx])*(press[k+1] - press[k])
                        else:   # near top of grid
                            IWV -= 0.25*(q[next_nonnan_idx] + q[prev_nonnan_idx])*(press[k+1] - press[k])
                    
                    elif (len(next_nonnan_idx) > 0) and (prev_nonnan_idx < 0):  # bottom of height grid
                        next_nonnan_idx = next_nonnan_idx[0] + k    # plus k because searched over part of q

                        # fixing height grid variable in case only the lowest measurement doesn't exist:
                        if np.isnan(press[0]) and not (np.isnan(press[1]+press[2])):
                            IWV -= q[next_nonnan_idx]*(press[2] - press[1])
                        else:
                            IWV -= q[next_nonnan_idx]*(press[k+1] - press[k])
                        

                    else: # reached top of grid
                        IWV += 0.0

                else:
                    prev_nonnan_idx = k

                    if k < n_height-2:  # bottom or mid of grid
                        IWV -= q[k]*(press[k+1] - press[k])
                    else:               # top of grid
                        IWV -= 0.5*q[k]*(press[-1] - press[-2])

                k += 1

        else:
            IWV = np.nan


    IWV = IWV / g       # yet had to be divided by gravitational acceleration

    return IWV


def wspeed_wdir_to_u_v(
    wspeed,
    wdir,
    convention='towards'):

    """
    This will compute u and v wind components from wind speed and wind direction
    (in deg from northward facing wind). u and v will have the same units as
    wspeed. The default convention is that wdir indicates where the wind will flow
    to. Note, that meteorological wind direction is defined as from where the wind is coming
    from.

    Parameters:
    -----------
    wspeed : array of float or int
        Wind speed array.
    wdir : array of float or int
        Wind direction in deg from northward facing (or northerly) wind (for convention
        = towards, the wind flows northwards, wdir is 0; for convention = from, the wind
        comes from the north for wdir = 0).
    convention : str
        Convention of how wdir is to be interpreted. Options: 'towards' means that
        wdir indicates where the wind points to (where parcels will move to); 'from'
        means that wdir indicates where the wind comes from.
    """

    if convention == 'towards':
        wdir_rad = np.radians(wdir)
        u = np.sin(wdir_rad)*wspeed
        v = np.cos(wdir_rad)*wspeed

    elif convention == 'from':
        wdir_rad = np.radians(wdir+180)
        wdir_rad[wdir_rad > 2*np.pi] -= 2*np.pi

        u = np.sin(wdir_rad)*wspeed
        v = np.cos(wdir_rad)*wspeed

    return u, v


def u_v_to_wspeed_wdir(
    u,
    v,
    convention='towards'):

    """
    This will compute wind speed (in units of u and v) and wind direction (in deg from 
    northward facing or from north coming wind (depends on convention)) from u and v wind 
    components.The default convention is that wdir indicates where the wind will flow
    to. Note, that meteorological wind direction is defined as from where the wind is coming
    from.

    Parameters:
    -----------
    u : array of float or int
        Zonal component of wind (eastwards > 0).
    v : array of float or int
        Meridional component of wind (northwards > 0).
    convention : str
        Convention of how wdir is to be interpreted. Options: 'towards' means that
        wdir indicates where the wind points to (where parcels will move to); 'from'
        means that wdir indicates where the wind comes from.
    """

    assert u.shape == v.shape   # check if both have the same dimension

    # flatten array and put it back into shape later.
    u_shape = u.shape
    u = u.flatten()
    v = v.flatten()
    wspeed = (u**2.0 + v**2.0)**0.5

    if convention == 'from':
        u *= (-1.0)
        v *= (-1.0)

    # distinguish the two semi circles to compute the correct wind direction:
    u_greater_0 = np.where(u >= 0)[0]
    u_smaller_0 = np.where(u < 0)[0]

    # compute wind direction based on the semi circle:
    wdir = np.zeros(u.shape)
    wdir[u_greater_0] = np.arccos(v[u_greater_0] / wspeed[u_greater_0])
    wdir[u_smaller_0] = 2.0*np.pi - np.arccos(v[u_smaller_0] / wspeed[u_smaller_0])

    # convert wdir to deg:
    wdir = np.degrees(wdir)

    # back to old shape:
    wspeed = np.reshape(wspeed, u_shape)
    wdir = np.reshape(wdir, u_shape)

    return wspeed, wdir


def compute_divergence(
    u,
    v,
    lon,
    lat):

    """
    Computes convergence of a wind field (u,v) on a coordinate grid
    (lon, lat) on a height layer and for a certain time (u,v are both 2D arrays).
    The formula behind it is convergence = du/dx + dv/dy.

    Parameters:
    u : array of floats
        Wind vector in zonal direction (in m s-1).
    v : array of floats
        Wind vector in meridional direction (in m s-1).
    lon : array of floats
        Longitude grid points (in decimal degrees East). 1D array.
    lat : array of floats
        Latitude grid points (in decimal degrees North). 1D array.
    """

    from geopy import distance      # needed to compute dx and dy

    nx = len(lon)
    ny = len(lat)

    divergence = np.full((ny, nx), np.nan)
    for ii in range(ny):        # loop through row indices (latitudes)
        for jj in range(nx):    # loop through column indices (longitudes)

            # compute du, dv, dx, dy:
            if ii == 0:     # first row (highest latitude)
                dy = distance.distance((lat[ii+1], lon[jj]), (lat[ii], lon[jj])).km*1000.0
                dv = v[ii,jj] - v[ii+1,jj]

            elif ii < ny-1:
                # 2 steps for centered difference:
                dy = distance.distance((lat[ii-1], lon[jj]), (lat[ii+1], lon[jj])).km*1000.0
                dv = v[ii-1,jj] - v[ii+1,jj]

            else:   # last row (lowest latitude)
                dy = distance.distance((lat[ii-1], lon[jj]), (lat[ii], lon[jj])).km*1000.0
                dv = v[ii-1,jj] - v[ii,jj]

            if jj == 0: # western border
                du = u[ii,jj+1] - u[ii,jj]
                dx = distance.distance((lat[ii], lon[jj+1]), (lat[ii], lon[jj])).km*1000.0      # distance in meters

            elif jj < nx-1: # between western and eastern border
                du = u[ii,jj+1] - u[ii,jj-1]
                dx = distance.distance((lat[ii], lon[jj+1]), (lat[ii], lon[jj-1])).km*1000.0        # distance in meters

            else:   # eastern border
                dx = distance.distance((lat[ii], lon[jj-1]), (lat[ii], lon[jj])).km*1000.0
                du = u[ii,jj] - u[ii,jj-1]

            # compute convergece: du/dx + dv/dy
            if dx == 0 and dy != 0:
                divergence[ii,jj] = dv/dy
            elif dy == 0 and dx != 0:
                divergence[ii,jj] = du/dx
            elif dx == 0 and dy == 0:
                divergence[ii,jj] = 0.0
            else:
                divergence[ii,jj] = du/dx + dv/dy

    return divergence


def relative_vorticity_advection(
    u,
    v,
    lon,
    lat):

    """
    Computes relative vorticity advection according to -v*grad(rel_vorticity_z) where
    v is the 2D wind vector of (u, v) = (zonal, meridional) wind, rel_vorticity_z is 
    rot(v) z component on a (lon,lat) grid. The wind components are 2D arrays of the 
    shape (len(lat), len(lon)).

    Parameters:
    u : array of floats
        Wind vector in zonal direction (in m s-1).
    v : array of floats
        Wind vector in meridional direction (in m s-1).
    lon : array of floats
        Longitude grid points (in decimal degrees East). 1D array.
    lat : array of floats
        Latitude grid points (in decimal degrees North). 1D array.
    """

    from geopy import distance      # needed to compute dx and dy

    nx = len(lon)
    ny = len(lat)
    dx = np.zeros((ny,nx))
    dy = np.zeros((ny,nx))
    du = np.zeros((ny,nx))
    dv = np.zeros((ny,nx))
    du_dy = np.zeros((ny,nx))
    dv_dx = np.zeros((ny,nx))

    rva = np.full((ny, nx), np.nan)
    for ii in range(ny):        # loop through row indices (latitudes)
        for jj in range(nx):    # loop through column indices (longitudes)

            # compute du, dv, dx, dy:
            if ii == 0:     # first row (highest latitude)
                dy[ii,jj] = distance.distance((lat[ii+1], lon[jj]), (lat[ii], lon[jj])).km*1000.0
                dv[ii,jj] = v[ii,jj] - v[ii+1,jj]

            elif ii < ny-1:
                # 2 steps for centered difference:
                dy[ii,jj] = distance.distance((lat[ii-1], lon[jj]), (lat[ii+1], lon[jj])).km*1000.0
                dv[ii,jj] = v[ii-1,jj] - v[ii+1,jj]

            else:   # last row (lowest latitude)
                dy[ii,jj] = distance.distance((lat[ii-1], lon[jj]), (lat[ii], lon[jj])).km*1000.0
                dv[ii,jj] = v[ii-1,jj] - v[ii,jj]

            if jj == 0: # western border
                du[ii,jj] = u[ii,jj+1] - u[ii,jj]
                dx[ii,jj] = distance.distance((lat[ii], lon[jj+1]), (lat[ii], lon[jj])).km*1000.0       # distance in meters

            elif jj < nx-1: # between western and eastern border
                du[ii,jj] = u[ii,jj+1] - u[ii,jj-1]
                dx[ii,jj] = distance.distance((lat[ii], lon[jj+1]), (lat[ii], lon[jj-1])).km*1000.0     # distance in meters

            else:   # eastern border
                du[ii,jj] = u[ii,jj] - u[ii,jj-1]
                dx[ii,jj] = distance.distance((lat[ii], lon[jj-1]), (lat[ii], lon[jj])).km*1000.0


            # first, compute for all grid points: du/dy and dv/dx
            if dy[ii,jj] != 0:
                du_dy[ii,jj] = du[ii,jj]/dy[ii,jj]
            else:
                du_dy[ii,jj] = 0.0
            if dx[ii,jj] != 0:
                dv_dx[ii,jj] = dv[ii,jj]/dx[ii,jj]


    # now we need to know how du/dy and dv/dx change in x and y directions:
    for ii in range(ny):        # loop through row indices (latitudes)
        for jj in range(nx):    # loop through column indices (longitudes)

            # compute d(du_dy)/dx, d(du_dy)/dy, d(dv_dx)/dx, d(dv_dx)/dy
            if ii == 0:     # first row (highest latitude)
                if dy[ii,jj] == 0:      # capture division by zero before errors occur
                    ddu_dydy = 0.0
                    ddv_dxdy = 0.0
                else:
                    ddu_dydy = (du_dy[ii,jj] - du_dy[ii+1,jj])/dy[ii,jj]
                    ddv_dxdy = (dv_dx[ii,jj] - dv_dx[ii+1,jj])/dy[ii,jj]

            elif ii < ny-1:
                # 2 steps for centered difference:
                if dy[ii,jj] == 0:
                    ddu_dydy = 0.0
                    ddv_dxdy = 0.0
                else:
                    ddu_dydy = (du_dy[ii-1,jj] - du_dy[ii+1,jj])/dy[ii,jj]
                    ddv_dxdy = (dv_dx[ii-1,jj] - dv_dx[ii+1,jj])/dy[ii,jj]

            else:   # last row (lowest latitude)
                if dy[ii,jj] == 0:
                    ddu_dydy = 0.0
                    ddv_dxdy = 0.0
                else:
                    ddu_dydy = (du_dy[ii-1,jj] - du_dy[ii,jj])/dy[ii,jj]
                    ddv_dxdy = (dv_dx[ii-1,jj] - dv_dx[ii,jj])/dy[ii,jj]

            if jj == 0: # western border
                if dx[ii,jj] == 0:
                    ddu_dydx = 0.0
                    ddv_dxdx = 0.0
                else:
                    ddu_dydx = (du_dy[ii,jj+1] - du_dy[ii,jj])/dx[ii,jj]
                    ddv_dxdx = (dv_dx[ii,jj+1] - dv_dx[ii,jj])/dx[ii,jj]

            elif jj < nx-1: # between western and eastern border
                if dx[ii,jj] == 0:
                    ddu_dydx = 0.0
                    ddv_dxdx = 0.0
                else:
                    ddu_dydx = (du_dy[ii,jj+1] - du_dy[ii,jj-1])/dx[ii,jj]
                    ddv_dxdx = (dv_dx[ii,jj+1] - dv_dx[ii,jj-1])/dx[ii,jj]              

            else:   # eastern border
                if dx[ii,jj] == 0:
                    ddu_dydx = 0.0
                    ddv_dxdx = 0.0
                else:
                    ddu_dydx = (du_dy[ii,jj] - du_dy[ii,jj-1])/dx[ii,jj]
                    ddv_dxdx = (dv_dx[ii,jj] - dv_dx[ii,jj-1])/dx[ii,jj]

            # compute relative vorticity advection (rva):
            rva[ii,jj] = u[ii,jj]*(ddu_dydx - ddv_dxdx) + v[ii,jj]*(ddu_dydy - ddv_dxdy)

    return rva


def absolute_vorticity_advection(
    u,
    v,
    lon,
    lat):

    """
    Computes absolute vorticity advection (relative + planetary) according to 
    -v*grad(rel_vorticity_z + f) where v is the 2D wind vector of (u, v) = (zonal, meridional) wind, 
    rel_vorticity_z is rot(v) z component, f is Coriolis parameter on a (lon,lat) grid. The wind 
    components are 2D arrays of the shape (len(lat), len(lon)).

    Parameters:
    u : array of floats
        Wind vector in zonal direction (in m s-1).
    v : array of floats
        Wind vector in meridional direction (in m s-1).
    lon : array of floats
        Longitude grid points (in decimal degrees East). 1D array.
    lat : array of floats
        Latitude grid points (in decimal degrees North). 1D array.
    """

    from geopy import distance      # needed to compute dx and dy

    nx = len(lon)
    ny = len(lat)
    dx = np.zeros((ny,nx))
    dy = np.zeros((ny,nx))
    du = np.zeros((ny,nx))
    dv = np.zeros((ny,nx))
    du_dy = np.zeros((ny,nx))
    dv_dx = np.zeros((ny,nx))

    # compute coriolis parameter for each grid point:
    f = np.repeat(np.reshape(2*omega_earth*np.sin(np.radians(lat)), (ny,1)), nx, axis=1)

    ava = np.full((ny, nx), np.nan)
    for ii in range(ny):        # loop through row indices (latitudes)
        for jj in range(nx):    # loop through column indices (longitudes)

            # compute du, dv, dx, dy:
            if ii == 0:     # first row (highest latitude)
                dy[ii,jj] = distance.distance((lat[ii+1], lon[jj]), (lat[ii], lon[jj])).km*1000.0
                dv[ii,jj] = v[ii,jj] - v[ii+1,jj]

            elif ii < ny-1:
                # 2 steps for centered difference:
                dy[ii,jj] = distance.distance((lat[ii-1], lon[jj]), (lat[ii+1], lon[jj])).km*1000.0
                dv[ii,jj] = v[ii-1,jj] - v[ii+1,jj]

            else:   # last row (lowest latitude)
                dy[ii,jj] = distance.distance((lat[ii-1], lon[jj]), (lat[ii], lon[jj])).km*1000.0
                dv[ii,jj] = v[ii-1,jj] - v[ii,jj]


            if jj == 0: # western border
                du[ii,jj] = u[ii,jj+1] - u[ii,jj]
                dx[ii,jj] = distance.distance((lat[ii], lon[jj+1]), (lat[ii], lon[jj])).km*1000.0       # distance in meters

            elif jj < nx-1: # between western and eastern border
                du[ii,jj] = u[ii,jj+1] - u[ii,jj-1]
                dx[ii,jj] = distance.distance((lat[ii], lon[jj+1]), (lat[ii], lon[jj-1])).km*1000.0     # distance in meters

            else:   # eastern border
                du[ii,jj] = u[ii,jj] - u[ii,jj-1]
                dx[ii,jj] = distance.distance((lat[ii], lon[jj-1]), (lat[ii], lon[jj])).km*1000.0


            # first, compute for all grid points: du/dy and dv/dx
            if dy[ii,jj] != 0:
                du_dy[ii,jj] = du[ii,jj]/dy[ii,jj]
            else:
                du_dy[ii,jj] = 0.0
            if dx[ii,jj] != 0:
                dv_dx[ii,jj] = dv[ii,jj]/dx[ii,jj]


    # now we need to know how du/dy and dv/dx change in x and y directions:
    for ii in range(ny):        # loop through row indices (latitudes)
        for jj in range(nx):    # loop through column indices (longitudes)

            # compute d(du_dy)/dx, d(du_dy)/dy, d(dv_dx)/dx, d(dv_dx)/dy
            if ii == 0:     # first row (highest latitude)
                if dy[ii,jj] == 0:      # capture division by zero before errors occur
                    ddu_dydy = 0.0
                    ddv_dxdy = 0.0
                    df_dy = 0.0
                else:
                    ddu_dydy = (du_dy[ii,jj] - du_dy[ii+1,jj])/dy[ii,jj]
                    ddv_dxdy = (dv_dx[ii,jj] - dv_dx[ii+1,jj])/dy[ii,jj]
                    df_dy = (f[ii,jj] - f[ii+1,jj])/dy[ii,jj]

            elif ii < ny-1:
                # 2 steps for centered difference:
                if dy[ii,jj] == 0:
                    ddu_dydy = 0.0
                    ddv_dxdy = 0.0
                    df_dy = 0.0
                else:
                    ddu_dydy = (du_dy[ii-1,jj] - du_dy[ii+1,jj])/dy[ii,jj]
                    ddv_dxdy = (dv_dx[ii-1,jj] - dv_dx[ii+1,jj])/dy[ii,jj]
                    df_dy = (f[ii-1,jj] - f[ii+1,jj])/dy[ii,jj]

            else:   # last row (lowest latitude)
                if dy[ii,jj] == 0:
                    ddu_dydy = 0.0
                    ddv_dxdy = 0.0
                    df_dy = 0.0
                else:
                    ddu_dydy = (du_dy[ii-1,jj] - du_dy[ii,jj])/dy[ii,jj]
                    ddv_dxdy = (dv_dx[ii-1,jj] - dv_dx[ii,jj])/dy[ii,jj]
                    df_dy = (f[ii-1,jj] - f[ii,jj])/dy[ii,jj]


            if jj == 0: # western border
                if dx[ii,jj] == 0:
                    ddu_dydx = 0.0
                    ddv_dxdx = 0.0
                    df_dx = 0.0
                else:
                    ddu_dydx = (du_dy[ii,jj+1] - du_dy[ii,jj])/dx[ii,jj]
                    ddv_dxdx = (dv_dx[ii,jj+1] - dv_dx[ii,jj])/dx[ii,jj]
                    df_dx = (f[ii,jj+1] - f[ii,jj])/dx[ii,jj]

            elif jj < nx-1: # between western and eastern border
                if dx[ii,jj] == 0:
                    ddu_dydx = 0.0
                    ddv_dxdx = 0.0
                    df_dx = 0.0
                else:
                    ddu_dydx = (du_dy[ii,jj+1] - du_dy[ii,jj-1])/dx[ii,jj]
                    ddv_dxdx = (dv_dx[ii,jj+1] - dv_dx[ii,jj-1])/dx[ii,jj]
                    df_dx = (f[ii,jj+1] - f[ii,jj-1])/dx[ii,jj]

            else:   # eastern border
                if dx[ii,jj] == 0:
                    ddu_dydx = 0.0
                    ddv_dxdx = 0.0
                    df_dx = 0.0
                else:
                    ddu_dydx = (du_dy[ii,jj] - du_dy[ii,jj-1])/dx[ii,jj]
                    ddv_dxdx = (dv_dx[ii,jj] - dv_dx[ii,jj-1])/dx[ii,jj]
                    df_dx = (f[ii,jj] - f[ii,jj-1])/dx[ii,jj]

            # compute absolute vorticity advection (ava):
            ava[ii,jj] = -1.0*(u[ii,jj]*(ddv_dxdx - ddu_dydx + df_dx) + v[ii,jj]*(ddv_dxdy - ddu_dydy + df_dy))

    return ava


def potential_temperature(
    press,
    temp,
    press_sfc=100000.0,
    height_axis=None):

    """
    Computes potential temperature theta from pressure and temperature of a certain level, and
    surface pressure according to theta = T*(p_s/p)**(R/c_p).

    Parameters:
    temp : array of floats
        Temperature at a certain height level in K.
    press : array of floats
        Pressure (best in Pa, else: same units as press_sfc) at a certain height level. Shape can
        be equal to temp.shape but can also be a 1D array.
    press_sfc : float
        Surface or reference pressure (in same units as press, preferably in Pa or hPa). Usually
        100000 Pa. 
    height_axis : int or None
        Identifier to locate the height axis of temp (i.e., 0, 1 or 2).
    """

    if press.ndim == 1: # expand press to shape of temp
        n_press = len(press)
        
        if height_axis == None:
            raise ValueError("Please specify which is the height axis of the temperature data as integer.")

        else:
            # build new shape list
            press_shape_new = list()
            for k in range(temp.ndim): press_shape_new.append(1)
            press_shape_new[height_axis] = temp.shape[height_axis]
            press = np.reshape(press, press_shape_new)

            # repeat pressure values:
            for k, tt in enumerate(temp.shape):
                if k != height_axis:
                    press = np.repeat(press, tt, axis=k)

            # compute pot. temperature:
            theta = temp*(press_sfc/press)**(R_d/c_pd)

    elif press.shape == temp.shape:
        theta = temp*(press_sfc/press)**(R_d/c_pd)

    return theta


def e_sat(
    temp,
    which_algo='hyland_and_wexler'):

    """
    Calculates the saturation pressure over water after Goff and Gratch (1946)
    or Hyland and Wexler (1983).
    Source: Smithsonian Tables 1984, after Goff and Gratch 1946
    http://cires.colorado.edu/~voemel/vp.html
    http://hurri.kean.edu/~yoh/calculations/satvap/satvap.html

    e_sat_gg_water in Pa.

    Parameters:
    -----------
    temp : array of floats
        Array of temperature (in K).
    which_algo : str
        Specify which algorithm is chosen to compute e_sat (in Pa). Options:
        'hyland_and_wexler' (default), 'goff_and_gratch'
    """

    if which_algo == 'hyland_and_wexler':
        e_sat_gg_water = temp**(0.65459673e+01) * np.exp(-0.58002206e+04 / temp + 0.13914993e+01 - 0.48640239e-01*temp + 
                                0.41764768e-04*(temp**2) - 0.14452093e-07*(temp**3))

    elif which_algo == 'goff_and_gratch':
        e_sat_gg_water = 100 * 1013.246 * 10**(-7.90298*(373.16/temp-1) + 5.02808*np.log10(
                373.16/temp) - 1.3816e-7*(10**(11.344*(1-temp/373.16))-1) + 8.1328e-3 * (10**(-3.49149*(373.16/temp-1))-1))

    return e_sat_gg_water


def convert_rh_to_abshum(
    temp,
    relhum):

    """
    Convert array of relative humidity (between 0 and 1) to absolute humidity
    in kg m^-3. 

    Saturation water vapour pressure computation is based on: see e_sat(temp).

    Parameters:
    -----------
    temp : array of floats
        Array of temperature (in K).
    relhum : array of floats
        Array of relative humidity (between 0 and 1).
    """

    e_sat_water = e_sat(temp)

    rho_v = relhum * e_sat_water / (R_v * temp)

    return rho_v


def convert_rh_to_spechum(
    temp,
    pres,
    relhum):

    """
    Convert array of relative humidity (between 0 and 1) to specific humidity
    in kg kg^-1.

    Saturation water vapour pressure computation is based on: see e_sat(temp).

    Parameters:
    -----------
    temp : array of floats
        Array of temperature (in K).
    pres : array of floats
        Array of pressure (in Pa).
    relhum : array of floats
        Array of relative humidity (between 0 and 1).
    """

    e_sat_water = e_sat(temp)

    e = e_sat_water * relhum
    q = M_dv * e / (e*(M_dv - 1) + pres)

    return q
    
    
def convert_abshum_to_spechum(
    temp,
    pres,
    abshum):

    """
    Convert array of absolute humidity (kg m^-3) to specific humidity
    in kg kg^-1.

    Parameters:
    -----------
    temp : array of floats
        Array of temperature (in K).
    pres : array of floats
        Array of pressure (in Pa).
    abshum : array of floats
        Array of absolute humidity (in kg m^-3).
    """

    q = abshum / (abshum*(1 - 1/M_dv) + (pres/(R_d*temp)))

    return q


def convert_spechum_to_mix_rat(
    q,
    q_add=np.nan):

    """
    Convert array (of float) of specific humidity (kg kg-1) to water vapour 
    mixing ratio (in kg kg-1). Also other hydrometeors (cloud liquid, 
    cloud rain water, snow, ice) can be respected.

    Parameters:
    -----------
    q : float or array of floats
        Specific humidity in kg kg-1.
    q_add : float or array of floats
        Sum of other hydrometeors (i.e., cloud liquid, cloud ice, snow, rain) as
        'specific' contents (in kg kg-1). 
    """

    if ((type(q_add) == type(np.array([]))) and q_add.size == 0) or ((type(q_add) == float) and (np.isnan(q_add))):
        r_v = q / (1 - q)
    else:
        r_v = q / (1 - q - q_add)

    return r_v


def convert_mix_rat_to_spechum(
    r_v):

    """
    Convert array (of float) of water vapour mixing ratio (in kg kg-1) to specific humidity 
    (also in kg kg-1). 

    Parameters:
    -----------
    r_v : float or array of floats
        Water vapour mixing ratio in kg kg-1.
    """

    q = r_v / (1 + r_v)

    return q


def convert_relhum_to_mix_rat(
    relhum,
    temp,
    pres):

    """
    Convert relative humidity (in [0,1]) to water vapour mixing ratio (in kg kg-1).

    Parameters:
    -----------
    relhum : array of floats or float
        Array of relative humidity (between 0 and 1).
    temp : array of floats or float
        Array of temperature (in K).
    pres : array of floats or float
        Array of air pressure (in Pa).
    """

    # convert relhum to abshum:
    abshum = convert_rh_to_abshum(temp, relhum)
    r_v = abshum / ((pres - e_sat(temp)*relhum) / (R_d * temp))

    return r_v


def rho_air(
    pres,
    temp,
    abshum):

    """
    Compute the density of air (in kg m-3) with a certain moisture load.

    Parameters:
    -----------
    pres : array of floats
        Array of pressure (in Pa).
    temp : array of floats
        Array of temperature (in K).
    abshum : array of floats
        Array of absolute humidity (in kg m^-3).
    """

    rho = (pres - abshum*R_v*temp) / (R_d*temp) + abshum

    return rho


def convert_spechum_to_abshum(
    temp,
    pres,
    q):

    """
    Convert array of specific humidity (kg kg^-1) to absolute humidity
    in kg m^-3.

    Parameters:
    -----------
    temp : array of floats
        Array of temperature (in K).
    pres : array of floats
        Array of pressure (in Pa).
    q : array of floats
        Array of specific humidity (in kg kg^-1).
    """

    abshum = pres / (R_d*temp*(1/q + 1/M_dv - 1))

    return abshum


def convert_abshum_to_relhum(
    temp,
    abshum):

    """
    Convert array of absolute humidity (in kg m^-3) to relative humidity (in [0...1]).

    Parameters:
    -----------
    temp : array of floats
        Array of temperature (in K).
    abshum : array of floats
        Array of absolute humidity (in kg m^-3).
    """

    e = abshum*R_v*temp
    e_sat_water = e_sat(temp)
    relhum = e/e_sat_water

    return relhum


def convert_spechum_to_relhum(
    temp,
    pres,
    q):

    """
    Convert array of specific humidity (kg kg^-1) to relative humidity
    in [0...1].

    Parameters:
    -----------
    temp : array of floats
        Array of temperature (in K).
    pres : array of floats
        Array of pressure (in Pa).
    q : array of floats
        Array of specific humidity (in kg kg^-1).
    """

    e = pres / (M_dv*(1/q + 1/M_dv - 1))
    e_sat_water = e_sat(temp)
    relhum = e/e_sat_water

    return relhum


def q_to_h2ovmr(q: np.ndarray):
    
    """
    Converts specific humidity q (in kg kg-1) to volume mixing ratio (unitless).
    
    Parameters:
    -----------
    q : np.ndarray or xr.DataArray
        Specific humidity in kg kg-1.
    """
    
    return m_mol_air*q / (mw_h2o*(1.0 - q))


def h2ovmr_to_q(h2ovmr: np.ndarray):
    
    """
    Converts water vapour volume mixing ratio (unitless) to specific humidity q (in kg kg-1).
    
    Parameters:
    -----------
    h2ovmr : np.ndarray or xr.DataArray
        Water vapour volume mixing ratio.
    """
    
    return h2ovmr / ((m_mol_air/mw_h2o) + h2ovmr)


def virtual_temp(
    temp,
    q):

    """
    Compute the virtual temperature from specific humidity and temperature.

    Parameters:
    -----------
    temp : array of floats
        Array of temperature (in K).
    q : array of floats
        Array of specific humidity (in kg kg^-1).
    """

    temp_v = temp * (q/M_dv + (1 - q))

    return temp_v


def create_ICAO_std_atmosphere(z=None):
    
    """
    Create a standard temperature, pressure, height and specific humidity profile using the ICAO 
    standard atmosphere: https://ntrs.nasa.gov/api/citations/19770009539/downloads/19770009539.pdf ,
    pages 3,4,11,12.
    
    Parameters:
    -----------
    z : np.ndarray
        Optionally, a height grid z can be provided on which the temperature, pressure and specific
        humidity profiles are returned.
    """
    
    if z is None:
        z = np.arange(0., 11000.01, 10.)
    
    std_hgt = np.array([0.0, 11000.0])  # height boundaries of ICAO std atmosphere
    std_temp = np.array([15.0, -56.5])+273.15   # temperature of ICAO std atmosphere
    std_rh = np.array([0.75,0.0])       # assumed relative humidity to generate a default q profile
                                        # surface relative humidity used according to Peixoto and Oort (1996)
                                        # https://doi.org/10.1175/1520-0442(1996)009<3443:TCORHI>2.0.CO;2

    # interpolate standard atmosphere to current height grid:
    std_temp = np.interp(z, std_hgt, std_temp, left=np.nan, right=np.nan)
    std_rh = np.interp(z, std_hgt, std_rh, left=np.nan, right=np.nan)
    std_hgt = z

    L_isa = -6.5 / 1000.        # lapse rate in K m-1
    std_pres = (101325.0 * (std_temp[0] / (std_temp[0] + L_isa*(std_hgt - std_hgt[0])))**(
                g*m_mol_air / (R_*L_isa)))      # formula 33a in the reference given in the function description
    std_q = convert_rh_to_spechum(std_temp, std_pres, std_rh)
    std_q[np.isnan(std_q)] = 0.0
    
    return std_pres, std_hgt, std_temp, std_q


def equiv_pot_temperature(
    temp,
    pres,
    relhum=np.array([]),
    q=np.array([]),
    q_hyd=np.array([]),
    neglect_rtc=True):

    """
    Computes the equivalent potential temperature following 
    https://glossary.ametsoc.org/wiki/Equivalent_potential_temperature .
    The given air pressure must be reduced to partial pressure of dry air.
    temp, pres, relhum, q and q_hyd must have the same shape. Either relhum
    or q must be provided.

    Parameters:
    -----------
    temp : array of floats
        Temperature in K.
    pres : rray of floats
        Air pressure in Pa.
    relhum : array of floats
        Relative humidity in [0,1].
    q : array of floats
        Specific humidity in kg kg-1.
    q_hyd : array of floats
        Specific content of several hydrometeors (i.e., cloud liquid, ice, snow, rain)
        in kg kg-1. Can be neglected
    neglect_rtc : bool
        Option whether to neglect the terms r_t*c_h2o (setting r_t = 0) or not.
        According to https://glossary.ametsoc.org/wiki/Equivalent_potential_temperature
        both can be used with good accuracy.
    """

    if (relhum.size == 0) and (q.size == 0):
        raise ValueError("Specific or relative humidity must be provided.")
    elif q.size == 0:
        r_v = convert_relhum_to_mix_rat(relhum, temp, pres)
        e = e_sat(temp) * relhum
    else:
        r_v = convert_spechum_to_mix_rat(q, q_hyd)
        e = pres / (1 + M_dv*(1/q - 1))     # partial pressure of water vapour in Pa

    if q_hyd.size == 0:
        neglect_rtc = True

    pres_dry = pres - e                 # partial pressure of dry air in Pa

    # compute total water mixing ratio (vapour, liquid, ice, snow, rain) in kg kg-1
    if neglect_rtc:
        r_t = np.zeros(temp.shape)      # total water mixing ratio (vapour, liquid, ice, snow, rain)
    else:
        # convert q_hyd + q to r_t
        r_t = convert_spechum_to_mix_rat(q_hyd + q)

    cpd_rtc = c_pd + r_t*c_h2o
    theta_e = temp * (100000.0 / pres_dry)**(R_d / cpd_rtc) * relhum**(-r_v*R_v / cpd_rtc) * np.exp(L_v*r_v / (cpd_rtc*temp))

    return theta_e


def detect_hum_inversions_merged_20240424(
    q,
    z,
    is_radiosonde=False,
    delta_q=np.array([])):

    """
    Detect humidity inversions defined as increasing specific humidity q with altitude. The bottom
    is the height level where q starts increasing with height z. The top is the height level where
    q starts decreasing with height again. Humidity inversion strength is defined as q at top minus
    q at bottom. Depth is z at top - z at bottom. The humidity inversion detection is mainly based
    on Nygard et al. 2014 and Chellini et al. 2022, with additions from Naakka et al. 2018 and
    Devasthale et al. 2011. q and z must have the same shape (even if z doesn't vary in the other 
    dimensions).

    Parameters:
    -----------
    q : array of floats or xr.DataArray
        Array of specific humidity (in kg kg-1) with the shape (...,height). If DataArray, 
        the dimension name must be (..., 'z'). Bottom (top) of the profile must be at index 0 (-1).
    z : array of floats or xr.DataArray
        Array of height (in m) with shape (...,height). If DataArray, the dimension name must be 
        (..., 'z'). Bottom (top) of the profile must be at index 0 (-1). Shape must be identical to 
        q.shape.
    is_radiosonde : bool
        Boolean value indicating whether the q measurements are from radiosondes. If True, minimum
        inversion strength and depth requirements must be overcome to detect true inversions. 
        Otherwise, potential inversions will be mistaken for mere fluctuations.
    delta_q : array of floats
        If is_radiosonde is True, specific humidity error estimates delta_q (in kg kg-1) can be 
        given as input. The vertical maximum delta_q is then used as minimum inversion strength
        threshold.
    """

    inv_dict = dict()   # dictionary that will contain the humidity inversion characteristics
    z_b, z_t, q_b, q_t, inv_str, inv_str_rel, inv_dep, inv_sfc, n_inv = (np.array([]), np.array([]), np.array([]), 
                                            np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), 0)

    if q.shape != z.shape:
        print("z and q must have the same shape for humidity inversion detection. Aborted detect_hum_inversions....")
        return inv_dict


    # detection thresholds:
    min_dep = 100       # minimum q inv depth in m
    if is_radiosonde:
        if (len(delta_q) > 0) and (np.nanmax(delta_q, axis=-1) < 5e-04):
            min_str = np.nanmax(delta_q, axis=-1)   # min strength of q inversion (in kg kg-1) to filter fluctuations
        else:
            min_str = 5e-04
        min_rel_str = 0.05      # fraction of q at inv top that must be exceeded to be detected as q inversion
        q_reduction_fraction = 0.05 # fraction of q_t of an inv; q reduction to next inv must be smaller
    else:
        q_reduction_fraction = 0.05
        min_str = 0.0
        min_rel_str = 0.0


    # change height axis: remove last value:
    z = z[:-1]

    # find height levels where q increases with height and limit to altitudes below 7000 m:
    check = q[:-1] < q[1:]
    check[z >= 7000.] = False
    check = check.astype('int')     # int needed

    # inversion base and top heights can be detected via differences of the check mask with height:
    check_shifted = np.zeros_like(check)
    check_shifted[1:] = check[:-1]
    base = (check - check_shifted) == 1
    top = (check - check_shifted) == -1


    # merge consecutive inversions if they fulfill some conditions
    q_t = q[:-1][top]       # q at inversion top
    q_b = q[:-1][base]      # q at inversion base
    q_inv_diff = q_t[:-1] - q_b[1:] # difference of lower inversion top to upper inversion base


    # check conditions: Does q not decrease too much before the next inversion starts and is q_b
    # of next inv smaller than q_t of current inversion. If True, can be merged. Ignore index -1
    # because no inversion base will follow.
    merge_inv_check = (q_inv_diff <= (q_reduction_fraction*q_t[:-1])) & (q_inv_diff >= 0)

    # also make sure that q_t of lower inversion is greater than q_t of upper inversion. 
    merge_inv_check = (merge_inv_check & (-np.diff(q_t) > 0))

    # if two inversions should be connected:
    if np.any(merge_inv_check):
        base_idx = np.where(base)[0]
        top_idx = np.where(top)[0]

        # set base, top to False, if that inversion is to be merged: ignore lowest base and highest top.
        base[base_idx[1:][merge_inv_check]] = False
        top[top_idx[:-1][merge_inv_check]] = False

        # set check values between the top of the lower and the base of the upper of the merged inversions to True
        check_idx = np.array([], dtype='int32')
        for k, mic in enumerate(merge_inv_check):
            if mic:
                # top index of current inversion until base index of the next inversion:
                check_idx = np.concatenate((check_idx, range(top_idx[np.where(merge_inv_check)[0][0]], 
                                            base_idx[np.where(merge_inv_check)[0][0]+1]+1)))        # the final +1 is due to python indexing
                merge_inv_check[k] = 0      # because this merge_inv_check is updated, np.where(...)[0][0] is used above
        check[check_idx] = 1


    # compute inversion strength and delete inversions that are too weak:
    # identify single inversions and determine strength, depth, q_top, q_base, ...:
    idx_base = np.where(base)[0]
    idx_top = np.where(top)[0]

    z_b = z[idx_base]
    z_t = z[idx_top]
    q_b = q[idx_base]
    q_t = q[idx_top]
    inv_str = q_t - q_b
    inv_str_rel = inv_str / q_t
    inv_dep = z_t - z_b
    inv_sfc = z_b <= 50

    # check criteria to filter small fluctuations:
    where_no_fluctuation = np.where((inv_str >= min_rel_str*q_t) & (inv_dep >= min_dep) & (inv_str >= min_str))[0]
    inv_dict['z_b'] = z_b[where_no_fluctuation]         # inversion base height in m
    inv_dict['z_t'] = z_t[where_no_fluctuation]         # ... top height in m
    inv_dict['q_b'] = q_b[where_no_fluctuation]         # q at base of inversion in kg kg-1
    inv_dict['q_t'] = q_t[where_no_fluctuation]         # q at top of inversion in kg kg-1
    inv_dict['inv_str'] = inv_str[where_no_fluctuation] # inversion strength in kg kg-1
    inv_dict['inv_str_rel'] = inv_str_rel[where_no_fluctuation] # relative inversion strength in "1"
    inv_dict['inv_dep'] = inv_dep[where_no_fluctuation] # inversion depth in m
    inv_dict['inv_sfc'] = inv_sfc[where_no_fluctuation] # is it a surface-based inversion?
    inv_dict['n_inv'] = len(inv_dict['z_b'])

    return inv_dict


def detect_hum_inversions_merged(
    q,
    z,
    is_radiosonde=False,
    delta_q=np.array([])):

    """
    Detect humidity inversions defined as increasing specific humidity q with altitude. The bottom
    is the height level where q starts increasing with height z. The top is the height level where
    q starts decreasing with height again. Humidity inversion strength is defined as q at top minus
    q at bottom. Depth is z at top - z at bottom. The humidity inversion detection is mainly based
    on Nygard et al. 2014 and Chellini et al. 2022, with additions from Naakka et al. 2018 and
    Devasthale et al. 2011. q and z must have the same shape (even if z doesn't vary in the other 
    dimensions).

    Parameters:
    -----------
    q : array of floats or xr.DataArray
        Array of specific humidity (in kg kg-1) with the shape (...,height). If DataArray, 
        the dimension name must be (..., 'z'). Bottom (top) of the profile must be at index 0 (-1).
    z : array of floats or xr.DataArray
        Array of height (in m) with shape (...,height). If DataArray, the dimension name must be 
        (..., 'z'). Bottom (top) of the profile must be at index 0 (-1). Shape must be identical to 
        q.shape.
    is_radiosonde : bool
        Boolean value indicating whether the q measurements are from radiosondes. If True, minimum
        inversion strength and depth requirements must be overcome to detect true inversions. 
        Otherwise, potential inversions will be mistaken for mere fluctuations.
    delta_q : array of floats
        If is_radiosonde is True, specific humidity error estimates delta_q (in kg kg-1) can be 
        given as input. The vertical maximum delta_q is then used as minimum inversion strength
        threshold.
    """

    inv_dict = dict()   # dictionary that will contain the humidity inversion characteristics
    z_b, z_t, q_b, q_t, inv_str, inv_str_rel, inv_dep, inv_sfc, n_inv = (np.array([]), np.array([]), np.array([]), 
                                            np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), 0)

    if q.shape != z.shape:
        print("z and q must have the same shape for humidity inversion detection. Aborted detect_hum_inversions....")
        return inv_dict


    # detection thresholds:
    min_dep = 100       # minimum q inv depth in m
    if is_radiosonde:
        if (len(delta_q) > 0) and (np.nanmax(delta_q, axis=-1) < 5e-04):
            min_str = np.nanmax(delta_q, axis=-1)   # min strength of q inversion (in kg kg-1) to filter fluctuations
        else:
            min_str = 5e-04     # max threshold of min strength
        min_rel_str = 0.05      # fraction of q at inv top that must be exceeded to be detected as q inversion
        q_reduction_fraction = 0.05 # fraction of q_t of an inv; q reduction to next inv must be smaller
    else:
        q_reduction_fraction = 0.05
        min_str = 0.0
        min_rel_str = 0.0


    # change height axis: remove last value:
    z = z[:-1]

    # find height levels where q increases with height and limit to altitudes below 7000 m:
    check = q[:-1] < q[1:]
    check[z >= 7000.] = False
    check = check.astype('int')     # int needed
    idx_max = np.where(z>=7000.)[0][0]      # highest permitted height index

    # inversion base and top heights can be detected via differences of the check mask with height:
    check_shifted = np.zeros_like(check)
    check_shifted[1:] = check[:-1]
    base = (check - check_shifted) == 1
    top = (check - check_shifted) == -1


    # loop over detected inversions:
    n_inv_temp = np.count_nonzero(base)
    where_base = np.where(base)[0]
    if n_inv_temp != np.count_nonzero(top): pdb.set_trace() ## debug
    k = 0
    for inv_k in range(n_inv_temp):
        # because for some reason, python keeps iterating when updating n_inv_temp below so that
        # k > n_inv_temp is possible within the loop, catch that unintended behaviour:
        if k >= n_inv_temp: break

        # find an extended top to the current inversion base: where is q lower than q at base for the 
        # first time:
        q_b_cur = q[where_base[k]]
        idx_top_temp = np.where(q[where_base[k]+1:] <= q_b_cur)[0][0] + where_base[k] + 1
        if idx_top_temp > idx_max: idx_top_temp = idx_max + 1       # + 1 because of python indexing


        # check if that extended inversion is deep enough: If true, check if it can be merged
        # with another one:
        z_b_temp = z[where_base[k]]
        z_t_temp = z[idx_top_temp-1]
        if np.abs(z_t_temp - z_b_temp) < 2*min_dep: # twice min_dep because here, top is extended
            check[where_base[k]:idx_top_temp] = 0
            base[where_base[k]:idx_top_temp] = 0
            top[where_base[k]:idx_top_temp] = 0
            n_sign_changes = 0

        else:
            # check if dq changes sign more than once within this extended inversion. This can be used
            # to identify smaller inversions that could be merged into a larger one.
            dq_inv = np.sign(np.diff(q[where_base[k]:idx_top_temp]))
            dq_inv[dq_inv == -1.] = 0.
            n_sign_changes = np.count_nonzero(np.diff(dq_inv))


        if n_sign_changes > 1:  # then merge smaller inversions into larger one
            idx_bases_in_inv = np.where(base[where_base[k]:idx_top_temp])[0] + where_base[k]
            idx_tops_in_inv = np.where(top[where_base[k]:idx_top_temp])[0] + where_base[k]
            q_bases = q[idx_bases_in_inv]
            q_tops = q[idx_tops_in_inv]

            # compute strength, depth and check if thresholds are exceeded:
            inv_str_temp = q_tops - q_bases

            # discard smaller inversions 1) if no inversion follows the weak one until idx_top_temp or
            # 2) if weak one is above a strong one. However, if weak inversion is followed by strong
            # one or if strong one is followed by strong one, merge both.
            str_enough = inv_str_temp >= min_str
            len_sm_inv = len(str_enough)        # number of inv within where_base[k]:idx_top_temp
            n_merged = 0    # merged inversion counter -> to skip next bases when updating k
            for kk in range(len_sm_inv):
                if (kk < (len_sm_inv-1)) and ((~str_enough[kk]) & str_enough[kk+1]):
                    # check if dip from current inv top to next inv base is small enough:
                    merge_inv_check = np.abs(q_tops[kk] - q_bases[kk+1]) < q_reduction_fraction*q_tops[kk]

                    # set top of current (weak) inv to top of next (strong) inv; only use the base
                    # of the lowewr (weak) inversion for the merged inversion
                    if merge_inv_check:
                        top[idx_tops_in_inv[kk]] = 0
                        base[idx_bases_in_inv[kk+1]] = 0
                        check[idx_bases_in_inv[kk]:idx_tops_in_inv[kk+1]] = 1
                        n_merged += 1
                    else:       # discard weak inversion
                        base[idx_bases_in_inv[kk]] = 0
                        top[idx_tops_in_inv[kk]] = 0
                        check[idx_bases_in_inv[kk]:idx_tops_in_inv[kk]] = 0

                elif not str_enough[kk]: # discard weak inversion
                    base[idx_bases_in_inv[kk]] = 0
                    top[idx_tops_in_inv[kk]] = 0
                    check[idx_bases_in_inv[kk]:idx_tops_in_inv[kk]] = 0


            # update where_base and n_inv_temp:
            where_base = np.where(base)[0]
            n_inv_temp = np.count_nonzero(base)

            # if there are still some inversions left to check: skip to the next base that was
            # not yet considered in the merging process:
            if np.any(where_base >= idx_top_temp):
                k = np.where(where_base >= idx_top_temp)[0][0]
            else:
                k += 1 + n_merged

        else:
            k += 1


    # merge consecutive inversions if they fulfill some conditions
    q_t = q[:-1][top]       # q at inversion top
    q_b = q[:-1][base]      # q at inversion base
    q_inv_diff = q_t[:-1] - q_b[1:] # difference of lower inversion top to upper inversion base


    # check conditions: Does q not decrease too much before the next inversion starts and is q_b
    # of next inv smaller than q_t of current inversion. If True, can be merged. Ignore index -1
    # because no inversion base will follow.
    merge_inv_check = (q_inv_diff <= (q_reduction_fraction*q_t[:-1])) & (q_inv_diff >= 0)

    # if two inversions should be connected:
    if np.any(merge_inv_check):
        base_idx = np.where(base)[0]
        top_idx = np.where(top)[0]

        # set base, top to False, if that inversion is to be merged: ignore lowest base and highest top.
        base[base_idx[1:][merge_inv_check]] = False
        top[top_idx[:-1][merge_inv_check]] = False

        # set check values between the top of the lower and the base of the upper of the merged inversions to True
        check_idx = np.array([], dtype='int32')
        for k, mic in enumerate(merge_inv_check):
            if mic:
                # top index of current inversion until base index of the next inversion:
                check_idx = np.concatenate((check_idx, range(top_idx[np.where(merge_inv_check)[0][0]], 
                                            base_idx[np.where(merge_inv_check)[0][0]+1]+1)))        # the final +1 is due to python indexing
                merge_inv_check[k] = 0      # because this merge_inv_check is updated, np.where(...)[0][0] is used above
        check[check_idx] = 1


    # compute inversion strength and delete inversions that are too weak:
    # identify single inversions and determine strength, depth, q_top, q_base, ...:
    idx_base = np.where(base)[0]
    idx_top = np.where(top)[0]

    z_b = z[idx_base]
    z_t = z[idx_top]
    q_b = q[idx_base]
    q_t = q[idx_top]
    inv_str = q_t - q_b
    inv_str_rel = inv_str / q_t
    inv_dep = z_t - z_b
    inv_sfc = z_b <= 50

    # check criteria to filter small fluctuations:
    where_no_fluctuation = np.where(((inv_dep < min_dep) & (inv_str >= min_str*2) & (inv_str >= min_rel_str*q_t*2)) | 
                                    ((inv_dep >= min_dep) & (inv_str >= min_str) & (inv_str >= min_rel_str*q_t)))[0]
    inv_dict['z_b'] = z_b[where_no_fluctuation]         # inversion base height in m
    inv_dict['z_t'] = z_t[where_no_fluctuation]         # ... top height in m
    inv_dict['q_b'] = q_b[where_no_fluctuation]         # q at base of inversion in kg kg-1
    inv_dict['q_t'] = q_t[where_no_fluctuation]         # q at top of inversion in kg kg-1
    inv_dict['inv_str'] = inv_str[where_no_fluctuation] # inversion strength in kg kg-1
    inv_dict['inv_str_rel'] = inv_str_rel[where_no_fluctuation] # relative inversion strength in "1"
    inv_dict['inv_dep'] = inv_dep[where_no_fluctuation] # inversion depth in m
    inv_dict['inv_sfc'] = inv_sfc[where_no_fluctuation] # is it a surface-based inversion?
    inv_dict['n_inv'] = len(inv_dict['z_b'])

    return inv_dict


def detect_hum_inversions_bulk(
    q,
    z,
    is_radiosonde=False,
    delta_q=np.array([])):

    """
    Detect humidity inversions defined as increasing specific humidity q with altitude. The bottom
    is the height level where q starts increasing with height z. The top is the height level where
    q starts decreasing with height again. Humidity inversion strength is defined as q at top minus
    q at bottom. Depth is z at top - z at bottom. Algorithm: Find lowest index where q increases
    with height (inversion base) and the first height where q at inversion base is undershot (this 
    is then the 'extended inversion top'). The actual inversion top is then the height within 
    base - extended inversion top where q is maximum. Depth, strength and inversion top are then
    computed using this actual inversion top. Then, the search for further inversions starts at
    the extended inversion top. q and z must have the same shape (even if z doesn't vary in the 
    other dimensions).

    Parameters:
    -----------
    q : array of floats or xr.DataArray
        Array of specific humidity (in kg kg-1) with the shape (...,height). If DataArray, 
        the dimension name must be (..., 'z'). Bottom (top) of the profile must be at index 0 (-1).
    z : array of floats or xr.DataArray
        Array of height (in m) with shape (...,height). If DataArray, the dimension name must be 
        (..., 'z'). Bottom (top) of the profile must be at index 0 (-1). Shape must be identical to 
        q.shape.
    is_radiosonde : bool
        Boolean value indicating whether the q measurements are from radiosondes. If True, minimum
        inversion strength and depth requirements must be overcome to detect true inversions. 
        Otherwise, potential inversions will be mistaken for mere fluctuations.
    delta_q : array of floats
        If is_radiosonde is True, specific humidity error estimates delta_q (in kg kg-1) can be 
        given as input. The vertical maximum delta_q is then used as minimum inversion strength
        threshold.
    """

    inv_dict = dict()   # dictionary that will contain the humidity inversion characteristics
    z_b, z_t, q_b, q_t, inv_str, inv_str_rel, inv_dep, inv_sfc, n_inv = (np.array([]), np.array([]), np.array([]), 
                                            np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), 0)

    if q.shape != z.shape:
        print("z and q must have the same shape for humidity inversion detection. Aborted detect_hum_inversions....")
        return inv_dict


    # detection thresholds:
    min_dep = 100       # minimum q inv depth in m
    d_q_check = False   # initialized boolean to check if delta_q data is available for min_str thresholds
    if is_radiosonde:
        d_q_check = (len(delta_q) > 0 and (np.nanmax(delta_q, axis=-1) < 5e-04))
        if d_q_check:
            min_str = np.nanmax(delta_q, axis=-1)   # min strength of q inversion (in kg kg-1) to filter fluctuations
        else:
            min_str = 5e-04     # max threshold of min strength
        min_rel_str = 0.05      # fraction of q at inv top that must be exceeded to be detected as q inversion
    else:
        min_str = 5e-05
        min_rel_str = 0.05


    # change height axis: remove last value:
    z_o = copy.deepcopy(z)      # original z
    z = z[:-1]
    n_hgt = len(z)

    # find height levels where q increases with height and limit to altitudes below 7000 m:
    check = q[:-1] < q[1:]
    check[z >= 7000.] = False
    check = check.astype('int')     # int needed
    idx_max = np.where(z>=7000.)[0][0]      # highest permitted height index

    # inversion base and top heights can be detected via differences of the check mask with height:
    check_shifted = np.zeros_like(check)
    check_shifted[1:] = check[:-1]
    base = (check - check_shifted) == 1
    top = (check - check_shifted) == -1


    # starting at the lowest base (z_b), find height where q_b is undershot for the first time again.
    where_base = np.where(base)[0]
    if len(where_base) > 0: # otherwise, no inversions were found:
        i_b = where_base[0]                 # height index showing the current base
    else:
        i_b = idx_max

    # loop over potential inversion bases:
    idx_dict = {'base': [], 'top': [], 'top_ex': []}    # here, indices of inversions are saved to
    while i_b < idx_max:
        # pdb.set_trace()
        q_bt = q[i_b]
        z_bt = z_o[i_b]

        # find temporary top where q_bt is undershot the next time:
        where_tt = np.where(q[i_b+1:] <= q_bt)[0]
        if len(where_tt) > 0:
            where_tt = where_tt[0] + i_b + 1    # idx where q_bt is undershot
        else:   # q_bt is no longer undershot
            i_b = idx_max
            continue


        z_tt = z_o[where_tt]        # temporary top (extended top) of the inversion

        # check if inversion is deep enough (doubled threshold because z_tt is above the defined 
        # top of an inversion):
        is_deep = (z_tt - z_bt) >= min_dep*2
        if is_deep:
            # find max q within that layer <-> actual top of this inversion:
            q_t = np.nanmax(q[i_b:where_tt], axis=-1)
            where_max = np.nanargmax(q[i_b:where_tt], axis=-1) + i_b
            z_t = z_o[where_max]
            inv_str = q_t - q_bt
            inv_dep = z_t - z_bt

            # # # # # # # # # # # # # # use a flexible min_str threshold if radiosonde and if delta_q is available:
            # # # # # # # # # # # # # if d_q_check:
                # # # # # # # # # # # # # min_str = delta_q[where_max]
                # # # # # # # # # # # # # if min_str < 5e-05: min_str = 5e-05       # absolute min strength requirement

            # check if strength and depth requirements are met:
            no_fluctuation = (((inv_dep < min_dep) & (inv_str >= min_str*2) & (inv_str >= min_rel_str*q_t*2)) | 
                                    ((inv_dep >= min_dep) & (inv_str >= min_str) & (inv_str >= min_rel_str*q_t)))
            if no_fluctuation:
                # save indices of current inversion:
                idx_dict['base'] += [i_b]
                idx_dict['top'] += [where_max]
                idx_dict['top_ex'] += [where_tt]


        # move to the next inversion base that follows after the current inversion/fluctuation's
        # top:
        i_b = where_base[where_base >= where_tt]
        if len(i_b) > 0: 
            i_b = i_b[0]
        else:
            i_b = idx_max


    # compute inversion strength, depth, q_top, q_base, ...:
    z_b = z_o[idx_dict['base']]
    z_t = z_o[idx_dict['top']]
    q_b = q[idx_dict['base']]
    q_t = q[idx_dict['top']]
    z_t_ex = z_o[idx_dict['top_ex']]
    inv_str = q_t - q_b
    inv_str_rel = inv_str / q_t
    inv_dep = z_t - z_b
    inv_dep_ex = z_t_ex - z_b
    inv_sfc = z_b <= 50

    inv_dict['z_b'] = z_b                   # inversion base height in m
    inv_dict['z_t'] = z_t                   # ... top height in m
    inv_dict['q_b'] = q_b                   # q at base of inversion in kg kg-1
    inv_dict['q_t'] = q_t                   # q at top of inversion in kg kg-1
    inv_dict['z_t_ex'] = z_t_ex             # extended inversion height in m
    inv_dict['inv_str'] = inv_str           # inversion strength in kg kg-1
    inv_dict['inv_str_rel'] = inv_str_rel   # relative inversion strength in "1"
    inv_dict['inv_dep'] = inv_dep           # inversion depth in m
    inv_dict['inv_dep_ex'] = inv_dep_ex     # extended inversion depth in m
    inv_dict['inv_sfc'] = inv_sfc           # is it a surface-based inversion?
    inv_dict['n_inv'] = len(inv_dict['z_b'])    # number of detected inversions

    return inv_dict


def limit_to_troposphere(z: np.ndarray, q: np.ndarray, pres: np.ndarray, q_p: np.ndarray):
    
    in_troposphere = z <= 11000.
    q_tropo = q[in_troposphere]
    z_tropo = z[in_troposphere]
    pres_tropo = pres[in_troposphere]
    q_p_tropo = q_p[in_troposphere]
    
    return in_troposphere, q_tropo, z_tropo, pres_tropo, q_p_tropo


def modify_inv_str(
    x,
    q,
    z,
    q_b,
    z_b,
    z_t_ex,
    pres=None,
    conserve_IWV=True,
    max_it_neg_q=5):

    """
    Modify strength of a specific humidity inversion, whose base humidity is q_b, by a factor x
    (for example, x=0.4 reduces inversion strength from 1 g kg-1 to 0.4 g kg-1). The specific 
    humidity profile q will be canged between the base and the 'extended' height of the inversion
    (z_b and z_t_ex). The q profile changes are not uniform with height, but depend on the 
    difference q - q_b to allow for smooth transitions below z_b and above z_t_ex. It can be 
    chosen whether IWV should be conserved or not.

    Parameters:
    -----------
    x : float
        Inversion strength reduction factor. The new inversion strength is a fraction of the 
        original strength: inv_str' = inv_str * x . Values of x should be in [0,1], but values
        above 1 probably also work.
    q : array of floats
        Specific humidity profile (in kg kg-1) as 1D-array of shape (height,) that will be 
        modified. 
    z : array of floats
        Height profile (in m) with the same shape as q (height,).
    q_b : float or array of floats
        Specific humidity at inversion base (in kg kg-1). If array of floats, then q_b is of shape
        (n_inv,) where n_inv is the number of inversions.
    z_b : float or array of floats
        Height of the humidity inversion base (in m). If array of floats, then z_b is of shape
        (n_inv,) where n_inv is the number of inversions.
    z_t_ex : float or array of floats
        Height of the 'extended' inversion top (in m). It's the height where q = q_b again. If 
        array of floats, then z_t_ex is of shape (n_inv,) where n_inv is the number of inversions.
    pres : array of floats
        One dimensional array of pressure in Pa. Needed if conserve_IWV=True
    conserve_IWV : bool
        Boolean indicating whether IWV of the q profile should be conserved. If True, IWV will
        be conserved.
    max_it_neg_q : int
        Integer indicating the maximum number of iterations for the procedure to eliminate negative
        q during IWV conservation before getting suspicious. Default: 5
    """

    # first, use a copy of the original q profile as q_perturbed (q_p):
    q_p = copy.deepcopy(q)

    # convert input to arrays if only float:
    if (z_b.ndim == 0) and (z_t_ex.ndim == 0) and (q_b.ndim == 0):
        z_b = np.array([float(z_b)])
        z_t_ex = np.array([float(z_t_ex)])
        q_b = np.array([float(q_b)])

    # find altitude indices that need to be modified: Loop over inversions:
    for zb, ztex, qb in zip(z_b, z_t_ex, q_b):
        idx_mod = np.where((z >= zb) & (z <= ztex))[0]
        if len(idx_mod) == 0: continue

        # modify q:
        q_p[...,idx_mod] = qb + (q[...,idx_mod] - qb)*x     # see notes, p. 195 for algorithm


    # if desired, modify q further to conserve IWV: compute IWV of q_p before and after
    # perturbation and distribute the lost (or gained) IWV fraction to the entire q profile. 
    # Distribute the water vapour in an exponential form: more at the surface, less at the top.
    if conserve_IWV:
        q_p = conserve_IWV_factorial(q, q_p, pres, z, max_it_neg_q=max_it_neg_q)
        
    return q_p
    

def shift_inv_zbase(
    dz_b,
    q,
    z,
    q_b,
    z_b,
    z_t_ex,
    pres=None,
    conserve_IWV=True,
    max_it_neg_q=5):

    """
    Shift base height of specific humidity inversion whose base humidity is q_b, by a defined 
    value dz_b. The height grid z must have equidistant spacing to ensure that the inversion is
    homogeneously shifted (otherwise, the top may be shifted too much/less). The humidity 
    inversion will be shifted between the base and the 'extended' height of the inversion
    (z_b and z_t_ex). At the base of the original q profile, q=q_b for the heights z_b to
    z_b+dz_b. At the top of the modified profile, a transition zone to the old profile is created.
    It can be chosen whether IWV should be conserved or not.

    Parameters:
    -----------
    dz_b : float
        Shift inversion base height by this value (in m).
    q : array of floats
        Specific humidity profile (in kg kg-1) as 1D-array of shape (height,) that will be 
        modified. 
    z : array of floats
        Height profile (in m) with the same shape as q (height,).
    q_b : float or array of floats
        Specific humidity at inversion base (in kg kg-1). If array of floats, then q_b is of shape
        (n_inv,) where n_inv is the number of inversions.
    z_b : float or array of floats
        Height of the humidity inversion base (in m). If array of floats, then z_b is of shape
        (n_inv,) where n_inv is the number of inversions.
    z_t_ex : float or array of floats
        Height of the 'extended' inversion top (in m). It's the height where q = q_b again. If 
        array of floats, then z_t_ex is of shape (n_inv,) where n_inv is the number of inversions.
    pres : array of floats
        One dimensional array of pressure in Pa. Needed if conserve_IWV=True
    conserve_IWV : bool
        Boolean indicating whether IWV of the q profile should be conserved. If True, IWV will
        be conserved.
    max_it_neg_q : int
        Integer indicating the maximum number of iterations for the procedure to eliminate negative
        q during IWV conservation before getting suspicious. Default: 5
    """

    # first, use a copy of the original q profile as q_perturbed (q_p):
    q_p = copy.deepcopy(q)

    # convert input to arrays if only float:
    if (z_b.ndim == 0) and (z_t_ex.ndim == 0) and (q_b.ndim == 0):
        z_b = np.array([float(z_b)])
        z_t_ex = np.array([float(z_t_ex)])
        q_b = np.array([float(q_b)])
        

    # loop over inversions: start with the highest inversion:
    for zb, ztex, qb in zip(z_b[::-1], z_t_ex[::-1], q_b[::-1]):
        if ~np.isnan(zb+ztex+qb):

            # find height indices of the original base and the shifted base:
            idx_zb = np.where(z == zb)[0]
            if len(idx_zb) > 0: 
                idx_zb = idx_zb[0]
            else:   # not finding the base height in the height grid is unexpected. Check if the height
                    # grid here equals the one used in find_hum_inv.py
                pdb.set_trace()

            idx_zb_shift = np.where(z >= (zb + dz_b))[0]
            if len(idx_zb_shift) > 0: 
                idx_zb_shift = idx_zb_shift[0]
            else:
                pdb.set_trace()     # should not occur
            idx_diff = idx_zb_shift - idx_zb

            # idx of extended inversion top of original profile:
            idx_ztex = np.where(z == ztex)[0]
            if len(idx_ztex) > 0:
                idx_ztex = idx_ztex[0]
            else:
                pdb.set_trace()


            # shift inversion and adapt the perturbed profile from old to new base: q=q_b:
            q_p[idx_zb_shift:idx_ztex+idx_diff+1] = q[idx_zb:idx_ztex+1]
            q_p[idx_zb:idx_zb_shift] = qb


            # if the base shift value doesn't equal the value of an unperturbed profile (=0), 
            # modify perturbed profile at the top of the extended profile: linear transition from 
            # perturbed to original profile over a certain transition region of depth dz_tr.
            if not np.isclose(dz_b, 0.0, atol=1e+00):
                dz_tr = 250     # depth of transition zone in m
                idx_tr = np.where((z >= z[idx_ztex+idx_diff]) & (z <= (z[idx_ztex+idx_diff] + dz_tr)))[0]

                # check if q of original profile at the top of the transition zone is greater than that
                # of q_p. If True, use q=const as transition zone until q original is undershot again.
                if q[idx_tr[-1]] > q_p[idx_ztex+idx_diff] and (z[idx_ztex+idx_diff] < ztex):
                    q_transition = q_p[idx_ztex+idx_diff]       # use this as constant value
                    idx_tr_end = np.where(q[idx_ztex+idx_diff:] <= q_transition)[0] + idx_ztex+idx_diff
                    if len(idx_tr_end) > 0:
                        idx_tr_end = idx_tr_end[0]      # end of new transition zone
                    q_p[idx_tr[0]:idx_tr_end] = q_transition

                elif idx_diff != 0:
                    # if idx_diff == 0, which can occur if inversion is not shifted (e.g., negative dz_b at the surface)
                    # no further q variation needs to be done, but if idx_diff != 0, the transition zone must be
                    # used
                    q_transition = ((q[idx_tr[-1]] - q_p[idx_ztex+idx_diff])*(z[idx_tr] - z[idx_ztex+idx_diff]) / 
                                        (dz_tr) + q_p[idx_ztex+idx_diff])
                    q_p[idx_tr] = q_transition


    # if desired, modify q further to conserve IWV: compute IWV of q_p before and after
    # perturbation and distribute the lost (or gained) IWV fraction to the entire q profile. 
    # Distribute the water vapour in an exponential form: more at the surface, less at the top.
    if conserve_IWV:
        q_p = conserve_IWV_factorial(q, q_p, pres, z, max_it_neg_q=max_it_neg_q)

    return q_p


def modify_inv_depth(
    bloat,
    q,
    z,
    q_b,
    z_b,
    z_t_ex,
    pres=None,
    conserve_IWV=True,
    max_it_neg_q=5):

    """
    Modify depth of inversion (height difference between base z_b and extended top z_t_ex) 
    by a factor 'bloat'. A high resolution height grid is beneficial to increase accuracy
    of the actual increased inversion depth and the desired bloat value. The height grid
    resolution should also be constant with height. The base of the original q profile 
    remains unchanged in the perturbed profile, but z_t_ex of the perturbed profile will be 
    approximately z_b + bloat*(z_t_ex - z_b). At the top of the modified  profile, a 
    transition zone to the old profile is created. It can be chosen whether IWV should be 
    conserved or not.

    Parameters:
    -----------
    bloat : float
        Factor by which the height difference between inversion base (z_b) and the extended 
        top (z_t_ex) will be increased / decreased.
    q : array of floats
        Specific humidity profile (in kg kg-1) as 1D-array of shape (height,) that will be 
        modified. 
    z : array of floats
        Height profile (in m) with the same shape as q (height,).
    q_b : float or array of floats
        Specific humidity at inversion base (in kg kg-1). If array of floats, then q_b is of shape
        (n_inv,) where n_inv is the number of inversions.
    z_b : float or array of floats
        Height of the humidity inversion base (in m). If array of floats, then z_b is of shape
        (n_inv,) where n_inv is the number of inversions.
    z_t_ex : float or array of floats
        Height of the 'extended' inversion top (in m). It's the height where q = q_b again. If 
        array of floats, then z_t_ex is of shape (n_inv,) where n_inv is the number of inversions.
    pres : array of floats
        One dimensional array of pressure in Pa. Needed if conserve_IWV=True
    conserve_IWV : bool
        Boolean indicating whether IWV of the q profile should be conserved. If True, IWV will
        be conserved.
    max_it_neg_q : int
        Integer indicating the maximum number of iterations for the procedure to eliminate negative
        q during IWV conservation before getting suspicious. Default: 5
    """

    # first, use a copy of the original q profile as q_perturbed (q_p):
    q_p = copy.deepcopy(q)

    # convert input to arrays if only float:
    if (z_b.ndim == 0) and (z_t_ex.ndim == 0) and (q_b.ndim == 0):
        z_b = np.array([float(z_b)])
        z_t_ex = np.array([float(z_t_ex)])
        q_b = np.array([float(q_b)])

    
    # loop over inversions: start with the highest inversion:
    for zb, ztex, qb in zip(z_b[::-1], z_t_ex[::-1], q_b[::-1]):
        if ~np.isnan(zb+ztex+qb):
            idx_inv = np.where((z >= zb) & (z <= ztex))[0]      # height indices of current inv
            z_inv_p = bloat*z[idx_inv] + zb*(1 - bloat)

            idx_inv_p = np.where((z >= z_inv_p[0]) & (z <= z_inv_p[-1]))[0]

            # interpolate the original q profile from the current inversion's height
            # grid to the same height, but the number of entries must be changed according
            # to the perturbed height profile of the inversion
            q_p_inv = np.interp(np.linspace(zb, ztex, len(idx_inv_p)),  # unperturbed height grid of inv
                                np.linspace(zb, ztex, len(idx_inv)),    # same height grid, but more indices
                                q[idx_inv])

            # q_p_inv can then be used to replace q_p from z_b to the bloated z_t_ex of the current
            # inversion.
            q_p[idx_inv_p] = q_p_inv


            # if the bloat value doesn't equal the value of an unperturbed profile (=1), 
            # modify perturbed profile at the top of the extended profile: linear transition from 
            # perturbed to original profile over a certain transition region of depth dz_tr.
            if not np.isclose(bloat, 1.0, atol=1e-03):
                dz_tr = 250     # depth of transition zone in m
                idx_tr = np.where((z >= z[idx_inv_p[-1]]) & (z <= (z[idx_inv_p[-1]] + dz_tr)))[0]

                # check if q of original profile at the top of the transition zone is greater than that
                # of q_p. If True, use q=const as transition zone until q original is undershot again.
                if (q[idx_tr[-1]] > q_p[idx_inv_p[-1]]) and (z[idx_inv_p[-1]] < ztex):
                    q_transition = q_p[idx_inv_p[-1]]       # use this as constant value
                    idx_tr_end = np.where(q[idx_inv_p[-1]:] <= q_transition)[0] + idx_inv_p[-1]
                    if len(idx_tr_end) > 0:
                        idx_tr_end = idx_tr_end[0]      # end of new transition zone
                    q_p[idx_tr[0]:idx_tr_end] = q_transition

                else:
                    q_transition = ((q[idx_tr[-1]] - q_p[idx_inv_p[-1]])*(z[idx_tr] - z[idx_inv_p[-1]]) / 
                                        (dz_tr) + q_p[idx_inv_p[-1]])
                    q_p[idx_tr] = q_transition


    # if desired, modify q further to conserve IWV: compute IWV of q_p before and after
    # perturbation and distribute the lost (or gained) IWV fraction to the entire q profile. 
    # Distribute the water vapour in an exponential form: more at the surface, less at the top.
    if conserve_IWV:
        q_p = conserve_IWV_factorial(q, q_p, pres, z, max_it_neg_q=max_it_neg_q)

    return q_p


def q_p_to_conserve_IWV(
    q,
    q_p,
    pres,
    z,
    IWV=np.nan,
    IWV_p=np.nan,
    max_it_neg_q=10):

    """
    Conservation of IWV of a modified specific humidity profile (q_p) with respect to a given 
    original q profile. Compute IWV of unperturbed and perturbed q profiles and distribute the 
    lost (or gained) IWV fraction to the entire q profile. Pressure and height are also needed
    to compute a standard specific humidity profile that will be used to scale the modified IWV
    to the original IWV in an exponential form (more at the surface, less at the top). The bottom 
    enhanced profile is computed using standard atmosphere and relative humidity assumptions. 
    
    Parameters:
    -----------
    q : array of floats
        Specific humidity profile (in kg kg-1) as 1D-array of shape (height,) that was supposed to
        be modified. 
    q_p : array of floats
        Modified specific humidity profile (in kg kg-1) as 1D-array of shape (height,). 
    pres : array of floats
        One dimensional array of pressure in Pa. Needed if conserve_IWV=True
    z : array of floats
        Height profile (in m) with the same shape as q (height,).
    IWV : float
        Integrated water vapour based on the original q profile (in kg m-2).
    IWV_p : float
        Integrated water vapour based on the perturbed q profile q_p (in kg m-2).
    max_it_neg_q : int
        Integer indicating the maximum number of iterations for the procedure to eliminate negative
        q during IWV conservation before getting suspicious. Default: 5
    """

    if type(pres) == type(None): 
        print("Could not conserve IWV if air pressure data is not provided.")
        return q_p
    
    # compute IWV of original and modified profile:
    if np.isnan(IWV):
        IWV = compute_IWV_q(q, pres, nan_threshold=0.1, scheme='balanced')
    if np.isnan(IWV_p):
        IWV_p = compute_IWV_q(q_p, pres, nan_threshold=0.1, scheme='balanced')
        
    IWV_to_add = IWV - IWV_p        # this amount of IWV must be added to q_p

    n_hgt = len(z)
    dp = get_delta_pres(pres, n_hgt)


    # Create a standard inversion-free specific humidity profile whose shape can be used
    # to distribute the IWV that needs to be added to q_p in a sensible and physically
    # more sound way. Use ICAO standard atmosphere to compute temperature and pressure:
    # https://ntrs.nasa.gov/api/citations/19770009539/downloads/19770009539.pdf p. 3,4,11,12.
    std_hgt = np.array([0.0, 11000.0])  # height boundaries of ICAO std atmosphere
    std_temp = np.array([15.0, -56.5])+273.15   # temperature of ICAO std atmosphere
    std_rh = np.array([0.75,0.0])       # assumed relative humidity to generate a default q profile
                                        # surface relative humidity used according to Peixoto and Oort (1996)
                                        # https://doi.org/10.1175/1520-0442(1996)009<3443:TCORHI>2.0.CO;2

    # interpolate standard atmosphere to current height grid:
    std_temp = np.interp(z, std_hgt, std_temp, left=np.nan, right=np.nan)
    std_rh = np.interp(z, std_hgt, std_rh, left=np.nan, right=np.nan)
    std_hgt = z

    L_isa = -6.5 / 1000.        # lapse rate in K m-1
    std_pres = (101325.0 * (std_temp[0] / (std_temp[0] + L_isa*(std_hgt - std_hgt[0])))**(
                g*m_mol_air / (R_*L_isa)))      # formula 33a in the ref given above
    std_q = convert_rh_to_spechum(std_temp, std_pres, std_rh)
    std_q[np.isnan(std_q)] = 0.0

    # now, use the shape of the standard q profile as a scale function. Norm the scale
    # function by the sum of std_q*layer_thicknesses to get the correct IWV. 
    # sum(scale_function*dp) must be 1.0!
    if IWV_to_add < 0: # invert the standard q profile shape
        scale_function = std_q[::-1] / np.nansum(std_q[::-1]*dp)
    else:
        scale_function = std_q / np.nansum(std_q*dp)

    dq_p = -IWV_to_add*g*scale_function

    q_p += dq_p

    # check if q_p is < 0:
    ik = 0                      # counts iterations of the negative value cure
    fill_val = 1.0e-05          # negative values will be replaced by this quite dry value
    while np.any(q_p < 0):
        if ik > max_it_neg_q: 
            print("Make sure to limit the profiles to the troposphere (e.g., max 11 km) to avoid too many iterations!")
            pdb.set_trace()       # try to avoid too many iterations

        q_p_to_change = q_p[q_p < 0]        # q_p sub zero that needs to be changed
        lost_wv_change = np.nansum(-1*(q_p_to_change - fill_val)*dp[q_p < 0])/g # specific humidity that has still to be
                                                                    # distributed over the q_p profile
                                                                    # roughly equals IWV when computed after the
                                                                    # following step

        q_p[q_p < 0] = fill_val     # set to some very dry value

        # distribute water vapour over all heights:
        dq_p_new = -lost_wv_change*g*scale_function
        q_p += dq_p_new

        ik += 1

    return q_p


def conserve_IWV_uniform(
    q, 
    q_p, 
    pres, 
    z,
    IWV=np.nan,
    IWV_p=np.nan,
    max_it_neg_q=5):

    """
    Conservation of IWV of a modified specific humidity profile (q_p) with respect to a given 
    original q profile. Compute IWV of unperturbed and perturbed q profiles and distribute the 
    lost (or gained) IWV fraction uniformly to the entire q profile. Check for possible negative
    q values and correct them by shifting the total profile again.

    Parameters:
    -----------
    q : array of floats
        Specific humidity profile (in kg kg-1) as 1D-array of shape (height,) that was supposed to
        be modified. 
    q_p : array of floats
        Modified specific humidity profile (in kg kg-1) as 1D-array of shape (height,). 
    pres : array of floats
        One dimensional array of pressure in Pa. 
    z : array of floats
        Height profile (in m) with the same shape as q (height,).
    IWV : float
        Integrated water vapour based on the original q profile (in kg m-2).
    IWV_p : float
        Integrated water vapour based on the perturbed q profile q_p (in kg m-2).
    max_it_neg_q : int
        Integer indicating the maximum number of iterations for the procedure to eliminate negative
        q during IWV conservation before getting suspicious. Default: 5
    """

    # compute IWV of original and modified profile:
    if np.isnan(IWV):
        IWV = compute_IWV_q(q, pres, nan_threshold=0.1, scheme='balanced')
    if np.isnan(IWV_p):
        IWV_p = compute_IWV_q(q_p, pres, nan_threshold=0.1, scheme='balanced')

    # IWV that needs to be added:
    IWV_to_add = IWV - IWV_p

    # distribute water vapour uniformly over all heights:
    dq_p_new = -IWV_to_add*g / (pres[-1] - pres[0])
    q_p += dq_p_new

    n_hgt = len(z)
    dp = get_delta_pres(pres, n_hgt)
    

    # check for negative q values:
    ik = 0                      # counts iterations of the negative value cure
    fill_val = 1.0e-05          # negative values will be replaced by this quite dry value
    q_p_new_edited = -9999      # see below for description
    while np.any(q_p < 0):
        if ik > max_it_neg_q: pdb.set_trace()       # try to avoid too many iterations

        q_p_to_change = q_p[q_p < 0]        # q_p sub zero that needs to be changed
        lost_wv_change = np.nansum(-1*(q_p_to_change - fill_val)*dp[q_p < 0])/g # specific humidity that has still to be
                                                                    # distributed over the q_p profile
                                                                    # roughly equals IWV when computed after the
                                                                    # following step

        if q_p_new_edited > 0: fill_val = q_p_new_edited        # update fill_val if iteration has been done before
                                                                # and if that new fill_val isn't < 0
        q_p[q_p < 0] = fill_val     # set to some very dry value

        # distribute water vapour over all heights:
        dq_p_new = -lost_wv_change*g / (pres[-1] - pres[0])
        q_p += dq_p_new

        # save which value has been used for filling: if > 0, it can be used as a new
        # fill value to avoid adding many new weird q gradients to the profile:
        q_p_new_edited = fill_val + dq_p_new

        ik += 1

    return q_p


def conserve_IWV_factorial(
    q: np.ndarray, 
    q_p: np.ndarray, 
    pres: np.ndarray, 
    z: np.ndarray,
    IWV=np.nan,
    IWV_p=np.nan,
    max_it_neg_q=5):
    
    """
    Conservation of IWV of a modified specific humidity profile (q_p) with respect to a given 
    original q profile. Compute IWV of unperturbed and perturbed q profiles and distribute the 
    lost (or gained) IWV fraction by multiplying the q profile by a factor. Eventually occurring
    negative q values are corrected.

    Parameters:
    -----------
    q : array of floats
        Specific humidity profile (in kg kg-1) as 1D-array of shape (height,) that was supposed to
        be modified. 
    q_p : array of floats
        Modified specific humidity profile (in kg kg-1) as 1D-array of shape (height,). 
    pres : array of floats
        One dimensional array of pressure in Pa. 
    z : array of floats
        Height profile (in m) with the same shape as q (height,).
    IWV : float
        Integrated water vapour based on the original q profile (in kg m-2).
    IWV_p : float
        Integrated water vapour based on the perturbed q profile q_p (in kg m-2).
    max_it_neg_q : int
        Integer indicating the maximum number of iterations for the procedure to eliminate negative
        q during IWV conservation before getting suspicious. Default: 5
    """
    
    # compute IWV of original and modified profile:
    if np.isnan(IWV):
        IWV = compute_IWV_q(q, pres, nan_threshold=0.1, scheme='balanced')
    if np.isnan(IWV_p):
        IWV_p = compute_IWV_q(q_p, pres, nan_threshold=0.1, scheme='balanced')

    q_p *= (IWV/IWV_p)
    
    # check for negative q values:
    ik = 0
    fill_val = 1.0e-05          # negative values will be replaced by this quite dry value
    while np.any(q_p < 0):
        if ik > max_it_neg_q: pdb.set_trace()

        q_p[q_p < 0] = fill_val
        pdb.set_trace()
        IWV_temp = compute_IWV_q(q_p, pres, nan_threshold=0.1, scheme='balanced')
        q_p *= (IWV/IWV_temp)

        ik += 1
    
    return q_p


def get_delta_pres(pres: np.ndarray, n_lev: int):
    
    """
    Get the pressure difference of each height level.
    
    Parameters:
    -----------
    pres : np.ndarray
        1D-array of air pressure (in Pa).
    n_lev : int
        Number of height levels.
    """
    
    dp = np.zeros((n_lev,), dtype=np.float64)
    for k in range(n_lev):
        if k == 0:      # bottom of grid
            dp[k] = 0.5*(pres[k+1] - pres[k])

        elif k == n_lev-1:  # top of grid
            dp[k] = 0.5*(pres[k] - pres[k-1])

        else:           # mid of grid
            dp[k] = 0.5*(pres[k+1] - pres[k-1])
            
    return dp


def Z_from_GP(
    gp):

    """
    Computes geopotential height (in m) from geopotential.

    Parameters:
    -----------
    gp : float or array of floats
        Geopotential in m^2 s^-2.
    """

    return gp / g


def height_from_GP(
    gp):

    """
    Computes height (not geopotential height) in m from geopotential. Based on 
    metpy.calc.geopotential_to_height who uses equation 3.21 on p. 69 in Hobbs, 2006.

    Parameters:
    -----------
    gp : float or array of floats
        Geopotential in m^2 s^-2.
    """

    return (gp * R_e) / (g*R_e - gp)


def mean_scale_height(
    pres,
    temp):

    """
    Computes the mean scale height in m. H = R_d <T> / g with 
    <T> = integral(p2, p1){T(p) d lnp} / integral(p2, p1){d lnp}

    Parameters:
    -----------
    pres : array of floats
        Air pressure in Pa.
    temp : array of floats
        Temperature in K.
    """

    # need to compute the layer averaged mean vertical temperature:
    pdb.set_trace()

    temp_m = np.sum(temp[...,:-1] * np.diff(np.log(pres))) / (np.cumsum(np.diff(np.log(pres))))
    MSH = R_d * temp_m / g

    return MSH


def Z_from_pres_old(
    pres,
    rho,
    pres_sfc):

    """
    Computes the geopotential height in m based on the hydrostatic equation. Pressure must be sorted 
    from high pressure at index 0 to low pressure at the last index. Height axis must be the last axis.
    The returned height array will be sorted from low values at the first to high values at the last index.
    Surface pressure can be added to identify whether the pressure axis contains values below the actual 
    surface. 3D or higher dimension arrays are not (yet) supported.

    Parameters:
    -----------
    pres : array of floats
        Air pressure in Pa.
    rho : array of floats
        Air density (with moisture content) in kg m-3.
    pres_sfc : array of floats or float
        Surface air pressure in Pa.
    """

    # check if pressure array is sorted correctly. If it isn't, flip pressure
    # and density arrays:
    if np.any(np.diff(pres, axis=-1) > 0.0):
        pres = pres[...,::-1]
        rho = rho[...,::-1]

    # check if pres has got values below the surface pressure:
    # pres and pres_sfc might have the same numer of dimensions. 
    # then pres is probably just a "height axis", while pres_sfc contains several data samples
    if pres.ndim == pres_sfc.ndim and len(pres_sfc) > 1:
        # expand pres dimensions:
        pres = np.repeat(np.reshape(pres, (1,len(pres))), len(pres_sfc), axis=0)

        # identify locations below surface
        idx_not_sub = [np.where(pres[k,:] <= pres_sfc[k])[0] for k in range(len(pres_sfc))]

    # compute Z:
    Z = np.full_like(rho, 0.0)

    if rho.ndim == 2:
        n_s = rho.shape[0]
        for k in range(n_s):
            pdb.set_trace()
            Z[k,idx_not_sub[k][:-1]] = -(1.0/g) * np.cumsum((1/rho[k,idx_not_sub[k][:-1]]) * np.diff(pres[k,idx_not_sub[k]], axis=-1), axis=-1)
            Z[k,idx_not_sub[k][-1]] = (Z[k,idx_not_sub[k][-2]] - (1.0/g) * (1/rho[k,idx_not_sub[k][-1]]) * (pres[k,idx_not_sub[k][-1]] - pres[k,idx_not_sub[k][-2]]))

    1/0 # costruction site

    return Z


def Z_from_pres(
    pres, 
    temp, 
    q):
    
    """
    Compute geopotential height Z from pressure (pres) and temperature (temp) and from specific 
    humidity (q) using the hypsometric equation. The thickness of each pressure layer is computed
    using the mean virtual temperature of that layer. The last axis is expected to be the height
    axis. Height is expected to increase with index (index 0 = surface). 
    
    Parameters:
    -----------
    pres : np.ndarray
        Air pressure in Pa.
    temp : np.ndarray
        Air temperature in K.
    q : np.ndarray
        Specific humidity in kg kg-1.
    """
    
    temp_v = virtual_temp(temp, q)
    temp_v_lay = 0.5*(temp_v[...,:-1] + temp_v[...,1:])
    
    Z_lay = np.nancumsum(R_d * temp_v_lay * np.log(pres[...,:-1] / pres[...,1:]) / g, axis=-1)
    z_sfc = np.full(temp.shape[:-1] + (1,), 0.)         # assumes surface to be at sea level
    Z_lev = np.concatenate((z_sfc, Z_lay), axis=-1)
    
    return Z_lev


def ZR_rain_rate(
    Z,
    dsd='mp'):

    """
    Compute rain rate from ZR relation: Z = a*R^b
    a and b vary.

    Parameters:
    -----------
    Z : float or array of floats
        Radar reflectivity factor (in mm^6 m^-3).
    dsd : str
        Identifier of the drop size distribution. Valid options: 'mp'
        'mp': Marshall and Palmer 1948
    """

    if dsd == 'mp':
        a = 296
        b = 1.47
        R = (Z/a)**(1/b)

    else:
        raise ValueError("Function 'ZR_rain_rate' currently only implemented dsd='mp'.")

    return R


def compute_LWC_from_Z(
    Z,
    cloud_type,
    **kwargs):

    """
    Compute Liquid Water Content (LWC, in g m^-3) of a cloud from radar reflectivity factor Z
    (in mm^6 m^-3) using the Z-LWC relation Z = a*LWC**b

    Parameters:
    -----------
    Z : float or array of floats
        Radar reflectivity factor (in mm^6 m^-3).
    cloud_type : str
        Specification of the cloud type. Valid options: 'no_drizzle', 'light_drizzle', 
        'heavy_drizzle'

    **kwargs:
    algorithm : str
        Specfiy the algorithm. This argument is ignored if cloud_type is not 'no_drizzle'.
        Valid options: 'Fox_and_Illingworth_1997_i', 'Fox_and_Illingworth_1997_ii',
        'Sauvageot_and_Omar_1987', 'Liao_and_Sassen_1994'
    """

    algorithms = {  'Fox_and_Illingworth_1997_i':       {'a': 0.012, 'b': 1.16},        # no drizzle
                    'Fox_and_Illingworth_1997_ii':      {'a': 0.031, 'b': 1.56},        # no drizzle
                    'Sauvageot_and_Omar_1987':          {'a': 0.030, 'b': 1.31},        # no drizzle
                    'Liao_and_Sassen_1994':             {'a': 0.036, 'b': 1.80},        # no drizzle
                    'Baedi_et_al_2000':                 {'a': 57.54, 'b': 5.17},        # light drizzle
                    'Krasnov_and_Russchenberg_2002':    {'a': 323.59, 'b': 1.58}}       # heavy drizzle

    # select algorithm:
    if cloud_type == 'no_drizzle':
        algorithm = 'Fox_and_Illingworth_1997_i'

        if 'algorithm' in kwargs.keys():
            algorithm = kwargs['algorithm']

    elif cloud_type == 'light_drizzle':
        algorithm = 'Baedi_et_al_2000'

    elif cloud_type == 'heavy_drizzle':
        algorithm = 'Krasnov_and_Russchenberg_2002'

    else:
        raise ValueError("'cloud_type' for compute_LWC_from_Z must be 'no_drizzle, " +
                            "'light_drizzle', or 'heavy_drizzle'.")

    LWC = (Z / algorithms[algorithm]['a'])**(1/algorithms[algorithm]['b'])

    return LWC