#%%
import atlite
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr


#import pandas as pd




#%% Import files and select He and Te
# saved_file = xr.open_dataset("western-europe-2011-01.nc")

# wave_height = saved_file.get('wave_height')
# wave_period = saved_file.get('wave_period')

#%%
cutout = atlite.Cutout(path="western-europe-2011-01.nc",
                       module="era5",
                       x=slice(-13.6913, 1.7712),
                       y=slice(49.9096, 60.8479),
                       time="2011-01",
                       #features = ["wave_height"]
                       )


#%%
def convert_and_aggregate(
    cutout,
    convert_func,
    matrix=None,
    index=None,
    layout=None,
    shapes=None,
    shapes_crs=4326,
    per_unit=False,
    return_capacity=False,
    capacity_factor=False,
    show_progress=True,
    dask_kwargs={},
    **convert_kwds,
):
    """
    Convert and aggregate a weather-based renewable generation time-series.

    NOTE: Not meant to be used by the user him or herself. Rather it is a
    gateway function that is called by all the individual time-series
    generation functions like pv and wind. Thus, all its parameters are also
    available from these.

    Parameters
    -----------
    matrix : N x S - xr.DataArray or sp.sparse.csr_matrix or None
        If given, it is used to aggregate the grid cells to buses.
        N is the number of buses, S the number of spatial coordinates, in the
        order of `cutout.grid`.
    index : pd.Index
        Index of Buses.
    layout : X x Y - xr.DataArray
        The capacity to be build in each of the `grid_cells`.
    shapes : list or pd.Series of shapely.geometry.Polygon
        If given, matrix is constructed as indicatormatrix of the polygons, its
        index determines the bus index on the time-series.
    shapes_crs : pyproj.CRS or compatible
        If different to the map crs of the cutout, the shapes are
        transformed to match cutout.crs (defaults to EPSG:4326).
    per_unit : boolean
        Returns the time-series in per-unit units, instead of in MW (defaults
        to False).
    return_capacity : boolean
        Additionally returns the installed capacity at each bus corresponding
        to `layout` (defaults to False).
    capacity_factor : boolean
        If True, the static capacity factor of the chosen resource for each
        grid cell is computed.
    show_progress : boolean, default True
        Whether to show a progress bar.
    dask_kwargs : dict, default {}
        Dict with keyword arguments passed to `dask.compute`.

    Other Parameters
    -----------------
    convert_func : Function
        Callback like convert_wind, convert_pv


    Returns
    -------
    resource : xr.DataArray
        Time-series of renewable generation aggregated to buses, if
        `matrix` or equivalents are provided else the total sum of
        generated energy.
    units : xr.DataArray (optional)
        The installed units per bus in MW corresponding to `layout`
        (only if `return_capacity` is True).

    """

    func_name = convert_func.__name__.replace("convert_", "")
    logger.info(f"Convert and aggregate '{func_name}'.")
    da = convert_func(cutout.data, **convert_kwds)

    no_args = all(v is None for v in [layout, shapes, matrix])

    if no_args:
        if per_unit or return_capacity:
            raise ValueError(
                "One of `matrix`, `shapes` and `layout` must be "
                "given for `per_unit` or `return_capacity`"
            )
        if capacity_factor:
            res = da.mean("time").rename("capacity factor")
            res.attrs["units"] = "p.u."
            return maybe_progressbar(res, show_progress, **dask_kwargs)
        else:
            res = da.sum("time", keep_attrs=True)
            return maybe_progressbar(res, show_progress, **dask_kwargs)

    if matrix is not None:

        if shapes is not None:
            raise ValueError(
                "Passing matrix and shapes is ambiguous. Pass " "only one of them."
            )

        if isinstance(matrix, xr.DataArray):

            coords = matrix.indexes.get(matrix.dims[1]).to_frame(index=False)
            if not np.array_equal(coords[["x", "y"]], cutout.grid[["x", "y"]]):
                raise ValueError(
                    "Matrix spatial coordinates not aligned with cutout spatial "
                    "coordinates."
                )

            if index is None:
                index = matrix

        if not matrix.ndim == 2:
            raise ValueError("Matrix not 2-dimensional.")

        matrix = csr_matrix(matrix)

    if shapes is not None:

        geoseries_like = (pd.Series, gpd.GeoDataFrame, gpd.GeoSeries)
        if isinstance(shapes, geoseries_like) and index is None:
            index = shapes.index

        matrix = cutout.indicatormatrix(shapes, shapes_crs)

    if layout is not None:

        assert isinstance(layout, xr.DataArray)
        layout = layout.reindex_like(cutout.data).stack(spatial=["y", "x"])

        if matrix is None:
            matrix = csr_matrix(layout.expand_dims("new"))
        else:
            matrix = csr_matrix(matrix) * spdiag(layout)

    # From here on, matrix is defined and ensured to be a csr matrix.
    if index is None:
        index = pd.RangeIndex(matrix.shape[0])

    results = aggregate_matrix(da, matrix=matrix, index=index)

    if per_unit or return_capacity:
        caps = matrix.sum(-1)
        capacity = xr.DataArray(np.asarray(caps).flatten(), [index])
        capacity.attrs["units"] = "MW"

    if per_unit:
        results = (results / capacity.where(capacity != 0)).fillna(0.0)
        results.attrs["units"] = "p.u."
    else:
        results.attrs["units"] = "MW"

    if return_capacity:
        return maybe_progressbar(results, show_progress, **dask_kwargs), capacity
    else:
        return maybe_progressbar(results, show_progress, **dask_kwargs)

#%%
def wec(cutout, **params):
    """
    Generate wind generation time-series

    Extrapolates 10m wind speed with monthly surface roughness to hub
    height and evaluates the power curve.

    Parameters
    ----------
    turbine : str or dict
        A turbineconfig dictionary with the keys 'hub_height' for the
        hub height and 'V', 'POW' defining the power curve.
        Alternatively a str refering to a local or remote turbine configuration
        as accepted by atlite.resource.get_windturbineconfig().


    """

 
    return cutout.convert_and_aggregate(
        convert_func=convert_wec,  **params
    )

#%%
def convert_wec(ds):

    #Get power matrix
    power_matrix = pd.read_excel("PowerMatrix_PyPsa.xlsx", header = 2, usecols= "C:AN", index_col=0)
    #pm = power_matrix.to_xarray()
    #power_matrix = gen['Power_Matrix']

    #max power
    max_pow = power_matrix.to_numpy().max()
    #max_pow = 750
 
    ###Round up values of Hs an Tp creating new datarrays
    Hs = np.ceil(ds['wave_height']*2)/2
    Tp = np.ceil(ds['wave_period']*2)/2

    # Hs = Hs.fillna(0)
    # Tp = Tp.fillna(0)
    #Empty dataarray of results
    da = xr.DataArray.copy(Hs)
    da[:] = np.nan
    
    #Call datarrays values from the above dataset Dataset. need to modify function
    #
    #data arrya with results
    cases = len(da.x.values) * len(da.y.values) * len(da.time.values)
    count = 0

    for x in da.x.values:
        
        for y in da.y.values:
            #print (x,y)
            for t in da.time.values:

                if count % 1000 == 0:
                    print('Case {} of {}: {} %'.format(count, cases, count/cases * 100))

                
                Hs_i= Hs.sel(x=x, y=y, time=t).values
                Tp_i =Tp.sel(x=x, y=y, time=t).values
                
                if np.isnan(Hs_i) or np.isnan(Tp_i):
                    power = np.nan
                else:
                    power =power_matrix.loc[Hs_i, Tp_i]
                    #power =np.ndarray.item(pm[Tp_i].sel(index = Hs_i).values)
                
                    if Hs_ind % 0.5 != 0:
                        print('Case {} of {}'.format(count, cases))
                        print('Hs_ind has wrong value')
                        print(t, x, y)

                    if Te_ind % 0.5 !=0:
                        print('Case {} of {}'.format(count, cases))
                        print('Te_ind has wrong value')
                        print(t, x, y)                
                
                
                cap = power/max_pow
                da.loc[dict(x= x, y= y,time = t )] = cap
                #print (t, x, y)
                #print (Hs_i, Tp_i, power, cap)
                # increas counter
                count += 1

    da.attrs["units"] = "KWh/KWp"
    da = da.rename("specific generation")
    return da
#%%
cases = len(power_test.x.values) * len(power_test.y.values) * len(power_test.time.values)
count = 0

for x in power_test.x.values:

    for y in power_test.y.values:

        for t in power_test.time.values:

            if count % 1000 == 0:
                print('Case {} of {}: {} %'.format(count, cases, count/cases * 100))

            # print (t, x, y)
            Hs_ind = Hs.sel(x=x, y=y, time=t).values
            Te_ind = Te.sel(x=x, y=y, time=t).values

            if np.isnan(Hs_ind) or np.isnan(Te_ind):
                power = np.nan
            else:
                power = power_matrix.loc[Hs_ind, Te_ind]

                if Hs_ind % 0.5 != 0:
                    print('Case {} of {}'.format(count, cases))
                    print('Hs_ind has wrong value')
                    print(t, x, y)

                if Te_ind % 0.5 !=0:
                    print('Case {} of {}'.format(count, cases))
                    print('Te_ind has wrong value')
                    print(t, x, y)

            power_test.loc[dict(x=x, y=y, time=t)] = power

            # increas counter
            count += 1

#%%
#power_test.to_netcdf('power_matrix.nc')

#%%
power_test.values


#%% load
power_test_dataset = xr.open_dataset('power_matrix.nc')
power_test_loaded = power_test_dataset.get('Generated Power')
print(power_test_loaded.values)



# %%

test = wec(cutout, capacity_factor = True, dask_kwargs = {'num_workers' : 4})
# %%
r=cutout.data['wave_height']
r
# %%
capacity_factor = cutout.wec(capacity_factor = True, dask_kwargs = {'num_workers' : 6})

# %%
