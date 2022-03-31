#%%
import numpy as np
import xarray as xr
#import pandas as pd




#%% Import files and select He and Te
saved_file = xr.open_dataset("western-europe-2011-01.nc")

wave_height = saved_file.get('wave_height')
wave_period = saved_file.get('wave_period')

# Round up to the next heights 0.5 in new Data arrays
Hs = np.ceil(wave_height*2)/2
Te = np.ceil(wave_period*2)/2

print(Hs.dims,Hs.shape, Te.dims, Te.shape)
#%% Import power_matrix
import pandas as pd
power_matrix = pd.read_excel("PowerMatrix_PyPsa.xlsx", header = 2, usecols= "C:AN", index_col=0)
print(power_matrix)

#%% create an empty matrix
power_test = xr.DataArray.copy(Hs)
power_test[:] = np.nan
power_test = power_test.rename('Generated Power')
print(power_test.dims, power_test.shape)

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
power_test.to_netcdf('power_matrix.nc')

#%%
power_test.values


#%% load
power_test_dataset = xr.open_dataset('power_matrix.nc')
power_test_loaded = power_test_dataset.get('Generated Power')
print(power_test_loaded.values)


