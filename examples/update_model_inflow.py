# %% [markdown]
# ## Update a Wflow model: river inflow point series
#
# With HydroMT you can add time-varying inflow (or outflow) at specific river locations
# using the `setup_point_series_river_flux` method.  This is useful when you have
# observed or modelled discharge from upstream tributaries, reservoirs, or water
# transfers that are not resolved by the model grid.
#
# This script demonstrates the full workflow:
#
# 1. Prepare a synthetic GeoDataset with inflow time series at two river points.
# 2. Register the dataset in the model data catalog.
# 3. Load an existing Wflow model (the Piave sub-basin example).
# 4. Call `setup_point_series_river_flux` to snap the points to the river network
#    and rasterise the time series onto the model forcing grid.
# 5. Write the updated forcing and configuration.
# 6. Inspect the result.

# %% Imports
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

from hydromt.data_catalog.sources import create_source
from hydromt_wflow import WflowSbmModel

# %% Paths
DATA_ROOT = Path("examples", "data", "inflow")
MODEL_ROOT = Path("examples", "wflow_piave_subbasin")
OUT_ROOT = Path("examples", "wflow_piave_subbasin_inflow")
TOML_FN = "wflow_sbm.toml"

DATA_ROOT.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### 1. Create a synthetic inflow GeoDataset
#
# The dataset must have:
#   - dims   : `(time, inflow_id)`
#   - coords : `time`, `inflow_id`, `lon(inflow_id)`, `lat(inflow_id)`
#   - variable: `Q`  [m³/s]
#
# Here we choose two points that lie on known Piave river cells:
#   - point 1: lon=12.45, lat=46.4833  — main Piave tributary
#   - point 2: lon=12.30, lat=46.4000  — main Piave river

# %%
lons = [12.45, 12.30]
lats = [46.4833, 46.4000]
ids = [1, 2]

# Match model time range: 2010-02-02 to 2010-02-10, daily
time_index = pd.date_range("2010-02-02", "2010-02-10", freq="1D")

rng = np.random.default_rng(42)
Q_data = np.column_stack([
    50.0 + rng.normal(0, 5, len(time_index)),   # point 1: ~50 m³/s
    20.0 + rng.normal(0, 2, len(time_index)),   # point 2: ~20 m³/s
]).astype(np.float32)

ds_inflow = xr.Dataset(
    {
        "Q": xr.DataArray(
            data=Q_data,
            dims=["time", "inflow_id"],
            coords={
                "time": time_index,
                "inflow_id": ids,
                "lon": ("inflow_id", lons),
                "lat": ("inflow_id", lats),
            },
            attrs={"units": "m3/s"},
        )
    }
)

print(ds_inflow)

# %% Save to disk
inflow_nc = DATA_ROOT / "inflow_sources.nc"
ds_inflow.to_netcdf(inflow_nc)
print(f"Saved inflow dataset to: {inflow_nc}")

# %% [markdown]
# ### 2. Load the model and register the inflow dataset

# %%
model = WflowSbmModel(root=str(MODEL_ROOT), config_filename=TOML_FN, mode="r+")
model.read()
model.root.set(OUT_ROOT, mode="w+")
model.write()

model = WflowSbmModel(root=str(OUT_ROOT), config_filename=TOML_FN, mode="r+")
model.read()

print(f"Model loaded.")
print(f"  Forcing vars : {list(model.forcing.data.data_vars)}")
print(f"  Time range   : {model.config.get_value('time.starttime')} -> "
      f"{model.config.get_value('time.endtime')}")

# Register inflow dataset in the model data catalog
source = create_source(
    data={
        "name": "piave_inflow_nc",
        "data_type": "GeoDataset",
        "driver": {"name": "geodataset_xarray"},
        "uri": str(inflow_nc),
        "metadata": {"crs": "EPSG:4326"},
    }
)
model.data_catalog.add_source(name="piave_inflow_nc", source=source)

# %% [markdown]
# ### 3. Add inflow forcing with `setup_point_series_river_flux`
#
# The method will:
# - Snap each point to the nearest river cell (within `max_dist` metres).
# - Rasterise the time series onto the model grid (zeros elsewhere).
# - Optionally add a binary `Q_mask` layer marking active inflow cells.
# - Update the TOML config to link the new forcing variable.

# %%
model.setup_point_series_river_flux(
    geodataset_source="piave_inflow_nc",
    flux_var="Q",
    wflow_var="river_water__external_inflow_volume_flow_rate",
    out_var="inflow",
    add_q_mask=True,
    max_dist=5e3,
)

# Point the TOML to a new forcing file so the original is not overwritten
model.setup_config({"input.path_forcing": "forcing_piave_with_inflow.nc"})

# %% Write outputs
model.config.write()
model.forcing.write()

print("\nUpdated forcing vars:", list(model.forcing.data.data_vars))

# %% [markdown]
# ### 4. Inspect the result

# %%
ds_out = model.forcing.data

if "inflow" in ds_out.data_vars:
    da_inflow = ds_out["inflow"]
    print(f"\ninflow shape       : {da_inflow.shape}")
    n_nonzero = int((da_inflow.values != 0).sum())
    print(f"Non-zero cells     : {n_nonzero}")
    print(f"Max inflow per step: {da_inflow.values.max(axis=(1, 2))}")

if "inflow_points" in model.geoms:
    print(f"\nSnapped inflow points:\n{model.geoms['inflow_points']}")

# %% [markdown]
# ### 5. Plot: inflow at the two river cells over time

# %%
if "inflow" in ds_out.data_vars:
    da_inflow = ds_out["inflow"]

    # Find cells with non-zero inflow (any timestep)
    active = (da_inflow.values != 0).any(axis=0)
    ys, xs = np.where(active)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (row, col) in enumerate(zip(ys, xs)):
        y_val = da_inflow[da_inflow.raster.y_dim].values[row]
        x_val = da_inflow[da_inflow.raster.x_dim].values[col]
        series = da_inflow.values[:, row, col]
        ax.plot(time_index, series, label=f"cell ({x_val:.3f}, {y_val:.3f})")

    ax.set_xlabel("time")
    ax.set_ylabel("inflow [m³/s]")
    ax.set_title("River inflow at snapped cells")
    ax.legend()
    fig.tight_layout()
    plt.show()
