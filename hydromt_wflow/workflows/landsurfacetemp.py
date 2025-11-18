"""Land surface temperature workflow for wflow."""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from hydromt.model.processes.meteo import resample_time

logger = logging.getLogger(__name__)

__all__ = ["albedo", "albedo_from_mapping", "setup_albedo", "emissivity", "emissivity_from_mapping", "setup_emissivity", "canopy_height_from_mapping", "setup_canopy_height", "setup_shortwave", "setup_wind", "setup_LST_forcing", "radiation", #"add_var_to_forcing", 
           "solar_declination", "relative_distance", "extraterrestrial_radiation",
           "compute_net_longwave_radiation", "compute_net_radiation", "wind", "parameter_from_mapping"]


def solar_declination(doy: int) -> float:
    """
    Calculate solar declination angle.
    
    Parameters
    ----------
    doy : int
        Day of year (1-365)
        
    Returns
    -------
    float
        Solar declination angle in radians
    """
    days_in_year = 365
    radians = np.sin((2 * np.pi * doy / days_in_year) - 1.39)
    decl = 0.409 * radians
    return decl


def relative_distance(doy: int) -> float:
    """
    Calculate relative distance between Earth and Sun.
    
    Parameters
    ----------
    doy : int
        Day of year (1-365)
        
    Returns
    -------
    float
        Relative distance in AU
    """
    days_in_year = 365
    radians = np.cos(2 * np.pi * doy / days_in_year)
    dist = radians * 0.033 + 1  # distance in AU
    return dist


def extraterrestrial_radiation(lat: float, doy: int) -> float:
    """
    Calculate extraterrestrial radiation.
    
    Parameters
    ----------
    lat : float
        Latitude in degrees
    doy : int
        Day of year (1-365)
        
    Returns
    -------
    float
        Extraterrestrial radiation in MJ/m²/day
    """
    gsc = 118.08  # MJ/m²/day
    lat_rad = np.radians(lat)
    decl = solar_declination(doy)
    dist = relative_distance(doy)
    sha = np.arccos(-np.tan(lat_rad) * np.tan(decl))
    Ra = (dist * gsc / np.pi * 
          (np.cos(lat_rad) * np.cos(decl) * np.sin(sha) + 
           sha * np.sin(lat_rad) * np.sin(decl)))
    return Ra


def compute_net_longwave_radiation(
    air_temperature: xr.DataArray,
    shortwave_radiation_in: xr.DataArray,
    latitude: xr.DataArray,
    time_coord: xr.DataArray,
    emissivity: Optional[Union[xr.DataArray, float]] = None,
) -> xr.DataArray:
    """
    Calculate net longwave radiation.
    
    Formula: RLN = ε(σTa^4)(0.34 - 0.14√ea)(1.35(Rins/Rso) - 0.35)
    
    Parameters
    ----------
    air_temperature : xr.DataArray
        Air temperature [°C]
    shortwave_radiation_in : xr.DataArray
        Incoming shortwave radiation [W m-2]
    latitude : xr.DataArray
        Latitude [degrees]
    time_coord : xr.DataArray
        Time coordinate for day of year calculation
    emissivity : xr.DataArray or float, optional
        Surface emissivity [-]. Defaults to 0.97 if not provided.
        
    Returns
    -------
    xr.DataArray
        Net longwave radiation [W m-2]
    """
    # Stefan-Boltzmann constant in MJK-4m-2day-1
    sigma = 4.903e-9
    
    # Convert temperature to Kelvin and calculate Ta^4
    temp_kelvin = air_temperature + 273.15
    temp_kelvin_4 = temp_kelvin ** 4
    
    # Calculate vapor pressure (ea)
    # ea = tetens formulation 2008 monteith update
    # temp in celcius and ea in kPa, expected values 25 degrees goes to 3.188
    vapor_pressure = 0.61078 * np.exp( ( air_temperature * 17.27 ) / (237.3 + air_temperature) )
    
    # Calculate (0.34 - 0.14√ea)
    b_term = 0.34 - 0.14 * np.sqrt(vapor_pressure)
    
    # Calculate extraterrestrial radiation for each time step
    doy = time_coord.dt.dayofyear
    
    Rso = xr.apply_ufunc(
        extraterrestrial_radiation,
        latitude,
        doy,
        input_core_dims=[[], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]
    ) * 0.75  # Clear sky solar radiation
    
    # Convert shortwave from W m-2 to MJ/m²/day for ratio calculation
    # 1 W m-2 = 0.0864 MJ/m²/day
    shortwave_mj = shortwave_radiation_in * 0.0864
    
    # Calculate (1.35(Rins/Rso) - 0.35)
    ratio = shortwave_mj / Rso
    
    c_term = (1.35 * ratio) - 0.35
    
    # Use emissivity if provided, otherwise default to 0.97
    if emissivity is None:
        eps = 0.97
    else:
        eps = emissivity
    
    # Calculate net longwave radiation with emissivity
    net_longwave = - eps * sigma * temp_kelvin_4 * b_term * c_term
    
    # Convert back to W m-2
    net_longwave_w = net_longwave / 0.0864
    
    # Set attributes
    net_longwave_w.name = "net_longwave_radiation"
    net_longwave_w.attrs.update({
        "unit": "W m-2",
        "long_name": "Net longwave radiation",
        "description": "Calculated using Stefan-Boltzmann law and atmospheric correction"
    })
    return net_longwave_w


def compute_net_radiation(
    albedo: xr.DataArray,
    shortwave_radiation_in: xr.DataArray,
    air_temperature: xr.DataArray,
    latitude: xr.DataArray,
    time_coord: xr.DataArray,
    emissivity: Optional[Union[xr.DataArray, float]] = None,
) -> xr.DataArray:
    """
    Calculate net radiation.
    
    Formula: Rn = Rins - Routs + Rinl - Routl
    Simplified: Rn = (1-α)Rins - RLN
    
    Parameters
    ----------
    albedo : xr.DataArray
        Surface albedo [-]
    shortwave_radiation_in : xr.DataArray
        Incoming shortwave radiation [W m-2]
    air_temperature : xr.DataArray
        Air temperature [°C]
    latitude : xr.DataArray
        Latitude [degrees]
    time_coord : xr.DataArray
        Time coordinate for day of year calculation
    emissivity : xr.DataArray or float, optional
        Surface emissivity [-]. Defaults to 0.97 if not provided.
        
    Returns
    -------
    xr.DataArray
        Net radiation [W m-2]
    """
    # Calculate net shortwave radiation: (1-α)Rins
    net_shortwave = (1 - albedo) * shortwave_radiation_in
    
    # Calculate net longwave radiation
    net_longwave = compute_net_longwave_radiation(
        air_temperature, shortwave_radiation_in, latitude, time_coord, emissivity
    )
    
    # Calculate net radiation: Rn = RSNet - RLN
    net_radiation = net_shortwave - net_longwave
    
    # Set attributes
    net_radiation.name = "net_radiation"
    net_radiation.attrs.update({
        "unit": "W m-2",
        "long_name": "Net radiation",
        "description": "Net radiation = net shortwave - net longwave"
    })
    
    return net_radiation


def parameter_from_mapping(
    mod,
    df: pd.DataFrame,
    param_name: str,
) -> xr.DataArray:
    """
    Create parameter map from landcover class mapping.
    
    Parameters
    ----------
    mod : WflowModel
        Model instance
    df : pd.DataFrame
        Mapping table with landuse class index and parameter column
    param_name : str
        Parameter name (e.g. 'albedo', 'emissivity')
        
    Returns
    -------
    xr.DataArray
        Static parameter map
    """
    landuse_name = mod._MAPS.get("landuse", "meta_landuse")
    if landuse_name not in mod.staticmaps.data:
        raise ValueError(f"Landuse map '{landuse_name}' not found in staticmaps")
    
    landuse = mod.staticmaps.data[landuse_name]
    
    if param_name not in df.columns:
        raise ValueError(f"DataFrame must contain '{param_name}' column")
    
    mapping = dict(zip(df.index, df[param_name]))
    nodata = df[param_name].iloc[-1] if len(df) > 0 else np.nan
    
    def reclass(x):
        return np.vectorize(mapping.get)(x, nodata)
    
    param_out = xr.apply_ufunc(
        reclass, landuse, dask="parallelized", output_dtypes=[float]
    )
    param_out.attrs.update(_FillValue=nodata)
    param_out.name = param_name
    param_out.attrs.update(unit="1")
    
    return param_out


def albedo_from_mapping(
    mod,
    albedo: pd.DataFrame,
) -> xr.DataArray:
    """Create albedo map from landcover class mapping."""
    return parameter_from_mapping(mod, albedo, "albedo")


def emissivity_from_mapping(
    mod,
    emissivity: pd.DataFrame,
) -> xr.DataArray:
    """Create emissivity map from landcover class mapping."""
    return parameter_from_mapping(mod, emissivity, "emissivity")


def canopy_height_from_mapping(
    mod,
    canopy_height: pd.DataFrame,
) -> xr.DataArray:
    """Create canopy height map from landcover class mapping."""
    return parameter_from_mapping(mod, canopy_height, "canopy_height")


def albedo(
    mod,
    albedo: xr.DataArray,
    freq: Optional[str] = None,
    reproj_method: str = "nearest_index",
    resample_kwargs: Optional[dict] = None,
) -> xr.DataArray:
    """
    Process albedo data for land surface temperature calculations.
    
    Parameters
    ----------
    albedo : xr.DataArray
        Albedo data array
    da_model : xr.DataArray or xr.Dataset
        Target grid for reprojection
    freq : str, optional
        Resampling frequency, by default None
    reproj_method : str, optional
        Reprojection method, by default "nearest_index"
    resample_kwargs : dict, optional
        Additional resampling arguments, by default None
        
    Returns
    -------
    xr.DataArray
        Processed albedo data
    """
    resample_kwargs = resample_kwargs or {}
    
    if albedo.raster.dim0 != "time":
        raise ValueError(f'First albedo dim should be "time", not {albedo.raster.dim0}')
    
    # reproject to model grid
    albedo_out = albedo.raster.reproject_like(mod.staticmaps.data["land_elevation"], method=reproj_method)
    
    # resample time if requested
    albedo_out.name = "albedo"
    albedo_out.attrs.update(unit="1")
    
    if freq is not None:
        resample_kwargs.update(upsampling="bfill", downsampling="mean")
        albedo_out = resample_time(albedo_out, freq, conserve_mass=False, **resample_kwargs)
    
    return albedo_out


def setup_albedo(
    mod,
    albedo: Union[str, xr.DataArray, None],
    starttime: str,
    endtime: str,
    freq: pd.Timedelta,
    reproj_method: str = "nearest_index",
) -> None:
    """
    Setup albedo data for land surface temperature calculations.
    
    Parameters
    ----------
    mod : WflowModel
        Model instance
    albedo : str, xr.DataArray, or None
        Albedo source name, DataArray, or None
    starttime : str
        Start time for data retrieval
    endtime : str
        End time for data retrieval
    freq : pd.Timedelta
        Time frequency for resampling
    reproj_method : str, optional
        Reprojection method, by default "nearest_index"
    """
    if albedo is None:
        return
    
    albedo_func = albedo
    
    if isinstance(albedo, str):
        if albedo in mod.data_catalog.sources.keys():
            source_type = mod.data_catalog.get_source(albedo).data_type
            if source_type == "DataFrame":
                albedo_df = mod.data_catalog.get_dataframe(albedo)
                albedo_out = albedo_from_mapping(mod=mod, albedo=albedo_df)
                logger.info("Processed albedo data from mapping table")
                mod.staticmaps.set(albedo_out, name="albedo_lulc")
                mod._update_config_variable_name("albedo_lulc", data_type="static")
                return
            elif source_type == "RasterDataset":
                albedo_func = mod.data_catalog.get_rasterdataset(
                    albedo,
                    geom=mod.region,
                    buffer=2,
                    time_tuple=(starttime, endtime),
                    variables=["albedo"]
                )
                logger.info(f"Retrieved albedo data from data catalog")
            else:
                albedo_func = (mod.staticmaps.data.meta_landuse * 0) + 0.23
                logger.info("No valid data source for albedo, setting default albedo of 0.23")
                mod.staticmaps.set(albedo_func, name="albedo")
                mod._update_config_variable_name("albedo", data_type="static")
                return
    
    if albedo_func is None:
        return
    
    if isinstance(albedo_func, xr.DataArray):
        albedo_da = albedo_func.astype("float32")
        
        if "time" in albedo_da.coords:
            logger.info("Processing time-varying albedo data for forcing")
            albedo_out = albedo(
                mod=mod,
                albedo=albedo_da,
                freq=freq,
                reproj_method=reproj_method,
            )
            mod.forcing.set(albedo_out, name="albedo")
            mod._update_config_variable_name("albedo", data_type="forcing")
        else:
            logger.info("Processing static albedo data for grid")
            albedo_out = albedo(
                mod=mod,
                albedo=albedo_da,
                reproj_method=reproj_method,
            )
            mod.staticmaps.set(albedo_out, name="albedo")
            mod._update_config_variable_name("albedo", data_type="static")
    else:
        raise ValueError(f"Invalid type for albedo: {type(albedo_func)}")


def emissivity(
    mod,
    emissivity: xr.DataArray,
    freq: Optional[str] = None,
    reproj_method: str = "nearest_index",
    resample_kwargs: Optional[dict] = None,
) -> xr.DataArray:
    """
    Process emissivity data for land surface temperature calculations.
    
    Parameters
    ----------
    emissivity : xr.DataArray
        Emissivity data array
    da_model : xr.DataArray or xr.Dataset
        Target grid for reprojection
    freq : str, optional
        Resampling frequency, by default None
    reproj_method : str, optional
        Reprojection method, by default "nearest_index"
    resample_kwargs : dict, optional
        Additional resampling arguments, by default None
        
    Returns
    -------
    xr.DataArray
        Processed emissivity data
    """
    resample_kwargs = resample_kwargs or {}
    
    if emissivity.raster.dim0 != "time":
        raise ValueError(f'First emissivity dim should be "time", not {emissivity.raster.dim0}')
    
    # reproject to model grid
    emissivity_out = emissivity.raster.reproject_like(mod.staticmaps.data["land_elevation"], method=reproj_method)
    
    # ensure values are between 0 and 1
    emissivity_out = np.clip(emissivity_out, 0, 1)
    
    # resample time if requested
    emissivity_out.name = "emissivity"
    emissivity_out.attrs.update(unit="1")
    if freq is not None:
        resample_kwargs.update(upsampling="bfill", downsampling="mean")
        emissivity_out = resample_time(emissivity_out, freq, conserve_mass=False, **resample_kwargs)
    
    return emissivity_out


def setup_emissivity(
    mod,
    emissivity: Union[str, xr.DataArray, None],
    starttime: str,
    endtime: str,
    freq: pd.Timedelta,
    reproj_method: str = "nearest_index",
) -> None:
    """
    Setup emissivity data for land surface temperature calculations.
    
    Parameters
    ----------
    mod : WflowModel
        Model instance
    emissivity : str, xr.DataArray, or None
        Emissivity source name, DataArray, or None
    starttime : str
        Start time for data retrieval
    endtime : str
        End time for data retrieval
    freq : pd.Timedelta
        Time frequency for resampling
    reproj_method : str, optional
        Reprojection method, by default "nearest_index"
    """
    if emissivity is None:
        return
    
    emissivity_da = emissivity
    
    if isinstance(emissivity, str):
        if emissivity in mod.data_catalog.sources.keys():
            source_type = mod.data_catalog.get_source(emissivity).data_type
            if source_type == "DataFrame":
                emissivity_df = mod.data_catalog.get_dataframe(emissivity)
                emissivity_da = emissivity_from_mapping(mod=mod, emissivity=emissivity_df)
                logger.info("Processed emissivity data from mapping table")
                mod.staticmaps.set(emissivity_da, name="emissivity")
                mod._update_config_variable_name("emissivity", data_type="static")
                return
            elif source_type == "RasterDataset":
                emissivity_da = mod.data_catalog.get_rasterdataset(
                    emissivity,
                    geom=mod.region,
                    buffer=2,
                    time_tuple=(starttime, endtime),
                    variables=["emissivity"]
                )
                logger.info(f"Retrieved emissivity data from data catalog")
            else:
                raise ValueError(f"Invalid source type for emissivity: {source_type}")
    
    if emissivity_da is None:
        return
    
    if isinstance(emissivity_da, xr.DataArray):
        emissivity_da = emissivity_da.astype("float32")
        
        if "time" in emissivity_da.coords:
            logger.info("Processing time-varying emissivity data for forcing")
            emissivity_out = emissivity(
                mod=mod,
                emissivity=emissivity_da,
                freq=freq,
                reproj_method=reproj_method,
            )
            mod.forcing.set(emissivity_out, name="emissivity")
            mod._update_config_variable_name("emissivity", data_type="forcing")
        else:
            logger.info("Processing static emissivity data for grid")
            emissivity_out = emissivity(
                mod=mod,
                emissivity=emissivity_da,
                reproj_method=reproj_method,
            )
            mod.staticmaps.set(emissivity_out, name="emissivity")
            mod._update_config_variable_name("emissivity", data_type="static")
    else:
        raise ValueError(f"Invalid type for emissivity: {type(emissivity_da)}")


def setup_canopy_height(
    mod,
    canopy_height: Union[str, xr.DataArray, None],
    starttime: str,
    endtime: str,
    freq: pd.Timedelta,
    reproj_method: str = "nearest_index",
) -> None:
    """
    Setup canopy height data for land surface temperature calculations.
    
    Parameters
    ----------
    mod : WflowModel
        Model instance
    canopy_height : str, xr.DataArray, or None
        Canopy height source name, DataArray, or None
    starttime : str
        Start time for data retrieval
    endtime : str
        End time for data retrieval
    freq : pd.Timedelta
        Time frequency for resampling
    reproj_method : str, optional
        Reprojection method, by default "nearest_index"
    """
    if canopy_height is None:
        return
    
    canopy_height_da = canopy_height
    
    if isinstance(canopy_height, str):
        if canopy_height in mod.data_catalog.sources.keys():
            source_type = mod.data_catalog.get_source(canopy_height).data_type
            if source_type == "DataFrame":
                canopy_height_df = mod.data_catalog.get_dataframe(canopy_height)
                canopy_height_da = canopy_height_from_mapping(mod=mod, canopy_height=canopy_height_df)
                logger.info("Processed canopy_height data from mapping table")
                mod.staticmaps.set(canopy_height_da, name="vegetation_height")
                mod._update_config_variable_name("vegetation_height", data_type="static")
                return
            elif source_type == "RasterDataset":
                canopy_height_da = mod.data_catalog.get_rasterdataset(
                    canopy_height,
                    geom=mod.region,
                    buffer=2,
                    time_tuple=(starttime, endtime),
                    variables=["canopy_height"]
                )
                logger.info(f"Retrieved canopy_height data from data catalog")
            else:
                raise ValueError(f"Invalid source type for canopy_height: {source_type}")
    
    if canopy_height_da is None:
        return
    
    if isinstance(canopy_height_da, xr.DataArray):
        canopy_height_da = canopy_height_da.astype("float32")
        
        if "time" in canopy_height_da.coords:
            logger.info("Processing time-varying canopy_height data for forcing")
            canopy_height_out = canopy_height_da.raster.reproject_like(
                mod.staticmaps.data["land_elevation"], method=reproj_method
            )
            mod.forcing.set(canopy_height_out, name="canopy_height")
            mod._update_config_variable_name("canopy_height", data_type="forcing")
        else:
            logger.info("Processing static canopy_height data for grid")
            canopy_height_out = canopy_height_da.raster.reproject_like(
                mod.staticmaps.data["land_elevation"], method=reproj_method
            )
            mod.staticmaps.set(canopy_height_out, name="vegetation_height")
            mod._update_config_variable_name("vegetation_height", data_type="static")
    else:
        raise ValueError(f"Invalid type for canopy_height: {type(canopy_height_da)}")


def _get_wind_processor():
    """Get reference to wind processing function to avoid name shadowing."""
    return wind


def setup_shortwave(
    mod,
    shortwave: Union[str, xr.DataArray, None],
    starttime: str,
    endtime: str,
    freq: pd.Timedelta,
    reproj_method: str = "nearest_index",
) -> None:
    """
    Setup shortwave radiation data for land surface temperature calculations.
    
    Parameters
    ----------
    mod : WflowModel
        Model instance
    shortwave : str, xr.DataArray, or None
        Shortwave source name, DataArray, or None
    starttime : str
        Start time for data retrieval
    endtime : str
        End time for data retrieval
    freq : pd.Timedelta
        Time frequency for resampling
    reproj_method : str, optional
        Reprojection method, by default "nearest_index"
    """
    if shortwave is None:
        return
    
    if isinstance(shortwave, str):
        if shortwave in mod.data_catalog.sources.keys():
            source_type = mod.data_catalog.get_source(shortwave).data_type
            if source_type == "RasterDataset":
                shortwave = mod.data_catalog.get_rasterdataset(
                    shortwave,
                    geom=mod.region,
                    buffer=2,
                    time_tuple=(starttime, endtime),
                    variables="shortwave_down"
                )
                logger.info(f"Retrieved shortwave radiation data from data catalog")
            else:
                raise ValueError(f"Shortwave source must be RasterDataset, got {source_type}")
        else:
            raise ValueError(f"Shortwave source '{shortwave}' not found in data catalog")
    
    if shortwave is None:
        return
    
    if isinstance(shortwave, xr.DataArray):
        shortwave = shortwave.astype("float32")
        shortwave_out = radiation(
            mod=mod,
            radiation=shortwave,
            var_name="shortwave_in",
            freq=freq,
            reproj_method=reproj_method,
        )
        # Store shortwave_in in memory for calculations but don't write to forcing
        mod.forcing.data["shortwave_in"] = shortwave_out
    else:
        raise ValueError(f"Invalid type for shortwave: {type(shortwave)}")


def setup_wind(
    mod,
    wind: Union[str, xr.DataArray, None],
    starttime: str,
    endtime: str,
    freq: pd.Timedelta,
    wind_altitude: float = 10.0,
    wind_altitude_correction: bool = False,
    reproj_method: str = "nearest_index",
) -> None:
    """
    Setup wind data for land surface temperature calculations.
    
    Parameters
    ----------
    mod : WflowModel
        Model instance
    wind : str, xr.DataArray, or None
        Wind source name, DataArray, or None
    starttime : str
        Start time for data retrieval
    endtime : str
        End time for data retrieval
    freq : pd.Timedelta
        Time frequency for resampling
    wind_altitude : float, optional
        Altitude of wind measurements [m], by default 10.0
    wind_altitude_correction : bool, optional
        Apply altitude correction, by default False
    reproj_method : str, optional
        Reprojection method, by default "nearest_index"
    """
    if wind is None:
        return
    
    wind_source = wind
    
    if isinstance(wind_source, str):
        if wind_source in mod.data_catalog.sources.keys():
            wind_u = mod.data_catalog.get_rasterdataset(
                wind_source,
                geom=mod.region,
                buffer=4,
                time_tuple=(starttime, endtime),
                variables=["wind10_u"]
            ).sel(time=slice(starttime, endtime))
            
            wind_v = mod.data_catalog.get_rasterdataset(
                wind_source,
                geom=mod.region,
                buffer=4,
                time_tuple=(starttime, endtime),
                variables=["wind10_v"]
            ).sel(time=slice(starttime, endtime))
            
            wind_u = wind_u.astype("float32")
            wind_v = wind_v.astype("float32")
            
            wind_processor = _get_wind_processor()
            wind_out = wind_processor(
                mod=mod,
                wind_u=wind_u,
                wind_v=wind_v,
                altitude=wind_altitude,
                altitude_correction=wind_altitude_correction,
                freq=freq,
                reproj_method=reproj_method,
            )
            mod.forcing.set(wind_out, name="wind")
            mod._update_config_variable_name("wind", data_type="forcing")
            logger.info("Wind data added to forcing and config updated")
            
            config_altitude = 2.0 if wind_altitude_correction else wind_altitude
            mod.config.set("input.wind_altitude", config_altitude)
        else:
            raise ValueError(f"Wind source '{wind_source}' not found in data catalog")
    else:
        raise ValueError(f"Wind must be a string source name, got {type(wind_source)}")


def setup_LST_forcing(
    mod,
    shortwave: Union[str, xr.DataArray, None] = None,
    wind: Union[str, xr.DataArray, None] = None,
    starttime: str = None,
    endtime: str = None,
    freq: pd.Timedelta = None,
    wind_altitude: float = 10.0,
    wind_altitude_correction: bool = False,
    reproj_method: str = "nearest_index",
) -> None:
    """
    Setup LST forcing variables: wind, shortwave, and net_radiation.
    
    Handles wind and shortwave setup, then calculates net radiation using
    available static or dynamic layers. Checks for time-varying canopy_height
    and raises NotImplementedError if found.
    
    Parameters
    ----------
    mod : WflowModel
        Model instance
    shortwave : str, xr.DataArray, or None, optional
        Shortwave source name or DataArray
    wind : str, xr.DataArray, or None, optional
        Wind source name or DataArray
    starttime : str, optional
        Start time for data retrieval
    endtime : str, optional
        End time for data retrieval
    freq : pd.Timedelta, optional
        Time frequency for resampling
    wind_altitude : float, optional
        Altitude of wind measurements [m], by default 10.0
    wind_altitude_correction : bool, optional
        Apply altitude correction, by default False
    reproj_method : str, optional
        Reprojection method, by default "nearest_index"
    """
    # Setup wind if provided
    if wind is not None:
        setup_wind(
            mod=mod,
            wind=wind,
            starttime=starttime,
            endtime=endtime,
            freq=freq,
            wind_altitude=wind_altitude,
            wind_altitude_correction=wind_altitude_correction,
            reproj_method=reproj_method,
        )
    
    # Setup shortwave if provided
    if shortwave is not None:
        setup_shortwave(
            mod=mod,
            shortwave=shortwave,
            starttime=starttime,
            endtime=endtime,
            freq=freq,
            reproj_method=reproj_method,
        )
    
    # Check for time-varying canopy_height
    if "canopy_height" in mod.forcing.data:
        raise NotImplementedError("Time-varying canopy_height in forcing is not yet implemented")
    
    # Check required variables
    required_vars = ["temp", "shortwave_in"]
    missing_vars = [var for var in required_vars if var not in mod.forcing.data]
    
    if missing_vars:
        logger.warning(f"Missing required variables for net radiation calculation: {missing_vars}")
        return
    
    # Get required variables
    temp = mod.forcing.data["temp"]
    shortwave_da = mod.forcing.data.get("shortwave_in")
    
    if shortwave_da is None:
        logger.warning("shortwave_in not found in forcing data, cannot calculate net radiation")
        return
    
    # Get latitude from grid
    if "lat" in mod.staticmaps.data.coords:
        latitude = mod.staticmaps.data["lat"]
    elif "latitude" in mod.staticmaps.data.coords:
        latitude = mod.staticmaps.data["latitude"]
    else:
        latitude = mod.staticmaps.data.raster.y_coords
    
    # Get emissivity if available (forcing or staticmaps)
    emissivity = None
    if "emissivity" in mod.forcing.data:
        emissivity = mod.forcing.data["emissivity"]
    elif "emissivity" in mod.staticmaps.data:
        emissivity = mod.staticmaps.data["emissivity"]
    
    # Get albedo if available (forcing, staticmaps, or albedo_lulc)
    albedo = None
    if "albedo" in mod.forcing.data:
        albedo = mod.forcing.data["albedo"]
    elif "albedo" in mod.staticmaps.data:
        albedo = mod.staticmaps.data["albedo"]
    elif "albedo_lulc" in mod.staticmaps.data:
        albedo = mod.staticmaps.data["albedo_lulc"]
    
    # Calculate net longwave radiation (intermediate product, not written to forcing)
    net_longwave = compute_net_longwave_radiation(
        air_temperature=temp,
        shortwave_radiation_in=shortwave_da,
        latitude=latitude,
        time_coord=temp.time,
        emissivity=emissivity
    )
    
    # Ensure time is first dimension for net_longwave (match temp dimension order)
    if "time" in net_longwave.dims and net_longwave.dims[0] != "time":
        target_dims = list(temp.dims)
        net_longwave = net_longwave.transpose(*target_dims)
    
    # Calculate net radiation if albedo is available
    if albedo is not None:
        net_radiation = compute_net_radiation(
            albedo=albedo,
            shortwave_radiation_in=shortwave_da,
            air_temperature=temp,
            latitude=latitude,
            time_coord=temp.time,
            emissivity=emissivity
        )
        
        # Ensure time is first dimension (match temp dimension order)
        if "time" in net_radiation.dims and net_radiation.dims[0] != "time":
            target_dims = list(temp.dims)
            net_radiation = net_radiation.transpose(*target_dims)
        
        source_msg = "calculated_from_albedo_temperature_shortwave"
        if emissivity is not None:
            source_msg += "_emissivity"
        net_radiation.attrs.update({"source": source_msg})
        mod.forcing.set(net_radiation, name="net_radiation")
        mod._update_config_variable_name("net_radiation", data_type="forcing")
        logger.info("Net radiation calculated and added to forcing")
    else:
        logger.info("Albedo not available, net radiation not calculated")
    
    # Remove intermediate variables from forcing (they're only needed for calculations)
    if "shortwave_in" in mod.forcing.data:
        del mod.forcing.data["shortwave_in"]
    
    # Ensure wind is in forcing
    if "wind" not in mod.forcing.data:
        logger.warning("Wind not found in forcing data")


def radiation(
    mod,
    radiation: xr.DataArray,
    var_name: str = "radiation",
    freq: Optional[str] = None,
    reproj_method: str = "nearest_index",
    resample_kwargs: Optional[dict] = None,
) -> xr.DataArray:
    """
    Process radiation data for land surface temperature calculations.
    
    Parameters
    ----------
    radiation : xr.DataArray
        Radiation data array [W m-2]
    da_model : xr.DataArray or xr.Dataset
        Target grid for reprojection
    var_name : str, optional
        Variable name for output, by default "radiation"
    freq : str, optional
        Resampling frequency, by default None
    reproj_method : str, optional
        Reprojection method, by default "nearest_index"
    resample_kwargs : dict, optional
        Additional resampling arguments, by default None
        
    Returns
    -------
    xr.DataArray
        Processed radiation data
    """
    resample_kwargs = resample_kwargs or {}
    
    if radiation.raster.dim0 != "time":
        raise ValueError(f'First radiation dim should be "time", not {radiation.raster.dim0}')
    
    # reproject to model grid
    radiation_out = radiation.raster.reproject_like(mod.staticmaps.data["land_elevation"], method=reproj_method)
    
    # ensure non-negative values
    radiation_out = np.fmax(radiation_out, 0)
    
    # resample time if requested
    radiation_out.name = var_name
    radiation_out.attrs.update(unit="W m-2")
    if freq is not None:
        resample_kwargs.update(upsampling="bfill", downsampling="mean")
        radiation_out = resample_time(radiation_out, freq, conserve_mass=False, **resample_kwargs)
    
    return radiation_out


def wind(
    mod,
    wind: Optional[xr.DataArray] = None,
    wind_u: Optional[xr.DataArray] = None,
    wind_v: Optional[xr.DataArray] = None,
    altitude: float = 10.0,
    altitude_correction: bool = False,
    freq: Optional[str] = None,
    reproj_method: str = "nearest_index",
    resample_kwargs: Optional[dict] = None,
) -> xr.DataArray:
    """
    Process wind data for land surface temperature calculations.
    
    Parameters
    ----------
    mod : WflowModel
        Wflow model instance
    wind : xr.DataArray
        Wind speed data array [m s-1]
    wind_u : xr.DataArray, optional
        U-component of wind [m s-1], by default None
    wind_v : xr.DataArray, optional
        V-component of wind [m s-1], by default None
    altitude : float, optional
        Altitude of wind measurements [m], by default 10.0
    altitude_correction : bool, optional
        Apply altitude correction to 2m, by default False
    freq : str, optional
        Resampling frequency, by default None
    reproj_method : str, optional
        Reprojection method, by default "nearest_index"
    resample_kwargs : dict, optional
        Additional resampling arguments, by default None
        
    Returns
    -------
    xr.DataArray
        Processed wind speed data [m s-1]
    """
    resample_kwargs = resample_kwargs or {}
    if (wind_u is None and wind_v is None) and (wind is None):
        raise ValueError("Either wind_u and wind_v or wind must be provided")
    # If wind components are provided, calculate wind speed
    if wind_u is not None and wind_v is not None:
        if wind_u.raster.dim0 != "time":
            raise ValueError(f'First wind_u dim should be "time", not {wind_u.raster.dim0}')
        if wind_v.raster.dim0 != "time":
            raise ValueError(f'First wind_v dim should be "time", not {wind_v.raster.dim0}')
        
        # reproject to model grid
        wind_u_out = wind_u.raster.reproject_like(mod.staticmaps.data["land_elevation"], method=reproj_method)
        wind_v_out = wind_v.raster.reproject_like(mod.staticmaps.data["land_elevation"], method=reproj_method)
        
        # calculate wind speed from components
        wind_out = np.sqrt(wind_u_out**2 + wind_v_out**2)
        wind_out.name = "wind_speed"
    else:
        # Use provided wind speed directly
        if wind.raster.dim0 != "time":
            raise ValueError(f'First wind dim should be "time", not {wind.raster.dim0}')
        
        # reproject to model grid
        wind_out = wind.raster.reproject_like(mod.staticmaps.data["land_elevation"], method=reproj_method)
        wind_out.name = "wind_speed"
    
    # Apply altitude correction if requested
    if altitude_correction and altitude != 2.0:
        # Simple logarithmic wind profile correction
        # wind_2m = wind_h * ln(2/z0) / ln(h/z0)
        # Using z0 = 0.1 m as typical surface roughness
        z0 = 0.1  # surface roughness length [m]
        wind_out = wind_out * np.log(2.0 / z0) / np.log(altitude / z0)
    
    wind_out = np.fmax(wind_out, 0)
    
    wind_out.attrs.update(unit="m s-1")
    if freq is not None:
        resample_kwargs.update(upsampling="bfill", downsampling="mean")
        wind_out = resample_time(wind_out, freq, conserve_mass=False, **resample_kwargs)
    
    return wind_out

