import os
import numpy
import iris
import pandas
import xarray
import cartopy.crs as ccrs
from iris_grib._load_convert import unscale, ellipsoid_geometry, ellipsoid
import iris.coord_systems as icoord_systems

_level_type_mappings = {
    "meanSea": (101, 255),
    "hybrid": (105, 255),
    "atmosphere": (10, 255),
    "atmosphereSingleLayer": (200, 255),
    "surface": (1, 255),
    "isobaricInhPa": (100, 255),
    "heightAboveGround": (103, 255),
    "depthBelowLandLayer": (106, 106),
    "heightAboveSea": (102, 255),
    "isobaricLayer": (100, 100),
    "nominalTop": (8, 255),
    "heightAboveGroundLayer": (103, 103),
    "tropopause": (7, 255),
    "maxWind": (6, 255),
    "isothermZero": (4, 255),
    "pressureFromGroundLayer": (108, 108),
    "sigmaLayer": (104, 104),
    "sigma": (104, 255),
    "theta": (107, 255),
    "potentialVorticity": (109, 255),
}

_field_keyval_mapping = {
    "air_temperature_isobaric":{
        "discipline":0,
        "category":0,
        "number":0,
        "type_of_level":"isobaricInhPa",
        "product_definition_template":0
    },
    "specific_humidity_isobaric":{
        "discipline":0,
        "category":1,
        "number":0,
        "type_of_level":"isobaricInhPa",
        "product_definition_template":0
    },
    "geopotential_height_isobaric":{
        "discipline":0,
        "category":3,
        "number":5,
        "type_of_level":"isobaricInhPa",
        "product_definition_template":0
    },
    "u_wind_isobaric":{
        "discipline":0,
        "category":2,
        "number":2,
        "type_of_level":"isobaricInhPa",
        "product_definition_template":0
    },
    "v_wind_isobaric":{
        "discipline":0,
        "category":2,
        "number":3,
        "type_of_level":"isobaricInhPa",
        "product_definition_template":0
    },
    "latitude":{
        "discipline":0,
        "category":191,
        "number":192,
        "type_of_level":"surface",
        "product_definition_template":0
    },
    "longitude":{
        "discipline":0,
        "category":191,
        "number":193,
        "type_of_level":"surface",
        "product_definition_template":0
    },
    "u_wind_surface":{  #U-component of winds at 10m AGL (m/s)
        "discipline":0,
        "category":2,
        "number":2,
        "type_of_level":"heightAboveGround",
        "product_definition_template":0,
        "level":10
    },
    "v_wind_surface":{  #V-component of winds at 10m AGL (m/s)
        "discipline":0,
        "category":2,
        "number":3,
        "type_of_level":"heightAboveGround",
        "product_definition_template":0,
        "level":10
    },
    "pressure_surface":{    #surface pressure (Pa)
        "discipline":0,
        "category":3,
        "number":0,
        "type_of_level":"surface",
        "product_definition_template":0
    },
    "air_temperature_surface":{ #air temperature 2-meters above ground
        "discipline":0,
        "category":0,
        "number":0,
        "type_of_level":"heightAboveGround",
        "product_definition_template":0,
        "level":2
    },
    "dewpoint_temperature_surface":{ #dew point temperature 2-meters above ground
        "discipline":0,
        "category":0,
        "number":6,
        "type_of_level":"heightAboveGround",
        "product_definition_template":0,
        "level":2
    },
    "terrain_height_surface":{  #Height (above sea-level) of the model terrain
        "discipline":0,
        "category":3,
        "number":5,
        "type_of_level":"surface",
        "product_definition_template":0
    },
}

def get_level_type(type_of_first_fixed_surface,type_of_second_fixed_surface):
    for key,val in _level_type_mappings.items():
        surface_values = (type_of_first_fixed_surface,type_of_second_fixed_surface)
        if surface_values == val:
            level_type = key
            break
        else:
            level_type = None

    return level_type

def _get_field_info_dict(message):

    sections = message.sections
    discipline = sections[0]["discipline"]
    parameter_category = sections[4]["parameterCategory"]
    parameter_number = sections[4]["parameterNumber"]
    product_definition_template = sections[4]["productDefinitionTemplateNumber"]
    type_of_first_fixed_surface = sections[4]["typeOfFirstFixedSurface"]
    scale_factor_of_first_fixed_surface = sections[4]["scaleFactorOfFirstFixedSurface"]
    scaled_value_of_first_fixed_surface = sections[4]["scaledValueOfFirstFixedSurface"]
    type_of_second_fixed_surface = sections[4]["typeOfSecondFixedSurface"]
    scale_factor_of_second_fixed_surface = sections[4]["scaleFactorOfSecondFixedSurface"]
    scaled_value_of_second_fixed_surface = sections[4]["scaledValueOfSecondFixedSurface"]

    #If needed, the grid resolution can be pulled out and included in the returned
    # meta dictionary. This is possible because all matching is done based on keys from
    # the mapping dictionary (not the meta-data dictionary returned here)
    #dx,dy = sections[3]['Dx'],sections[3]['Dy']

    level_type = get_level_type(type_of_first_fixed_surface,type_of_second_fixed_surface)

    if type_of_first_fixed_surface != 255 and type_of_second_fixed_surface == 255:
        level = unscale(scaled_value_of_first_fixed_surface,scale_factor_of_first_fixed_surface)
    else:
        level = None
    
    return {"discipline":discipline,
            "category":parameter_category,
            "number":parameter_number,
            "product_definition_template":product_definition_template,
            "type_of_level":level_type,
            "level":level
            }

def _get_grid_coordinate_reference_system(message):

    _GRID_ACCURACY_IN_DEGREES = 1e-6

    section = message.sections[3]
    major, minor, radius = ellipsoid_geometry(section)
    geog_cs = ellipsoid(section['shapeOfTheEarth'], major, minor, radius)
    central_latitude = section['LaD'] * _GRID_ACCURACY_IN_DEGREES
    central_longitude = section['LoV'] * _GRID_ACCURACY_IN_DEGREES
    false_easting = 0
    false_northing = 0
    secant_latitudes = (section['Latin1'] * _GRID_ACCURACY_IN_DEGREES,
                        section['Latin2'] * _GRID_ACCURACY_IN_DEGREES)

    cs = icoord_systems.LambertConformal(central_latitude,
                                        central_longitude,
                                        false_easting,
                                        false_northing,
                                        secant_latitudes=secant_latitudes,
                                        ellipsoid=geog_cs).as_cartopy_crs()
    

    return cs


def _add_field_metadata(cube,message,filename):
    cube.attributes.update(
            {
                "grib2_meta": _get_field_info_dict(message),
                "crs":_get_grid_coordinate_reference_system(message),
            }

        )

    for name,mapping in _field_keyval_mapping.items():
        field_info = {k:v for k,v in cube.attributes['grib2_meta'].items() if k in mapping.keys()}
        if field_info == mapping:
            cube.rename(name)

    return cube


def load_iris_cubes(grib_filename,field_list = _field_keyval_mapping.keys()):

    raw_cubes = iris.load(grib_filename,field_list,callback=_add_field_metadata)
    return list(raw_cubes)

def load_grib(filename):
    cubes = load_iris_cubes(filename)
    data_array_list = [xarray.DataArray.from_iris(c) for c in cubes]
    isobaric_da_list,surface_da_list, coord_da_list = _split_data_array_by_level_type(data_array_list)

    coord_ds = xarray.merge(coord_da_list)

    isobaric_ds = _create_dataset(isobaric_da_list,coord_ds,['forecast_period'])
    isobaric_ds['pressure'] = isobaric_ds['pressure']/100.
    surface_ds = _create_dataset(surface_da_list,coord_ds,['forecast_period','height'])
    
    isobaric_ds = _rename_variables(isobaric_ds)
    surface_ds = _rename_variables(surface_ds)

    return isobaric_ds,surface_ds

def _create_dataset(data_arrays,coord_ds,drop_vars):
    data_arrays = [
        da.drop_vars(
            drop_vars,
            errors="ignore"
        ).assign_coords({
            'time':da['time'],
            'forecast_reference_time':da['forecast_reference_time'],
            'latitude':coord_ds['latitude'],
            'longitude':coord_ds['longitude']
        }) for da in data_arrays
    ]

    ds = xarray.combine_by_coords(
        data_arrays,
        combine_attrs='drop_conflicts'
    )
 
    return ds
    

def _split_data_array_by_level_type(data_array_list):
    isobaric_data_array_list = [x.expand_dims('pressure') for x in data_array_list if "isobaric" in x.name]
    surface_data_array_list = [x.reset_coords() for x in data_array_list if "surface" in x.name]
    coordinate_data_array_list = [x.reset_coords() for x in data_array_list if x.name in ['latitude','longitude']]
    return isobaric_data_array_list,surface_data_array_list,coordinate_data_array_list


def _rename_variables(inds):
    '''Renames fields in a given dataset to a single set of standard names
    that make matching datasets simple.

    Args:
        inds (xarray.Dataset): dataset containing any number of meteorological fields
    Returns:
        (xarray.Dataset): dataset with fields renamed to standard names
    '''

    rename_dict = {
        'air_temperature_surface':'temperature',
        'dewpoint_temperature_surface':'dewpoint',
        'u_wind_surface':'uwind',
        'v_wind_surface':'vwind',
        'pressure_surface':'pressure',
        'terrain_height_surface':'height',
        'specific_humidity_isobaric':'specific_humidity',
        'air_temperature_isobaric':'temperature',
        'u_wind_isobaric':'uwind',
        'v_wind_isobaric':'vwind',
        'geopotential_height_isobaric':'height'

    }
    dataset_variables = list(inds.data_vars)
    rename_dict = {key:value for key,value in rename_dict.items() if key in dataset_variables}
    return inds.rename(rename_dict)