from ctypes import c_int, c_int32, POINTER
from openmc.capi.filter import SpatialLegendreFilter, ZernikeFilter, SphericalHarmonicsFilter, CellFilter

expansion_types = (SpatialLegendreFilter, ZernikeFilter, SphericalHarmonicsFilter)

def results(tally, cell_id):
    filters = tally.filters
    if len(filters) != 2:
        raise("We expect there to be two filters, "
              "one a cell filter and the other an expansion filter")
    index_to_id = {}
    for key,value in openmc.capi.cells.items():
        index_to_id[value._index] = key
    if isinstance(filters[0], CellFilter):
        cells = filters[0].bins
        cell_ids = [index_to_id[cell_index] for cell_index in cells]
        if cell_id not in cell_ids:
            raise RuntimeError("Requested cell_id not in the passed tally")
        stride_integer = cell_ids.index(cell_id)
        if not isinstance(filters[1], expansion_types):
            raise TypeError("Expected an expansion filter "
                            "as the second filter")
        num_bins = filters[1].order + 1
        starting_point = num_bins * stride_integer
        return tally.mean[starting_point:starting_point+num_bins]
    elif isinstance(filters[0], expansion_types):
        num_bins = filters[0].order + 1
        if not isinstance(filters[1], CellFilter):
            raise TypeError("Expected a cell filter as the second filter")
        cells = filters[1].bins
        cell_ids = [index_to_id[cell_index] for cell_index in cells]
        if cell_id not in cell_ids:
            raise RuntimeError("Requested cell_id not in the passed tally")
        stride_integer = cell_ids.index(cell_id)
        total_bins = cells.size * num_bins
        return tally.mean[stride_integer:stride_integer+total_bins+cells.size:cells.size]
