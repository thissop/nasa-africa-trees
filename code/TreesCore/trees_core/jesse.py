from time import time
start = time()

from osgeo import gdal, ogr
gdal.UseExceptions()
ogr.UseExceptions()

gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")
gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
gdal.SetConfigOption("NUM_THREADS", "ALL_CPUS")

def compute_tree_annotation_and_boundary_raster(vector_fp):
    print(vector_fp)

    # NOTE(Jesse): Find an accompanying raster filename based on the vector_fp.
    vector_fp_split = vector_fp.split('/')
    vector_fn = vector_fp_split[-1]
    raster_fp_base = "/".join(vector_fp_split[:-1]) + "/"
    vector_fn = vector_fn.split('.')[0]
    v_id = vector_fn.split('_')[-1]
    raster_fp = ""
    if "arthur" in vector_fn:
        raster_fp = raster_fp_base + f"ndvi_arthur_training_area_{v_id}.tif"
    else:
        raster_fp = raster_fp_base + f"extracted_ndvi_{int(v_id) - 1}.png"

    raster_disk_ds = gdal.Open(raster_fp)

    # NOTE(Jesse): Create in memory raster of the same geospatial extents as the mask for high performance access.
    raster_mem_ds = gdal.GetDriverByName("MEM").Create('', xsize=raster_disk_ds.RasterXSize, ysize=raster_disk_ds.RasterYSize, bands=1, eType=gdal.GDT_Byte)
    band = raster_mem_ds.GetRasterBand(1)
    band.SetNoDataValue(255)
    raster_mem_ds.SetGeoTransform(raster_disk_ds.GetGeoTransform())
    raster_mem_ds.SetProjection(raster_disk_ds.GetProjection())
    band.Fill(0)
    del raster_disk_ds

    # NOTE(Jesse): Similarly with the vector polygons.  Load from disk and into a memory dataset.
    vector_disk_ds = gdal.OpenEx(vector_fp, gdal.OF_VECTOR)
    vector_mem_ds = gdal.GetDriverByName("Memory").Create('', 0, 0, 0, gdal.GDT_Unknown) #NOTE(Jesse): GDAL has a highly unintuitive API
    vector_mem_ds.CopyLayer(vector_disk_ds.GetLayer(0), 'orig')
    del vector_disk_ds

    # NOTE(Jesse): 'Buffer' extends the geometry out by the geospatial unit amount, approximating 'scaling' by 1.5.
    #             OGR, believe it or not, does not have an easy way to scale geometries like this.
    #             SQL is our only performant recourse to apply these operations to the data within OGR.
    sql_layer = vector_mem_ds.ExecuteSQL("select Buffer(GEOMETRY, 1.5, 5) from orig", dialect="SQLITE")
    vector_mem_ds.CopyLayer(sql_layer, 'scaled') #NOTE(Jesse): The returned 'layer' is not part of the original dataset for some reason? Requires a manual copy.
    del sql_layer

    # NOTE(Jesse): "Burn" the unscaled vector polygons into the raster image.
    opt_orig = gdal.RasterizeOptions(bands=[1], burnValues=1, layers='orig')
    gdal.Rasterize(raster_mem_ds, vector_mem_ds, options=opt_orig)

    # NOTE(Jesse): Track which pixels were burned into (via the '1') here, and reuse the band later.
    orig_arr = band.ReadAsArray()
    orig_arr_mask = orig_arr == 1
    band.Fill(0)

    # NOTE(Jesse): Burn the scaled geometries with the 'add' option, which will add the burn value to the destination pixel
    #             for all geometries which overlap it.  Basically, create a heatmap.
    opt_scaled = gdal.RasterizeOptions(bands=[1], burnValues=1, layers='scaled', add=True)
    gdal.Rasterize(raster_mem_ds, vector_mem_ds, options=opt_scaled)

    # NOTE(Jesse): Retain pixels with burn values > 1 (0 means no polygon overlap, 1 means 1 polygon overlaps, and >2 means multiple overlaps)
    composite_arr = band.ReadAsArray()
    composite_arr[composite_arr > 1] = 2 #NOTE(Jesse): 2 means overlap
    composite_arr[composite_arr == 1] = 0 #NOTE(Jesse): 0 means no polygon coverage
    composite_arr[orig_arr_mask] = 1 #NOTE(Jesse): 1 means original canopy

    # NOTE(Jesse): Save the composite array out to disk.
    raster_disk_ds = gdal.GetDriverByName("GTiff").Create(raster_fp_base + f"annotation_and_boundary_{v_id}.tif", xsize=raster_mem_ds.RasterXSize, ysize=raster_mem_ds.RasterYSize, bands=1, eType=gdal.GDT_Byte)
    raster_disk_ds.GetRasterBand(1).SetNoDataValue(255)
    raster_disk_ds.SetGeoTransform(raster_mem_ds.GetGeoTransform())
    raster_disk_ds.SetProjection(raster_mem_ds.GetProjection())
    raster_disk_ds.GetRasterBand(1).WriteArray(composite_arr)
    del raster_disk_ds

if __name__ == "__jesse__":
    from multiprocessing import Pool

    def main():
        from os import listdir

        #NOTE(Jesse): Find training data.  We can later normalize where they are and their naming conventions.

        arthur_training_data_fp = "/Users/jrmeyer3/Downloads/arthur_training_data/arthur/"
        arthur_training_files = listdir(arthur_training_data_fp)
        arthur_training_files = [arthur_training_data_fp + f for f in arthur_training_files if f.endswith(".gpkg") and "patch" not in f and "area" in f]

        thaddaeus_training_data_fp = "/Users/jrmeyer3/Downloads/vector_rectangles/"
        thaddaeus_training_files = listdir(thaddaeus_training_data_fp)
        thaddaeus_training_files = [thaddaeus_training_data_fp + f for f in thaddaeus_training_files if f.endswith(".gpkg")]

        training_files = (*arthur_training_files, *thaddaeus_training_files)

        with Pool() as p:
            p.map(compute_tree_annotation_and_boundary_raster, training_files, chunksize=1)

    main()

    end = time()
    elapsed_minutes = (end - start) / 60.0
    print(elapsed_minutes)