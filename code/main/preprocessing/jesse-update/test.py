import os 

input_dir = '/mnt/c/Users/Research/Documents/GitHub/nasa-africa-trees/data/first_mosaic/rebuilt_approach/for-jesse-testing'
output_dir = '/mnt/c/Users/Research/Documents/GitHub/nasa-africa-trees/data/first_mosaic/rebuilt_approach/for-jesse-testing/output'

if input_dir[-1]!='/': 
    input_dir+='/'

if output_dir[-1]!='/':
    output_dir+='/'

annotations = []
boundaries = []
ndvi_images = []
pan_images = []
vector_rectangles = []

for file in os.listdir(input_dir):
    full_path = input_dir+file
    if '.png' in file: 
        if 'annotation' in file: 
            annotations.append(full_path)

        elif 'boundary' in file: 
            boundaries.append(full_path)

        elif 'ndvi' in file: 
            ndvi_images.append(full_path)

        elif 'extracted_pan' in file: 
            pan_images.append(full_path) 
        
    elif 'thaddaeus_vector_rectangle' in file: 
        vector_rectangles.append(full_path)

### MODIFICATION OF JESSE'S ROUTINE BELOW ### 

from osgeo import gdal, ogr
gdal.UseExceptions()
ogr.UseExceptions()

gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")
gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
gdal.SetConfigOption("NUM_THREADS", "ALL_CPUS")

def compute_tree_annotation_and_boundary_raster(vector_fp, raster_fp_base, raster_fp):

    r'''
    
    Arguments 
    ---------

    vector_fp : str
        The file containing the single "vector rectangle" that goes around the annotations region. 

    raster_fp_base : str
        Output path to save the combined annotation_boundary files 

    raster_fp : str
        An extracted ndvi image, e.g. currently follows format of extracted_ndvi_NUMBER.png
    '''

    if raster_fp_base[-1]!='/':
        raster_fp_base+='/'

    # NOTE(Jesse): Find an accompanying raster filename based on the vector_fp.
    vector_fp_split = vector_fp.split('/')
    vector_fn = vector_fp_split[-1]
    #raster_fp_base = "/".join(vector_fp_split[:-1]) + "/"
    vector_fn = vector_fn.split('.')[0]
    v_id = vector_fn.split('_')[-1]

    #raster_fp = raster_fp_base + f"extracted_ndvi_{int(v_id) - 1}.png" # FIX ! 

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

for i in range(len(vector_rectangles)):
    print('attempting run')
    compute_tree_annotation_and_boundary_raster(vector_rectangles[i], raster_fp_base=output_dir, raster_fp=ndvi_images[i])