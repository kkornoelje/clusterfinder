 def make_frame(input_map):
    frame= core.G3Frame()
    frame['T'] = maps.FlatSkyMap(np.asarray(input_map,order='C'),
                      res=0.25*core.G3Units.arcmin,
                      weighted=False,
                      alpha_center=(352.5)*core.G3Units.deg,
                      delta_center=(-55)*core.G3Units.deg,
                      proj=maps.MapProjection.Proj0)
    return frame



