import astropy.coordinates as ac

def get_sun_zenith_angle(array_center,time):
    '''Return the solar zenith angle in degrees at the given time.'''
    frame = ac.AltAz(location=array_center.earth_location,obstime=time)
    sun = ac.get_sun(time).transform_to(frame)
    return 90. - sun.alt.deg
