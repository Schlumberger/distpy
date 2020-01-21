# (C) 2020, Schlumberger. Refer to LICENSE
#
# unit_handler.py
#
# NOTE: units should only be used by plots, we allow an assumption of
#       metric everywhere for all calculations and processing.


from numba import vectorize, float64


#UNITS FOR DISPLAY
@vectorize([float64(float64)],)
def M_TO_FT(val):
    return 3.28084*val
@vectorize([float64(float64)])
def FT_TO_M(val):
    return val/3.28084

# NOTE: approximate temperature conversion
#        don't go C_TO_F(F_TO_C(C_TO_F())) etc. as there will be drift
@vectorize([float64(float64)])
def C_TO_F(val):
    return (1.4*val)+32.0
@vectorize([float64(float64)])
def F_TO_C(val):
    return (5.0/9.0)*(val-32.0)
