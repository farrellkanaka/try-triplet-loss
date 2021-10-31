import tensorflow as tf
from preprocessing import PreProcessing
from model import TripletLoss
import numpy as np
import tables as pt


# Create container
h5 = pt.open_file('myarray.h5', 'w')
filters = pt.Filters(complevel=6, complib='blosc')
carr = h5.create_carray('/', 'carray', atom=pt.Float32Atom(), shape=(1200, 500000), filters=filters)



