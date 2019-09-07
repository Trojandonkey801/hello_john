import h5py
import numpy
from decimal import Decimal

to_change = 'writeout.h5'

def quantize(weight):
    return numpy.round(weight,3)

def traverse_datasets(hdf_file):
    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    with h5py.File(hdf_file, 'r') as f:
        for path, _ in h5py_dataset_iterator(f):
            yield path

def read_file(file_name):
    with h5py.File(file_name, 'r') as f:
            for dset in traverse_datasets(file_name):
    #with h5py.File('resnet50_weights_tf_dim_ordering_tf_kernels_changed.h5', 'r') as f:
            #for dset in traverse_datasets('resnet50_weights_tf_dim_ordering_tf_kernels_changed.h5'):
                print('Path:', dset)
                print('Shape:', f[dset].shape)
                print('Data type:', f[dset].dtype)
                print('value:', f[dset][:])
                #for value in f[dset][:]:
                    #print('value',quantize(f[dset][value]))
                #print('value', quantize(f[dset][:]))


def traverse_and_change(hdf_file):
    def iterator_and_change(g, prefix = ''):
        for key in g.keys():
            item = g[key]
            if isinstance(item,h5py.Dataset):
                print("first",type(item[0]),item)
                item = quantize(item)
                print("second",type(item[0]),item)
            elif isinstance(item, h5py.Group): # test for group (go down)
               iterator_and_change(item)
    with h5py.File(hdf_file, 'r') as f:
        iterator_and_change(f)

with h5py.File(to_change) as f:
    traverse_and_change(to_change)

read_file(to_change)
