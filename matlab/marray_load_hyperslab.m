function M = marray_load_hyperslab(file_name, dataset_name, start, slabsize)
%
% loads a slab from an HDF5 dataset.
% the first coordinate is 'start'.
% the shape is 'slabsize'
%
offset = start(end:-1:1)-1;
slabsize = slabsize(end:-1:1);
file = H5F.open(file_name, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');
dataset = H5D.open(file, dataset_name);
dataspace = H5D.get_space(dataset);
stride = ones(1, numel(offset));
blocksize = ones(1, numel(offset));
H5S.select_hyperslab(dataspace, 'H5S_SELECT_SET', offset, ...
    stride, slabsize, blocksize);
memspace = H5S.create_simple(length(slabsize), slabsize, slabsize);
M = H5D.read(dataset, 'H5ML_DEFAULT', memspace, dataspace, 'H5P_DEFAULT');
H5S.close(dataspace);
H5S.close(memspace);
H5D.close(dataset);
H5F.close(file);
%
end