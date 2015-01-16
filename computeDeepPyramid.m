clear;

addpath('/home/yjlee/Downloads/caffe-master/matlab/caffe/');
addpath('/home/yjlee/Downloads/DeepPyramid/');

device_id = 1;
caffe('set_device', device_id);
cnn = init_cnn_model('use_gpu', true, 'use_caffe', true);

th = tic;
pyra = deep_pyramid(im, cnn);
pyra = deep_pyramid_add_padding(pyra, 0, 0);
fprintf('deep_pyramid took %.3fs\n', toc(th));
