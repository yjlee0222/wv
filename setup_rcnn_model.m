function rcnn_model = setup_rcnn_model(cnn_binary_file,cnn_definition_file,use_gpu)

% cwd = pwd;
addpath(genpath('/home/yjlee/Downloads/rcnn'));
% cd('/home/yjlee/Downloads/rcnn');

rcnn_model = rcnn_create_model(cnn_definition_file, cnn_binary_file, '');
    
rcnn_model.cnn.init_key = ...
    caffe('init', rcnn_model.cnn.definition_file, rcnn_model.cnn.binary_file);
if exist('use_gpu', 'var') && ~use_gpu
  caffe('set_mode_cpu');
else
  caffe('set_mode_gpu');
end
caffe('set_phase_test');
rcnn_model.cnn.layers = caffe('get_weights');

% cd(cwd);
