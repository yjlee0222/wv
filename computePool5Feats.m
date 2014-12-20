clear;

addpath('/home/yjlee/Downloads/VOCdevkit/VOCcode');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% setup caffe, rcnn (use caffenet model, not fine-tuned)
cnn_binary_file = '/home/yjlee/Downloads/caffe-master/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';
cnn_definition_file = '/home/yjlee/projects/weakVideo/data/pool5.prototxt';
rcnn_model_pool5 = setup_rcnn_model(cnn_binary_file,cnn_definition_file,true);
% cnn_definition_file = '/home/yjlee/projects/weakVideo/data/fc7.prototxt';
% rcnn_model_fc7 = setup_rcnn_model(cnn_binary_file,cnn_definition_file,true);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% initialize VOC options
VOCinit;

% % car = 7
% clsNdx = 7;
% cls = VOCopts.classes{clsNdx};

imgset = 'trainval';
ids = textread(sprintf(VOCopts.imgsetpath,imgset),'%s');

savedir = '/home/SSD1/yjlee-data/projects/weakVideo/PASCAL2007/trainval/pool5/';
if ~exist(savedir,'dir')
    mkdir(savedir);
end

for ii=1:numel(ids)
    savename = [savedir ids{ii} '.mat'];
    if exist(savename,'file')
        continue;
    end
    imgpath = sprintf(VOCopts.imgpath,ids{ii});
%     rec = PASreadrecord(sprintf(VOCopts.annopath,ids{ii}));
    im = imread(imgpath);
    
    [boxes,feat] = computeSelectiveSearchDeepFeats(im, rcnn_model_pool5);
%     feat = rcnn_pool5_to_fcX(feat, 7, rcnn_model_fc7);
      
    save(savename, 'boxes','feat');
    
    fprintf('done with im %d/%d\n',ii,numel(ids));
end




