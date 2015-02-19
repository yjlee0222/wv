function param = setupParams(clsNdx,VOCopts)

param.basedir = '/home/SSD1/yjlee-data/projects/weakVideo/PASCAL2007/';
imgset = 'trainval';

load([param.basedir imgset 'class_pos_images.mat'], 'class_pos_images');

param.K = ceil(numel(class_pos_images(clsNdx).ndx)/2);
param.NUMCLUSTERS = 100;
param.NUMTOPMATCHES = ceil(param.K/2);
param.NUMBBOXOVERLAP = ceil(param.K/20);
param.BBOXOVERLAP = 0.5;

param.file_prefix = [param.basedir imgset '/' VOCopts.classes{clsNdx} '/' ...
    'maxNumBboxPerCluster_' num2str(param.NUMTOPMATCHES) '_numCluster_' num2str(param.NUMCLUSTERS) ...
    '_numBboxOverlap_' num2str(param.NUMBBOXOVERLAP) '_bboxOverlap_' num2str(param.BBOXOVERLAP)];

param.num_match = 20;
    
%     load([file_prefix '_pseudo_gt_num_match=' num2str(num_match) '_mirrored.mat'], 'pseudo_gt');    

