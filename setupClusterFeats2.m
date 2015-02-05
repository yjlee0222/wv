% function setupClusterFeats(clusters,cnn_model,VOCopts,ids)
clear;

addpath('/home/yjlee/projects/weakVideo/misc/');
addpath('/home/yjlee/Downloads/caffe-master/matlab/caffe/');
addpath('/home/yjlee/Downloads/DeepPyramid/');
addpath('/home/yjlee/Downloads/VOCdevkit/VOCcode');

VOCinit;
clsNdx = 7;
imgset = 'trainval';
ids = textread(sprintf(VOCopts.imgsetpath,imgset),'%s');

cls = VOCopts.classes{clsNdx};
frame_names = getFrames(cls);

basedir = '/home/SSD1/yjlee-data/projects/weakVideo/PASCAL2007/';
load([basedir imgset 'class_pos_images.mat'], 'class_pos_images');
K = ceil(numel(class_pos_images(clsNdx).ndx)/2);
NUMCLUSTERS = 100;
NUMTOPMATCHES = ceil(K/2);
NUMBBOXOVERLAP = ceil(K/20);
BBOXOVERLAP = 0.5;

init_params.MAXDIM = 12;
    
save_prefix = [basedir imgset '/' VOCopts.classes{clsNdx} '/' ...
    'maxNumBboxPerCluster_' num2str(NUMTOPMATCHES) '_numCluster_' num2str(NUMCLUSTERS) ...
    '_numBboxOverlap_' num2str(NUMBBOXOVERLAP) '_bboxOverlap_' num2str(BBOXOVERLAP)];

if ~exist([save_prefix '_cluster_feats_resize.mat'],'file')
    g = gpuDevice(1);
    device_id = 0;
    caffe('set_device', device_id);
    cnn_scale_1 = init_cnn_model_batch('use_gpu', true, 'use_caffe', true);

    load([save_prefix '.mat'], 'clusters');
    tot_instances = 0;
    for ii=1:numel(clusters)
        tot_instances = tot_instances + numel(clusters(ii).imNdx);
    end
    featMat = zeros((init_params.MAXDIM)^2*256, tot_instances, 'single');
    
    clear imgInfo;
    imgInfo(tot_instances).img_id = 0;
    imgInfo(tot_instances).box = zeros(1,4);
    imgInfo(tot_instances).feat_size = zeros(1,2); 
    count = 1;
    
    imgCount = 0;
    img_ids = zeros(256,1);
    boxes = zeros(256,4);
    imgs = cell(1,256);
    imgs(:) = {zeros(177,177,3,'uint8')};
    
    tic;
    for ii=1:numel(clusters)
        for jj=1:numel(clusters(ii).imNdx)
            imgCount = imgCount + 1;
            
            img_id = clusters(ii).imNdx(jj);
            imgpath = sprintf(VOCopts.imgpath,ids{img_id});
            I = imread(imgpath); 

            bbox = clusters(ii).boxes(jj,:);
            x1 = max(1,bbox(1)); y1 = max(1,bbox(2)); 
            x2 = min(size(I,2),bbox(3)); y2 = min(size(I,1),bbox(4));

            h = y2-y1+1;
            w = x2-x1+1;
            if h>w 
                imgs{imgCount} = imresize(I(y1:y2,x1:x2,:), [177 w*(177/h)]);
            else
                imgs{imgCount} = imresize(I(y1:y2,x1:x2,:), [h*(177/w) 177]);
            end            
            img_ids(imgCount) = img_id;
            boxes(imgCount,:) = bbox;
            
            if (imgCount==256) || (ii==numel(clusters) && jj==numel(clusters(ii).imNdx))
                query_pyra = deep_pyramid_batch(imgs, cnn_scale_1);
                query_pyra = deep_pyramid_add_padding(query_pyra, 0, 0);
                
                for kk=1:imgCount
                    A = zeros(init_params.MAXDIM, init_params.MAXDIM, 256, 'single');
                    A(1:query_pyra.level_sizes(kk,1), 1:query_pyra.level_sizes(kk,2), :) = query_pyra.feat{kk};
                    featMat(:,count) = A(:);
                    
                    imgInfo(count).img_id = img_ids(kk);
                    imgInfo(count).box = boxes(kk,:);
                    imgInfo(count).feat_size = query_pyra.level_sizes(kk,:);
                
                    count = count + 1;
                end             
            
                imgCount = 0;
                img_ids = zeros(256,1);
                boxes = zeros(256,4);
                imgs(:) = {zeros(177,177,3,'uint8')};    
            end                    
        end
    end
    fprintf('total time: %f\n',toc);
   
    save('-v7.3',[save_prefix '_cluster_feats_resize.mat'], 'featMat','imgInfo');
else
    fprintf([save_prefix 'cluster_feats_resize.mat already exists\n\n']);    
end

if ~exist([save_prefix 'cluster_matches_resize.mat'],'file')
    load([save_prefix '_cluster_feats_resize.mat'], 'featMat','imgInfo');
    featMat = bsxfun(@times, featMat, 1./sqrt(sum(featMat.*featMat,1)));
    
    pyra_size = zeros(numel(imgInfo),2,'uint16');
    for ii=1:numel(imgInfo)
        pyra_size(ii,:) = imgInfo(ii).feat_size(1:2);          
    end
    h = init_params.MAXDIM;
    w = init_params.MAXDIM;

    g = gpuDevice(1);
    device_id = 0;
    caffe('set_device', device_id);
    cnn_scale_7 = init_cnn_model('use_gpu', true, 'use_caffe', true);

    computeClusterMatches2(featMat,pyra_size,h,w,cnn_scale_7,frame_names,save_prefix);
else
    fprintf([save_prefix 'cluster_matches_resize.mat already exists\n\n']);
end