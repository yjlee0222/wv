% function setupClusterFeats(clusters,cnn_model,VOCopts,ids)
clear;

clsNdx = 3;
device_id = 1;

addpath('/home/yjlee/projects/weakVideo/misc/');
addpath('/home/yjlee/Downloads/caffe-master/matlab/caffe/');
addpath('/home/yjlee/Downloads/DeepPyramid/');
addpath('/home/yjlee/Downloads/VOCdevkit/VOCcode');

VOCinit;
imgset = 'trainval';
ids = textread(sprintf(VOCopts.imgsetpath,imgset),'%s');

cls = VOCopts.classes{clsNdx};
param = setupParams(clsNdx,VOCopts);
save_prefix = param.file_prefix;

g = gpuDevice(device_id+1);
caffe('set_device', device_id);
cnn_scale_7 = init_cnn_model('use_gpu', true, 'use_caffe', true);

init_params.MAXDIM = 12;
init_params.BUFFER = 3;
init_params.goal_ncells = 48;
    

if ~exist([save_prefix '_cluster_feats.mat'],'file')
    fprintf('computing cluster features for %s\n', cls);  
    
    load([save_prefix '.mat'], 'clusters');

    tot_instances = 0;
    for ii=1:numel(clusters)
        tot_instances = tot_instances + numel(clusters(ii).imNdx);
    end
%     tot_instances = numel(cat(1,clusters.imNdx));
    
    featMat = zeros((init_params.MAXDIM+init_params.BUFFER)^2*256, tot_instances, 'single');
    
    clear imgInfo;
    imgInfo(tot_instances).img_id = 0;
    imgInfo(tot_instances).box = zeros(1,4); 
    imgInfo(tot_instances).feat_size = zeros(1,3);    
    imgInfo(tot_instances).pyra_scale = 0;
    imgInfo(tot_instances).pyra_level = 0;
    imgInfo(tot_instances).pyra_locs = zeros(1,4); 
    imgInfo(tot_instances).missed = 0;
    imgInfo(tot_instances).cluster_id = 0;
    
    count = 1; 
    missed_ndx = []; 
    tic;
    for ii=1:numel(clusters)
        for jj=1:numel(clusters(ii).imNdx)
            img_id = clusters(ii).imNdx(jj);
            imgpath = sprintf(VOCopts.imgpath,ids{img_id});
            I = imread(imgpath); 

            bbox = clusters(ii).boxes(jj,:); % [x1 y1 x2 y2]

            model = initialize_goalsize_exemplar(I, bbox, cnn_scale_7, init_params);

            A = zeros(init_params.MAXDIM+init_params.BUFFER, init_params.MAXDIM+init_params.BUFFER, 256);
            A(1:model.feat_size(1), 1:model.feat_size(2), :) = model.feats;

            try
                featMat(:,count) = A(:);
                missed = 0;
            catch
                missed = 1;
                missed_ndx = [missed_ndx; count];
            end

            imgInfo(count).img_id = img_id;
            imgInfo(count).box = bbox;
            imgInfo(count).feat_size = model.feat_size;
            imgInfo(count).pyra_scale = model.pyra_scale;
            imgInfo(count).pyra_level = model.pyra_level;
            imgInfo(count).pyra_locs = model.pyra_locs; 
            imgInfo(count).missed = missed;
            imgInfo(count).cluster_id = ii;
            
            fprintf('done with cluster %d: %d/%d\n\n',ii, jj,numel(clusters(ii).imNdx));
            count = count + 1;
        end
    end
    fprintf('total time: %f\n',toc);
    fprintf('total missed: %d\n\n',numel(missed_ndx));
    
    save('-v7.3',[save_prefix '_cluster_feats.mat'], 'featMat','imgInfo','missed_ndx');
else
    fprintf([save_prefix 'cluster_feats.mat already exists\n\n']);    
end

% % if ~exist([save_prefix 'cluster_matches.mat'],'file')
%     fprintf('computing cluster matches for %s\n', cls); 
%     tt = tic;
%     
%     load([save_prefix '_cluster_feats.mat'], 'featMat','imgInfo','missed_ndx');
%     featMat = bsxfun(@times, featMat, 1./sqrt(sum(featMat.*featMat,1)));
%     featMat(:,missed_ndx) = 0;
%     
%     pyra_size = zeros(numel(imgInfo),2,'uint16');
%     for ii=1:numel(imgInfo)
%         pyra_size(ii,:) = imgInfo(ii).feat_size(1:2);          
%     end
%     h = init_params.MAXDIM+init_params.BUFFER;
%     w = init_params.MAXDIM+init_params.BUFFER;
% 
%     computeClusterMatches(featMat,pyra_size,h,w,cnn_scale_7,frame_names,save_prefix,0);
%     % match to mirrored frame
% %     computeClusterMatches(featMat,pyra_size,h,w,cnn_scale_7,frame_names,save_prefix,1);
%     
%     toc(tt);
% % else
% %     fprintf([save_prefix 'cluster_matches.mat already exists\n\n']);
% % end