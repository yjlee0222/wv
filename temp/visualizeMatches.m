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

cluster_ndx = 1;
group_ndx = 1;

basedir = '/home/SSD1/yjlee-data/projects/weakVideo/PASCAL2007/';
load([basedir imgset 'class_pos_images.mat'], 'class_pos_images');
K = ceil(numel(class_pos_images(clsNdx).ndx)/2);
NUMCLUSTERS = 100;
NUMTOPMATCHES = ceil(K/2);
NUMBBOXOVERLAP = ceil(K/20);
BBOXOVERLAP = 0.5;

load([basedir imgset '/' VOCopts.classes{clsNdx} '/' ...
    'maxNumBboxPerCluster_' num2str(NUMTOPMATCHES) '_numCluster_' num2str(NUMCLUSTERS) ...
    '_numBboxOverlap_' num2str(NUMBBOXOVERLAP) '_bboxOverlap_' num2str(BBOXOVERLAP) '.mat'], ...
    'clusters');
group_size = 25;
clusters = initializeClustersForMatching(clusters,group_size);
    
device_id = 1;
caffe('set_device', device_id);
cnn_scale_1 = init_cnn_model_batch('use_gpu', true, 'use_caffe', true);
[query_pyra,imgs] = setupClusterFeats(clusters,1,1,cnn_scale_1,VOCopts,ids);
h = query_pyra.level_sizes(1,1);
w = query_pyra.level_sizes(1,2);

save_dir = [basedir imgset '/' cls '/cluster_to_frame_matches/'];
load([save_dir 'cluster' num2str(cluster_ndx) '_group' num2str(group_ndx) '.mat'], ...
        'matchVals','matchNdxs');
[matchVals, sorted_ndx] = sort(matchVals,2,'descend');

cnn_scale_7 = init_cnn_model('use_gpu', true, 'use_caffe', true);

for imNdx = 1:100
    this_ndx = clusters(1).group(1).cluster_instances(imNdx);
    img_id = clusters(1).imNdx(this_ndx);
    imgpath = sprintf(VOCopts.imgpath,ids{img_id});
    I = imread(imgpath);
    thisBox = clusters(1).boxes(this_ndx,:);
    x1 = thisBox(1); y1 = thisBox(2); x2 = thisBox(3); y2 = thisBox(4);

    figure(1); clf;
    axis tight; axis off; 
    imshow(I);
    rectangle('Position', [x1 y1 x2-x1+1 y2-y1+1], 'EdgeColor', 'r');

    matchVal = matchVals(imNdx,:);
    matchNdx = matchNdxs(imNdx,:);
    matchNdx = matchNdx(sorted_ndx(imNdx,:));
   
    figure(2); clf;
    for ii=1:12
        im = imread(frame_names{sorted_ndx(imNdx,ii)}); 
        pyra = deep_pyramid(im, cnn_scale_7);
        pyra = deep_pyramid_add_padding(pyra, 0, 0);
        pyra = pyramid2Mat(pyra,h,w,1);

        boxes = [pyra.featPos(2,matchNdx(ii)) pyra.featPos(1,matchNdx(ii)) ...
            pyra.featPos(2,matchNdx(ii))+w-1 pyra.featPos(1,matchNdx(ii))+h-1 pyra.featLevel(matchNdx(ii))];  
        im_boxes = pyra_to_im_coords(pyra, boxes);

        subplot(3,4,ii); 
        imshow(im); title(matchVal(ii));
        rectangle('Position', [im_boxes(1) im_boxes(2) im_boxes(3)-im_boxes(1)+1 im_boxes(4)-im_boxes(2)+1], 'EdgeColor', 'r');
    end
    pause;
end

% for ii=1:numel(frame_names)
%     im = imread(frame_names{ii});
%     
% %     th = tic;   
%     pyra = deep_pyramid(im, cnn_model);
%     pyra = deep_pyramid_add_padding(pyra, 0, 0);
% %     fprintf('deep_pyramid took %.3fs\n', toc(th));
% %     th = tic;
%     pyra = pyramid2Mat(pyra,h,w,1);
% %     fprintf('pyramid2mat took %.3fs\n', toc(th));
% 
%     D = query_pyra.Feats'*pyra.featMat;
%     [matchVal,matchNdx] = max(D,[],2);
% 
%     matchVals(:,ii) = matchVal;
%     matchNdxs(:,ii) = uint16(matchNdx);
%     
%     if mod(ii,100)==1
% %         save([save_dir 'cluster' num2str(cluster_ndx) '_group' num2str(group_ndx) '_chunk' num2str(chunk_count) '.mat'], ...
% %         'matchVals','matchNdxs');
%     
%         fprintf('done with %d/%d\n\n',ii,numel(frame_names));        
%         hrs_left = (numel(frame_names)-ii)*(toc(t1)/ii)/60/60;
%         fprintf('estimated hrs left: %f\n',hrs_left);
%     end
% end


