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
PYRASTRIDE = 16;

file_prefix = [basedir imgset '/' VOCopts.classes{clsNdx} '/' ...
    'maxNumBboxPerCluster_' num2str(NUMTOPMATCHES) '_numCluster_' num2str(NUMCLUSTERS) ...
    '_numBboxOverlap_' num2str(NUMBBOXOVERLAP) '_bboxOverlap_' num2str(BBOXOVERLAP)];
load([file_prefix '_cluster_feats.mat'], 'imgInfo','missed_ndx');
load([file_prefix '_cluster_matches.mat'], 'match_vals','match_ndxs','boxes','frame_ndxs');

[match_vals, sorted_ndx] = sort(match_vals,2,'descend');
for ii=1:size(match_ndxs,1)
    match_ndxs(ii,:) = match_ndxs(ii,sorted_ndx(ii,:));
    boxes(ii,:,:) = boxes(ii,sorted_ndx(ii,:),:);
end
frame_names = frame_names(frame_ndxs);

for imNdx = 1:numel(imgInfo)
    img_id = imgInfo(imNdx).img_id;
    imgpath = sprintf(VOCopts.imgpath,ids{img_id});
    I = imread(imgpath);
    this_box = imgInfo(imNdx).box;
    x1 = this_box(1); y1 = this_box(2); x2 = this_box(3); y2 = this_box(4);

    figure(1); clf;    
    axis tight; axis off; 
    imshow(I); title('red: original, cyan: adjusted to deep pyramid');
    hold on;
    rectangle('Position', [x1 y1 x2-x1+1 y2-y1+1], 'EdgeColor', 'r');

    pyra_locs = imgInfo(imNdx).pyra_locs;
    scale = PYRASTRIDE/imgInfo(imNdx).pyra_scale;
    im_box = (pyra_locs-1)*scale+1;
    rectangle('Position', [im_box(1) im_box(2) im_box(3)-im_box(1)+1 im_box(4)-im_box(2)+1], 'EdgeColor', 'c');
    hold off;
          
    pyra_width = [pyra_locs(3)-pyra_locs(1)];
    pyra_height = [pyra_locs(4)-pyra_locs(2)];
        
    figure(2); clf;
    for ii=1:12
        im = imread(frame_names{sorted_ndx(imNdx,ii)}); 
        pyra_box = squeeze(boxes(imNdx,ii,:))'; 
     
        scale = PYRASTRIDE/pyra_box(end);
        im_box = ([pyra_box(1:2) pyra_box(1)+pyra_width pyra_box(2)+pyra_height]-1)*scale+1;
        subplot(3,4,ii); 
        imshow(im); title(match_vals(imNdx,ii));
        rectangle('Position', [im_box(1) im_box(2) im_box(3)-im_box(1)+1 im_box(4)-im_box(2)+1], 'EdgeColor', 'r');
    end
    pause;
end



