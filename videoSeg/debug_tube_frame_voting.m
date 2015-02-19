clear;

addpath('/home/yjlee/projects/weakVideo/misc/');
addpath('/home/yjlee/projects/weakVideo/videoSeg/');
addpath('/home/yjlee/Downloads/VOCdevkit/VOCcode');

resize_factor = 1/4; %1/2
sample_rate = 8;
subdir = ['/shots_' num2str(resize_factor) '/OchsBroxMalik/Results/OchsBroxMalik' num2str(sample_rate) '_all_0000060.00/'];

VOCinit;
clsNdx = 7;
imgset = 'trainval';
ids = textread(sprintf(VOCopts.imgsetpath,imgset),'%s');

cls = VOCopts.classes{clsNdx};
frame_names = getFrames(cls);
datadir = ['/home/SSD1/yjlee-data/projects/weakVideo/YouTube-Objects/' cls '/data/'];

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
mirror = 1;
if mirror==0
    load([file_prefix '_cluster_matches.mat'], 'match_vals','boxes','frame_ndxs');
    frame_names = frame_names(frame_ndxs); 
    tic;
    frame = voteOnVideoFrameBbox(frame_names,boxes,match_vals,datadir,subdir,resize_factor,PYRASTRIDE,0);
    toc;
elseif mirror==1
    load([file_prefix '_cluster_matches_mirror.mat'], 'match_vals','boxes','frame_ndxs');
    frame_names = frame_names(frame_ndxs);
    tic;
    frame = voteOnVideoFrameBbox(frame_names,boxes,match_vals,datadir,subdir,resize_factor,PYRASTRIDE,1);
    toc;
end


for ii=1:numel(frame_names)
    img = imread(frame_names{ii}); 
    if mirror==1
        img = flipdim(img,2);
    end
    [~,max_ndx] = max(frame(ii).tube_bbox_weight);
    tube_bbox = frame(ii).tube_bbox;
    
    figure(1); clf; axis tight; axis off; 
    set(gca,'Position',[0 0 1 1]); % Make the axes occupy the whole figure
    imshow(img);       
    for jj=1:size(frame(ii).tube_bbox,1)  
        if jj==max_ndx
            rectangle('Position', [tube_bbox(jj,1) tube_bbox(jj,2) tube_bbox(jj,3)-tube_bbox(jj,1)+1 tube_bbox(jj,4)-tube_bbox(jj,2)+1], 'EdgeColor','c','LineWidth',3);
        else
            rectangle('Position', [tube_bbox(jj,1) tube_bbox(jj,2) tube_bbox(jj,3)-tube_bbox(jj,1)+1 tube_bbox(jj,4)-tube_bbox(jj,2)+1], 'EdgeColor','w','LineWidth',3);
        end
    end
    pause(0.2);
end


% clear frame;
% for ii=1:numel(frame_names)  
%     img = imread(frame_names{ii});
%     vote = zeros(size(img,1),size(img,2));
%     for jj=1:numel(imgInfo) % number of discriminative patches        
%         x1 = round(max(1,im_box(1)));
%         y1 = round(max(1,im_box(2)));
%         x2 = round(min(im_box(3),size(img,2)));
%         y2 = round(min(im_box(4),size(img,1)));
% 
%         vote(y1:y2,x1:x2) = vote(y1:y2,x1:x2) + match_vals(jj,ii);
%         figure(2); clf; axis tight; axis off; 
%         set(gca,'Position',[0 0 1 1]); % Make the axes occupy the whole figure
%         imshow(img); %title(match_vals(imNdx,ii));
%         rectangle('Position', [im_box(1) im_box(2) im_box(3)-im_box(1)+1 im_box(4)-im_box(2)+1], 'EdgeColor','c','LineWidth',3);
% 
%         pause;
%     end
% 
%     figure(1); clf; 
%     subplot(1,2,1); imagesc(img); title(ii); 
%     subplot(1,2,2); imagesc(vote); title(max(max(vote)));
%     pause(0.01);
%     frame(ii).vote = vote; 
% end
% 
% for ii=1:numel(frame_names)
%     img = imread(frame_names{ii}); 
%     
%     figure(1); clf; 
%     subplot(1,2,1); imagesc(img); title(ii); 
%     subplot(1,2,2); imagesc(frame(ii).vote); title(max(max(vote)));
%     pause(0.01);
% end
