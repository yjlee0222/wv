% function setupClusterFeats(clusters,cnn_model,VOCopts,ids)
clear;

addpath('/home/yjlee/projects/weakVideo/misc/');
addpath('/home/yjlee/Downloads/VOCdevkit/VOCcode');
addpath('/home/yjlee/projects/weakVideo/videoSeg/');
addpath('/home/yjlee/projects/weakVideo/external/MeanShift/');

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

load([file_prefix '.mat'], 'clusters');
load([file_prefix '_cluster_feats.mat'], 'imgInfo','missed_ndx');
load([file_prefix '_cluster_matches.mat'], 'match_vals','boxes','frame_ndxs');

[match_vals, sorted_ndx] = sort(match_vals,2,'descend');
for ii=1:size(boxes,1)
    boxes(ii,:,:) = boxes(ii,sorted_ndx(ii,:),:);
end
frame_names = frame_names(frame_ndxs);
frame_bbox = voteOnVideoFrameBbox(frame_names,boxes,match_vals,datadir,subdir,resize_factor,PYRASTRIDE);

% skip patches whose deep pyramid feature could not be computed
imgInfo(missed_ndx) = [];
boxes(missed_ndx,:,:) = [];
sorted_ndx(missed_ndx,:) = [];
match_vals(missed_ndx,:) = [];
img_ids = [imgInfo.img_id];
unique_img_ids = unique(img_ids);
for ii=1:numel(unique_img_ids)    
    imgpath = sprintf(VOCopts.imgpath,ids{unique_img_ids(ii)});
    query_img = imread(imgpath);
    query_width = size(query_img,2);
    query_height = size(query_img,1);
    
    figure(1); clf; axis tight; axis off; 
    set(gca,'Position',[0 0 1 1]); % Make the axes occupy the whole figure
    imshow(query_img);

    this_img_ids = find(img_ids==unique_img_ids(ii));
    for jj=1:numel(this_img_ids)
        pyra_locs = imgInfo(this_img_ids(jj)).pyra_locs;
        query_scale = PYRASTRIDE/imgInfo(this_img_ids(jj)).pyra_scale;
        query_im_box = (pyra_locs-1)*query_scale+1; 
        query_weight = numel(clusters(imgInfo(this_img_ids(jj)).cluster_id).imNdx)/NUMTOPMATCHES;
        
%         figure(1); 
%         hold on;
%         rectangle('Position', [query_im_box(1) query_im_box(2) query_im_box(3)-query_im_box(1)+1 query_im_box(4)-query_im_box(2)+1], 'EdgeColor','c','LineWidth',3);
%         hold off;
        
        retrieved_bboxes = [];
        retrieved_bboxes_weight = [];
        for kk=1:50
            img = imread(frame_names{sorted_ndx(this_img_ids(jj),kk)}); 

            pyra_box = squeeze(boxes(this_img_ids(jj),kk,:))';      
            scale = PYRASTRIDE/pyra_box(end);
            im_box = (pyra_box(1:4)-1)*scale+1;

            if (pyra_box(4)-pyra_box(2) ~= pyra_locs(4)-pyra_locs(2)) || ...
                (pyra_box(3)-pyra_box(1) ~= pyra_locs(3)-pyra_locs(1))
                error('something wrong')
            end
            
%             figure(2); clf; axis tight; axis off; 
%             set(gca,'Position',[0 0 1 1]); % Make the axes occupy the whole figure
%             imshow(img); %title(match_vals(imNdx,ii));
%             rectangle('Position', [im_box(1) im_box(2) im_box(3)-im_box(1)+1 im_box(4)-im_box(2)+1], 'EdgeColor','c','LineWidth',3);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % get automatically computed bounding box from video frame
            % get bbox with most votes (for now; could change to
            % threshold-based and/or even prune bboxes with temporal consistency)
            [max_vote, max_ndx] = max(frame_bbox(sorted_ndx(this_img_ids(jj),kk)).tube_bbox_weight);
            tube_bbox = frame_bbox(sorted_ndx(this_img_ids(jj),kk)).tube_bbox(max_ndx,:);
            
            if isempty(tube_bbox)
                continue;
            end
                
            overlaps = computeOverlap(im_box, tube_bbox);
            for mm=1:size(tube_bbox,1)
                if overlaps(mm)>0.05
                    retrieved_bbox = query_im_box + (tube_bbox(mm,1:4)-im_box)*query_scale/scale;
                    retrieved_bboxes = [retrieved_bboxes; retrieved_bbox];
%                     retrieved_bboxes_weight = [retrieved_bboxes_weight; query_weight];
                    retrieved_bboxes_weight = [retrieved_bboxes_weight; match_vals(this_img_ids(jj),kk)];
                    
%                     figure(2);
%                     rectangle('Position', [tube_bbox(mm,1) tube_bbox(mm,2) tube_bbox(mm,3)-tube_bbox(mm,1)+1 tube_bbox(mm,4)-tube_bbox(mm,2)+1], 'EdgeColor','y','LineWidth',3);
%                     title(overlaps(mm));
%                     
%                     figure(1); 
%                     hold on;
%                     rectangle('Position', [query_im_box(1) query_im_box(2) query_im_box(3)-query_im_box(1)+1 query_im_box(4)-query_im_box(2)+1], 'EdgeColor','c','LineWidth',3);
%                     rectangle('Position', [retrieved_bbox(1) retrieved_bbox(2) retrieved_bbox(3)-retrieved_bbox(1)+1 retrieved_bbox(4)-retrieved_bbox(2)+1], 'EdgeColor','y','LineWidth',3);       
%                     hold off;
                else
%                     figure(2);
%                     rectangle('Position', [tube_bbox(mm,1) tube_bbox(mm,2) tube_bbox(mm,3)-tube_bbox(mm,1)+1 tube_bbox(mm,4)-tube_bbox(mm,2)+1], 'EdgeColor','w','LineWidth',3);
%                     title(overlaps(mm));
                end   
%                 pause;
            end
        end 
        % could do mean shift per retrieval (which would allow multiple dets
        % per image)
        bandWidth = 100;
        [clustCent,data2cluster,cluster2dataCell] = MeanShiftCluster(retrieved_bboxes',bandWidth);
        cluster_size = cellfun(@numel,cluster2dataCell,'UniformOutput',true);
        [max_cluster_size,max_cluster_ndx] = max(cluster_size);

        if ~isempty(clustCent)
            clustCent(1,:) = max(clustCent(1,:),1);
            clustCent(2,:) = max(clustCent(2,:),1);
            clustCent(3,:) = min(clustCent(3,:),size(query_img,2));
            clustCent(4,:) = min(clustCent(4,:),size(query_img,1));
        end
    
        figure(1); clf; axis tight; axis off; 
        set(gca,'Position',[0 0 1 1]); % Make the axes occupy the whole figure
        subplot(1,2,1); imshow(query_img); title(max_cluster_size)
        hold on;
        for kk=1:size(clustCent,2)
            if kk==max_cluster_ndx
                rectangle('Position', [clustCent(1,kk) clustCent(2,kk) clustCent(3,kk)-clustCent(1,kk)+1 clustCent(4,kk)-clustCent(2,kk)+1], 'EdgeColor','y','LineWidth',6);   
            else
                rectangle('Position', [clustCent(1,kk) clustCent(2,kk) clustCent(3,kk)-clustCent(1,kk)+1 clustCent(4,kk)-clustCent(2,kk)+1], 'EdgeColor','w','LineWidth',3,'LineStyle','--');               
            end
        end
        hold off;


        [clustCent,data2cluster,cluster2dataCell] = MeanShiftClusterWeightedPts(retrieved_bboxes',(retrieved_bboxes_weight.^10)',bandWidth);
        cluster_size = cellfun(@numel,cluster2dataCell,'UniformOutput',true);
        [max_cluster_size,max_cluster_ndx] = max(cluster_size);    

        if ~isempty(clustCent)
            clustCent(1,:) = max(clustCent(1,:),1);
            clustCent(2,:) = max(clustCent(2,:),1);
            clustCent(3,:) = min(clustCent(3,:),size(query_img,2));
            clustCent(4,:) = min(clustCent(4,:),size(query_img,1));
        end

        subplot(1,2,2); imshow(query_img); title(max_cluster_size)
        hold on;
        for kk=1:size(clustCent,2)
            if kk==max_cluster_ndx
                rectangle('Position', [clustCent(1,kk) clustCent(2,kk) clustCent(3,kk)-clustCent(1,kk)+1 clustCent(4,kk)-clustCent(2,kk)+1], 'EdgeColor','y','LineWidth',6);   
            else
                rectangle('Position', [clustCent(1,kk) clustCent(2,kk) clustCent(3,kk)-clustCent(1,kk)+1 clustCent(4,kk)-clustCent(2,kk)+1], 'EdgeColor','w','LineWidth',3,'LineStyle','--');               
            end
        end
        hold off;

        pause;
    end
    
    
end



