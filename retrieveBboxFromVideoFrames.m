clear;

addpath('/home/yjlee/projects/weakVideo/misc/');
addpath('/home/yjlee/Downloads/VOCdevkit/VOCcode');
addpath('/home/yjlee/projects/weakVideo/videoSeg/');
addpath('/home/yjlee/projects/weakVideo/external/MeanShift/');

resize_factor = 1/4; %1/2
sample_rate = 8;
subdir = ['/shots_' num2str(resize_factor) '/OchsBroxMalik/Results/OchsBroxMalik' num2str(sample_rate) '_all_0000060.00/'];
bandWidth = 100;
num_match = 20;

VOCinit;
clsNdx = 7;
imgset = 'trainval';
ids = textread(sprintf(VOCopts.imgsetpath,imgset),'%s');

cls = VOCopts.classes{clsNdx};
frame_names = getFrames(cls);
datadir = ['/home/SSD1/yjlee-data/projects/weakVideo/YouTube-Objects/' cls '/data/'];

clear pseudo_gt;
pseudo_gt.bbox = cell(1,numel(ids));
pseudo_gt.cluster_size = zeros(1,numel(ids));
pseudo_gt.cls = cls;

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load cluster matches original
ld1 = load([file_prefix '_cluster_matches.mat'], 'match_vals','boxes','frame_ndxs');
ld1.frame_names = frame_names(ld1.frame_ndxs);
ld1.frame_bbox = voteOnVideoFrameBbox(ld1.frame_names,ld1.boxes,ld1.match_vals,datadir,subdir,resize_factor,PYRASTRIDE,0);

% load cluster matches left-right flipped
ld2 = load([file_prefix '_cluster_matches_mirror.mat'], 'match_vals','boxes','frame_ndxs');  
ld2.frame_names = frame_names(ld2.frame_ndxs);
ld2.frame_bbox = voteOnVideoFrameBbox(ld2.frame_names,ld2.boxes,ld2.match_vals,datadir,subdir,resize_factor,PYRASTRIDE,1);

% frame_names = [ld1.frame_names' ld2.frame_names'];
frame_bbox = [ld1.frame_bbox ld2.frame_bbox];
match_vals = [ld1.match_vals ld2.match_vals];
boxes = [ld1.boxes ld2.boxes];
mirror_flag = [zeros(1,numel(ld1.frame_names)) ones(1,numel(ld2.frame_names))];

clear ld1 ld2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% skip patches whose deep pyramid feature could not be computed
imgInfo(missed_ndx) = [];
match_vals(missed_ndx,:) = [];
[match_vals, sorted_ndx] = sort(match_vals,2,'descend');
boxes(missed_ndx,:,:) = [];
for ii=1:size(boxes,1)
    boxes(ii,:,:) = boxes(ii,sorted_ndx(ii,:),:);
end

img_ids = [imgInfo.img_id];
unique_img_ids = unique(img_ids);
for ii=1:numel(unique_img_ids)    
    imgpath = sprintf(VOCopts.imgpath,ids{unique_img_ids(ii)});
    query_img = imread(imgpath);
    query_width = size(query_img,2);
    query_height = size(query_img,1);

    this_img_ids = find(img_ids==unique_img_ids(ii));
    retrieved_bboxes = [];
    retrieved_bboxes_weight = [];
    for jj=1:numel(this_img_ids)
        pyra_locs = imgInfo(this_img_ids(jj)).pyra_locs;
        query_scale = PYRASTRIDE/imgInfo(this_img_ids(jj)).pyra_scale;
        query_im_box = (pyra_locs-1)*query_scale+1; 
        query_weight = numel(clusters(imgInfo(this_img_ids(jj)).cluster_id).imNdx)/NUMTOPMATCHES;
        
        for kk=1:num_match
            pyra_box = squeeze(boxes(this_img_ids(jj),kk,:))';      
            scale = PYRASTRIDE/pyra_box(end);
            im_box = (pyra_box(1:4)-1)*scale+1;

            if (pyra_box(4)-pyra_box(2) ~= pyra_locs(4)-pyra_locs(2)) || ...
                (pyra_box(3)-pyra_box(1) ~= pyra_locs(3)-pyra_locs(1))
                error('something wrong')
            end
            
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
                    retrieved_bboxes_weight = [retrieved_bboxes_weight; query_weight];                  
                end   
            end
        end        
    end
    
    % could do mean shift per retrieval (which would allow multiple dets
    % per image)
    [clustCent,data2cluster,cluster2dataCell] = MeanShiftCluster(retrieved_bboxes',bandWidth);
    cluster_size = cellfun(@numel,cluster2dataCell,'UniformOutput',true);
    [max_cluster_size,max_cluster_ndx] = max(cluster_size);
    
    if ~isempty(clustCent)
        clustCent(1,:) = max(clustCent(1,:),1);
        clustCent(2,:) = max(clustCent(2,:),1);
        clustCent(3,:) = min(clustCent(3,:),size(query_img,2));
        clustCent(4,:) = min(clustCent(4,:),size(query_img,1));
        
        pseudo_gt.bbox{unique_img_ids(ii)} = [clustCent(1:4,max_cluster_ndx)]';
        pseudo_gt.cluster_size(unique_img_ids(ii)) = max_cluster_size;
    end
end

save([file_prefix '_pseudo_gt_num_match=' num2str(num_match) '_mirrored.mat'], 'pseudo_gt');

