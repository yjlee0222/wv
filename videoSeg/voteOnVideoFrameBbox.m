function frame_bbox = voteOnVideoFrameBbox(frame_names,boxes,match_vals,datadir,subdir,resize_factor,PYRASTRIDE)

clear frame_bbox;
for ii=1:numel(frame_names)   
    [video_name,shot_name,frame_name] = parseFrameName(datadir,frame_names{ii});
    dense_seg = imread([datadir video_name subdir shot_name '/DenseSegmentation_bdry/' frame_name '_dense.ppm']);
    tube_bbox = getTubeBbox(dense_seg);
    tube_bbox = (tube_bbox-1)./resize_factor+1;

    if isempty(tube_bbox)
        frame_bbox(ii).tube_bbox = [];
        frame_bbox(ii).tube_bbox_weight = [];
        frame_bbox(ii).tube_bbox_count = [];
        continue;
    end
    
    pyra_box = squeeze(boxes(:,ii,:));      
    scale = PYRASTRIDE./pyra_box(:,end);
    im_box = bsxfun(@times,(pyra_box(:,1:4)-1),scale)+1;

    overlaps = computeOverlap(tube_bbox,im_box);
    % weight each box by match score * overlap
    tube_bbox_weight = sum(bsxfun(@times,overlaps,match_vals(:,ii)'),2);
    % just in case, need to normalized tube_bbox_weight by number of times the
    % bounding box was touched (may not be necessary, or could unwantingly reward spares activations...)
    tube_bbox_count = sum((overlaps>0),2);

    frame_bbox(ii).tube_bbox = tube_bbox;
    frame_bbox(ii).tube_bbox_weight = tube_bbox_weight;
    frame_bbox(ii).tube_bbox_count = tube_bbox_count;
%     ii
end