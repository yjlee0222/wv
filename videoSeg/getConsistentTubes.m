function tracks = getConsistentTubes(tracks,check_tube_length,overlap_tresh)

% addpath('/home/yjlee/projects/weakVideo/misc');

if nargin<2
    check_tube_length = 3; % 1 before and 1 after (make this odd)
end
if nargin <3
    overlap_tresh = 0.6;
end

num_frames = numel(tracks);

before_after_length = floor(check_tube_length/2);
for ii=1:num_frames   
    boxes = tracks(ii).bbox_info;   
    
    if ii<floor(check_tube_length/2)+1  % 1st frames
        check_frame_ndx = 0:1:before_after_length;
        check = zeros(size(boxes,1),before_after_length);
    elseif ii>num_frames-floor(check_tube_length/2) % last frames
        check_frame_ndx = -1*before_after_length:1:0;
        check = zeros(size(boxes,1),before_after_length);
    else % middle frames
        check_frame_ndx = -1*before_after_length:1:before_after_length;
        check = zeros(size(boxes,1),2*before_after_length);
    end
    
    count = 1;
    for jj=check_frame_ndx
        if jj==0
            continue;
        end
        this_boxes = tracks(ii+jj).bbox_info;
        for kk=1:size(boxes,1)
            this_label = boxes(kk,5);
            match = find(this_boxes(:,5)==this_label);
            
            if isempty(match)
                check(kk,count) = 0;
            else
                overlaps = computeOverlap(boxes(kk,:), this_boxes(match,:));
                if max(overlaps)>overlap_tresh
                    check(kk,count) = 1;
                end
            end
        end
        count = count + 1;
    end

    tracks(ii).consistent = (sum(check,2)==size(check,2));
end
