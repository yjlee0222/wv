% function setupClusterFeats(clusters,cnn_model,VOCopts,ids)
clear;

addpath('/home/yjlee/projects/weakVideo/misc/');
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
load([file_prefix '_cluster_matches.mat'], 'match_vals','match_ndxs','boxes','frame_ndxs');

[match_vals, sorted_ndx] = sort(match_vals,2,'descend');
for ii=1:size(match_ndxs,1)
    match_ndxs(ii,:) = match_ndxs(ii,sorted_ndx(ii,:));
    boxes(ii,:,:) = boxes(ii,sorted_ndx(ii,:),:);
end
frame_names = frame_names(frame_ndxs);

for imNdx = 1:numel(imgInfo)
    % skip images whos deep pyramid feature could not be computed
    if ~isempty(find(missed_ndx==imNdx))
        continue;
    end
    img_id = imgInfo(imNdx).img_id;
    imgpath = sprintf(VOCopts.imgpath,ids{img_id});
    query_img = imread(imgpath);

    pyra_locs = imgInfo(imNdx).pyra_locs;
    query_scale = PYRASTRIDE/imgInfo(imNdx).pyra_scale;
    query_im_box = (pyra_locs-1)*query_scale+1;      
    
    figure(1); clf; axis tight; axis off; 
    set(gca,'Position',[0 0 1 1]); % Make the axes occupy the whole figure
    imshow(query_img); %title('red: original, cyan: adjusted to deep pyramid');
    hold on;
    rectangle('Position', [query_im_box(1) query_im_box(2) query_im_box(3)-query_im_box(1)+1 query_im_box(4)-query_im_box(2)+1], 'EdgeColor','c','LineWidth',3);
    hold off;
    
    pyra_width = [pyra_locs(3)-pyra_locs(1)];
    pyra_height = [pyra_locs(4)-pyra_locs(2)];

    for ii=1:5
        img = imread(frame_names{sorted_ndx(imNdx,ii)}); 

        pyra_box = squeeze(boxes(imNdx,ii,:))';      
        scale = PYRASTRIDE/pyra_box(end);
        im_box = ([pyra_box(1:2) pyra_box(1)+pyra_width pyra_box(2)+pyra_height]-1)*scale+1;

        figure(2); clf; axis tight; axis off; 
        set(gca,'Position',[0 0 1 1]); % Make the axes occupy the whole figure
        imshow(img); %title(match_vals(imNdx,ii));
        rectangle('Position', [im_box(1) im_box(2) im_box(3)-im_box(1)+1 im_box(4)-im_box(2)+1], 'EdgeColor','c','LineWidth',3);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % get automatically computed bounding box from video frame
        [video_name,shot_name,frame_name] = parseFrameName(datadir,frame_names{sorted_ndx(imNdx,ii)});
       
%         dense_seg = imread([datadir video_name subdir shot_name '/DenseSegmentation_bdry/' frame_name '_dense.ppm']);
%         tube_bbox = getTubeBbox(dense_seg);
%         tube_bbox = (tube_bbox-1)./resize_factor+1;

        figure(3); clf;
        clear tracks;
        nn = 1;
        for jj=-1:1:1
            try 
                dense_seg = imread([datadir video_name subdir shot_name '/DenseSegmentation_bdry/' frame_name(1:5) sprintf('%04d',str2num(frame_name(6:end))+jj) '_dense.ppm']);
                tracks(nn).bbox_info = getTubeBbox(dense_seg);
                tracks(nn).bbox_info(:,5) = 1; % since video seg labels are brittle..
                if jj==0
                    valid_track_ndx = nn;
                end
                nn = nn + 1;
                figure(3); subplot(1,3,jj+2); imagesc(dense_seg);                
            catch
            end            
        end
        tracks = getConsistentTubes(tracks,3,0.1);
        
        % remove inconsistent tubes
        consistent_tubes = find(tracks(valid_track_ndx).consistent==1);
        tube_bbox = tracks(valid_track_ndx).bbox_info(consistent_tubes,1:4);        
%         tube_bbox = tracks(valid_track_ndx).bbox_info(:,1:4);        
        tube_bbox = (tube_bbox-1)./resize_factor+1;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
        overlaps = computeOverlap(im_box, tube_bbox);
        
        for jj=1:size(tube_bbox,1)
%             if ~isempty(find(consistent_tubes==jj))
                if overlaps(jj)>0.2
                    figure(2);
                    rectangle('Position', [tube_bbox(jj,1) tube_bbox(jj,2) tube_bbox(jj,3)-tube_bbox(jj,1)+1 tube_bbox(jj,4)-tube_bbox(jj,2)+1], 'EdgeColor','y','LineWidth',3);
                    title(overlaps(jj));

                    retrieved_bbox = query_im_box + (tube_bbox(jj,1:4)-im_box)*query_scale/scale;

                    figure(1); clf; axis tight; axis off; 
                    set(gca,'Position',[0 0 1 1]); % Make the axes occupy the whole figure
                    imshow(query_img); %title('red: original, cyan: adjusted to deep pyramid');
                    hold on;
                    rectangle('Position', [query_im_box(1) query_im_box(2) query_im_box(3)-query_im_box(1)+1 query_im_box(4)-query_im_box(2)+1], 'EdgeColor','c','LineWidth',3);
                    rectangle('Position', [retrieved_bbox(1) retrieved_bbox(2) retrieved_bbox(3)-retrieved_bbox(1)+1 retrieved_bbox(4)-retrieved_bbox(2)+1], 'EdgeColor','y','LineWidth',3);       
                    hold off;
                else
                    figure(2);
                    rectangle('Position', [tube_bbox(jj,1) tube_bbox(jj,2) tube_bbox(jj,3)-tube_bbox(jj,1)+1 tube_bbox(jj,4)-tube_bbox(jj,2)+1], 'EdgeColor','w','LineWidth',3);
                end            
%            else
%                 figure(2);
%                 rectangle('Position', [tube_bbox(jj,1) tube_bbox(jj,2) tube_bbox(jj,3)-tube_bbox(jj,1)+1 tube_bbox(jj,4)-tube_bbox(jj,2)+1], 'EdgeColor','k','LineWidth',3);
%            end
%             pause;
        end       
        
        pause;
    end
end


    