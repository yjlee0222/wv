
imgset = 'trainval';
ids = textread(sprintf(VOCopts.imgsetpath,imgset),'%s');
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
load([file_prefix '_cluster_feats.mat'], 'imgInfo');

write_dir = '/home/yjlee/web/weakVideo/cluster_matches/';
if ~exist(write_dir,'dir')
    mkdir(write_dir);
end

imNdx = 1:skip:numel(imgInfo);
if exist([write_dir cls '_img' num2str(imNdx(end)) '_match' num2str(num_matches) '.jpg'],'file')
    load([file_prefix '_cluster_matches.mat'], 'match_vals');
    [match_vals, sorted_ndx] = sort(match_vals,2,'descend');
else
    load([file_prefix '_cluster_matches.mat'], 'match_vals','match_ndxs','boxes','frame_ndxs');

    [match_vals, sorted_ndx] = sort(match_vals,2,'descend');
    for ii=1:size(match_ndxs,1)
        match_ndxs(ii,:) = match_ndxs(ii,sorted_ndx(ii,:));
        boxes(ii,:,:) = boxes(ii,sorted_ndx(ii,:),:);
    end
    frame_names = frame_names(frame_ndxs);

    % only display every n'th image
    for imNdx = 1:skip:numel(imgInfo)
        img_id = imgInfo(imNdx).img_id;
        imgpath = sprintf(VOCopts.imgpath,ids{img_id});
        img = imread(imgpath);
        this_box = imgInfo(imNdx).box;
        x1 = this_box(1); y1 = this_box(2); x2 = this_box(3); y2 = this_box(4);

        pyra_locs = imgInfo(imNdx).pyra_locs;
        scale = PYRASTRIDE/imgInfo(imNdx).pyra_scale;
        im_box = (pyra_locs-1)*scale+1;    

        template = zeros(100,100,3,'uint8');
        h = size(img,1);
        w = size(img,2);
        if h>w            
            small_img = imresize(img,[100 w/h*100]);
            offset = max(1,round((100-size(small_img,2))/2));            
            template(:,offset:offset+size(small_img,2)-1,:) = small_img;

            x1 = x1*(100/h)+offset; x2 = x2*(100/h)+offset;
            y1 = y1*(100/h); y2 = y2*(100/h);

            im_box(1) = im_box(1)*(100/h)+offset; im_box(3) = im_box(3)*(100/h)+offset;
            im_box(2) = im_box(2)*(100/h); im_box(4) = im_box(4)*(100/h);
        else
            small_img = imresize(img,[h/w*100 100]);
            offset = max(1,round((100-size(small_img,1))/2));
            template(offset:offset+size(small_img,1)-1,:,:) = small_img;

            x1 = x1*(100/w); x2 = x2*(100/w);
            y1 = y1*(100/w)+offset; y2 = y2*(100/w)+offset;

            im_box(1) = im_box(1)*(100/w); im_box(3) = im_box(3)*(100/w);
            im_box(2) = im_box(2)*(100/w)+offset; im_box(4) = im_box(4)*(100/w)+offset;
        end  

        figure(1); clf;    
        axis tight; axis off; 
        set(gca,'Position',[0 0 1 1]); % Make the axes occupy the whole figure
        imshow(template); %title('red: original, cyan: adjusted to deep pyramid');
        hold on;
        rectangle('Position', [x1 y1 x2-x1+1 y2-y1+1], 'EdgeColor','r','LineWidth',3);
        rectangle('Position', [im_box(1) im_box(2) im_box(3)-im_box(1)+1 im_box(4)-im_box(2)+1], 'EdgeColor','c','LineWidth',3);
        hold off;

        I = getframe(gcf);
        imwrite(I.cdata,[write_dir cls '_img' num2str(imNdx) '.jpg'],'jpg');

        pyra_width = [pyra_locs(3)-pyra_locs(1)];
        pyra_height = [pyra_locs(4)-pyra_locs(2)];

        for ii=1:num_matches
            img = imread(frame_names{sorted_ndx(imNdx,ii)}); 
            pyra_box = squeeze(boxes(imNdx,ii,:))';      
            scale = PYRASTRIDE/pyra_box(end);
            im_box = ([pyra_box(1:2) pyra_box(1)+pyra_width pyra_box(2)+pyra_height]-1)*scale+1;

            template = zeros(100,100,3,'uint8');
            h = size(img,1);
            w = size(img,2);
            if h>w            
                small_img = imresize(img,[100 w/h*100]);
                offset = max(1,round((100-size(small_img,2))/2));            
                template(:,offset:offset+size(small_img,2)-1,:) = small_img;

                im_box(1) = im_box(1)*(100/h)+offset; im_box(3) = im_box(3)*(100/h)+offset;
                im_box(2) = im_box(2)*(100/h); im_box(4) = im_box(4)*(100/h);
            else
                small_img = imresize(img,[h/w*100 100]);
                offset = max(1,round((100-size(small_img,1))/2));
                template(offset:offset+size(small_img,1)-1,:,:) = small_img;

                im_box(1) = im_box(1)*(100/w); im_box(3) = im_box(3)*(100/w);
                im_box(2) = im_box(2)*(100/w)+offset; im_box(4) = im_box(4)*(100/w)+offset;
            end  

            figure(1); clf;    
            axis tight; axis off; 
            set(gca,'Position',[0 0 1 1]); % Make the axes occupy the whole figure
            imshow(template); %title(match_vals(imNdx,ii));
            rectangle('Position', [im_box(1) im_box(2) im_box(3)-im_box(1)+1 im_box(4)-im_box(2)+1], 'EdgeColor','r','LineWidth',3);

            I = getframe(gcf);
            imwrite(I.cdata,[write_dir cls '_img' num2str(imNdx) '_match' num2str(ii) '.jpg'],'jpg');
        end
    end
end

