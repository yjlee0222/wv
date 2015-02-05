
imgset = 'trainval';
ids = textread(sprintf(VOCopts.imgsetpath,imgset),'%s');

basedir = '/home/SSD1/yjlee-data/projects/weakVideo/PASCAL2007/';
load([basedir imgset 'class_pos_images.mat'], 'class_pos_images');
K = ceil(numel(class_pos_images(clsNdx).ndx)/2);
NUMCLUSTERS = 100;
NUMTOPMATCHES = ceil(K/2);
NUMBBOXOVERLAP = ceil(K/20);
BBOXOVERLAP = 0.5;

file_prefix = [basedir imgset '/' VOCopts.classes{clsNdx} '/' ...
    'maxNumBboxPerCluster_' num2str(NUMTOPMATCHES) '_numCluster_' num2str(NUMCLUSTERS) ...
    '_numBboxOverlap_' num2str(NUMBBOXOVERLAP) '_bboxOverlap_' num2str(BBOXOVERLAP)];
load([file_prefix '.mat'], 'clusters');

write_dir = '/home/yjlee/web/weakVideo/clusters/';
if ~exist(write_dir,'dir')
    mkdir(write_dir);
end


if ~exist([write_dir cls '_cluster' num2str(numel(clusters)) '_img' num2str(size(clusters(end).imNdx,1)) '.jpg'],'file')
    for ii=1:numel(clusters)
        for jj=1:size(clusters(ii).imNdx,1)
            if exist([write_dir cls '_cluster' num2str(ii) '_img' num2str(jj) '.jpg'],'file')
                continue;
            end
            
            imgpath = sprintf(VOCopts.imgpath,ids{clusters(ii).imNdx(jj)});
            img = imread(imgpath);   

            % [x1 y1 x2 y2]
            x1 = max(1,clusters(ii).boxes(jj,1));
            y1 = max(1,clusters(ii).boxes(jj,2));
            x2 = min(size(img,2),clusters(ii).boxes(jj,3));
            y2 = min(size(img,1),clusters(ii).boxes(jj,4));

            template = zeros(100,100,3,'uint8');

            h = y2-y1+1;
            w = x2-x1+1;
            if h>w            
                small_img = imresize(img(y1:y2,x1:x2,:),[100 w/h*100]);
                offset = max(1,round((100-size(small_img,2))/2));            
                template(:,offset:offset+size(small_img,2)-1,:) = small_img;
            else
                small_img = imresize(img(y1:y2,x1:x2,:),[h/w*100 100]);
                offset = max(1,round((100-size(small_img,1))/2));
                template(offset:offset+size(small_img,1)-1,:,:) = small_img;
            end   

            imwrite(template,[write_dir cls '_cluster' num2str(ii) '_img' num2str(jj) '.jpg'],'jpg');

    %         figure(1); clf; imshow(template);
    %         pause;
        end
    end
end