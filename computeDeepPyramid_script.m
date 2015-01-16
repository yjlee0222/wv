clear;

addpath('/home/yjlee/projects/weakVideo/misc/');
addpath('/home/yjlee/Downloads/caffe-master/matlab/caffe/');
addpath('/home/yjlee/Downloads/DeepPyramid/');
addpath('/home/yjlee/Downloads/VOCdevkit/VOCcode');

datadir = '/home/SSD1/yjlee-data/projects/weakVideo/YouTube-Objects/car/data/';
d = dir(datadir);
d = d(3:end);

n = 1;
clear imgs;
for ii=1:2%numel(d)
    if isdir([datadir d(ii).name])
        dd = dir([datadir d(ii).name '/shots']);
        dd = dd(3:end);
        
        for jj=1:numel(dd)
            ddd = dir([datadir d(ii).name '/shots/' dd(jj).name '/*.jpg']);
            
            for kk=1:numel(ddd)
                imgs{n,1} = [datadir d(ii).name '/shots/' dd(jj).name '/' ddd(kk).name];
                n = n + 1;
            end
        end
    end
end


VOCinit;
clsNdx = 7;
imgset = 'trainval';
ids = textread(sprintf(VOCopts.imgsetpath,imgset),'%s');
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

img_id = clusters(1).imNdx(1);
thisBox = clusters(1).boxes(1,:);

% featdir = [basedir 'trainval/pool5_L2norm/'];
% load([featdir ids{img_id} '.mat'],'feat','boxes');
% featNdx = find(sum(bsxfun(@minus,boxes,thisBox).^2,2)==0);
% D = feat(featNdx,:)*pyra.featMat;

imgpath = sprintf(VOCopts.imgpath,ids{img_id});
I = imread(imgpath);   

device_id = 1;
caffe('set_device', device_id);
% cnn_scale_1 = init_cnn_model_one_level('use_gpu', true, 'use_caffe', true);
% query_pyra = deep_pyramid(I, cnn_scale_1);
cnn_scale_7 = init_cnn_model('use_gpu', true, 'use_caffe', true);
query_pyra = deep_pyramid(I, cnn_scale_7);
query_pyra = deep_pyramid_add_padding(query_pyra, 0, 0);
query_pyra_boxes = im_to_pyra_coords(query_pyra, thisBox);
thisScale = 6;
w = query_pyra_boxes{thisScale}(3)-query_pyra_boxes{thisScale}(1)+1;
h = query_pyra_boxes{thisScale}(4)-query_pyra_boxes{thisScale}(2)+1;
queryFeat = vec(query_pyra.feat{thisScale}(query_pyra_boxes{thisScale}(2):query_pyra_boxes{thisScale}(4),query_pyra_boxes{thisScale}(1):query_pyra_boxes{thisScale}(3),:));
queryFeat = queryFeat/sqrt(queryFeat'*queryFeat);

% im = imread(imgs{1843});
im = imread(imgs{25});
% im = imread(imgs{700});
th = tic;
cnn_scale_7 = init_cnn_model('use_gpu', true, 'use_caffe', true);
pyra = deep_pyramid(im, cnn_scale_7);
pyra = deep_pyramid_add_padding(pyra, 0, 0);
pyra = pyramid2Mat2(pyra,h,w,1);
fprintf('deep_pyramid took %.3fs\n', toc(th));

D = queryFeat'*pyra.featMat;
[matchVal,matchNdx] = sort(D,'descend');

figure(1); clf;
for ii=1:12
%     boxes = [pyra.featPos(2,matchNdx(ii)) pyra.featPos(1,matchNdx(ii)) ...
%         pyra.featPos(2,matchNdx(ii))+6 pyra.featPos(1,matchNdx(ii))+6 pyra.featLevel(matchNdx(ii))];
    boxes = [pyra.featPos(2,matchNdx(ii)) pyra.featPos(1,matchNdx(ii)) ...
        pyra.featPos(2,matchNdx(ii))+w-1 pyra.featPos(1,matchNdx(ii))+h-1 pyra.featLevel(matchNdx(ii))]  
    im_boxes = pyra_to_im_coords(pyra, boxes);
    
    subplot(3,4,ii); 
    imshow(im); title(matchVal(ii));
    rectangle('Position', [im_boxes(1) im_boxes(2) im_boxes(3)-im_boxes(1)+1 im_boxes(4)-im_boxes(2)+1], 'EdgeColor', 'r');
end

I = imread(imgpath);   
% x1 = thisBox(1); y1 = thisBox(2); x2 = thisBox(3); y2 = thisBox(4);
boxes = [query_pyra_boxes{thisScale}(1) query_pyra_boxes{thisScale}(2) ...
        query_pyra_boxes{thisScale}(3) query_pyra_boxes{thisScale}(4) thisScale];
im_boxes = pyra_to_im_coords(query_pyra, boxes);

figure(2); clf;
axis tight; axis off;
% imshow(I(y1:y2,x1:x2,:));
imshow(I);
rectangle('Position', [im_boxes(1) im_boxes(2) im_boxes(3)-im_boxes(1)+1 im_boxes(4)-im_boxes(2)+1], 'EdgeColor', 'r');


