clear;

addpath('/home/yjlee/Downloads/VOCdevkit/VOCcode');

VOCinit;

% car = 7
clsNdx = 7;

imgset = 'trainval';
ids = textread(sprintf(VOCopts.imgsetpath,imgset),'%s');

basedir = '/home/SSD1/yjlee-data/projects/weakVideo/PASCAL2007/';
load([basedir imgset 'class_pos_images.mat'], 'class_pos_images');

% featdir = [basedir 'trainval/pool5/'];
featdir = [basedir 'trainval/pool5_L2norm/'];
matchdir = [basedir 'trainval/pool5_matches_gpu/'];

numChunk = 100;
numIdPerRound = ceil(numel(ids)/numChunk);
numChunk = ceil(numel(ids)/numIdPerRound);

posLabels = zeros(1,numel(ids));
posLabels(class_pos_images(clsNdx).ndx) = 1;

K = ceil(numel(class_pos_images(clsNdx).ndx)/2);
clear img;
for ii=1:numel(class_pos_images(clsNdx).ndx)  
    this_id = class_pos_images(clsNdx).ndx(ii);    
    
    n = 1;
    for jj=1:numChunk   
        load([matchdir ids{this_id} '_chunk' num2str(jj) '.mat'], 'maxVals','maxNdxs');
        
        if jj==1
            maxValMat = zeros(size(maxVals,1),numel(ids),'single');
            maxNdxMat = zeros(size(maxNdxs,1),numel(ids),'single');
        end
        
        maxValMat(:,n:n+size(maxVals,2)-1) = maxVals;
        maxNdxMat(:,n:n+size(maxNdxs,2)-1) = maxNdxs;

        n = n + size(maxVals,2);
    end
    maxValMat = maxValMat(:,1:numel(ids));
    maxNdxMat = maxNdxMat(:,1:numel(ids));
    
    [maxValMat, sortedNdx] = sort(maxValMat,2,'descend');  
    posLabelMat = zeros(size(maxValMat),'single');
    for jj=1:size(maxNdxMat,1)
        maxNdxMat(jj,:) = maxNdxMat(jj,sortedNdx(jj,:));
        posLabelMat(jj,:) = posLabels(sortedNdx(jj,:));
    end
    
    img(ii).posRatio = sum(posLabelMat(:,1:K),2);
    ii
end

posRatio = [];
imgNdx = [];
boxNdx = [];
for ii=1:numel(class_pos_images(clsNdx).ndx) 
    posRatio = [posRatio; img(ii).posRatio];
    imgNdx = [imgNdx; ii*ones(size(img(ii).posRatio))];
    boxNdx = [boxNdx; (1:numel(img(ii).posRatio))'];
end

[posRatio,sortedNdx]= sort(posRatio,'descend');
imgNdx = imgNdx(sortedNdx);
boxNdx = boxNdx(sortedNdx);


