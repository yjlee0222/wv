clear;

addpath('/home/yjlee/Downloads/VOCdevkit/VOCcode');
addpath('/home/yjlee/projects/weakVideo/misc');

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
% clear img;
% only keep top M boxes per image
M = 200;
posRatio = zeros(numel(class_pos_images(clsNdx).ndx)*M, numel(ids),'uint16');
imgNdx = zeros(numel(class_pos_images(clsNdx).ndx)*M, numel(ids),'uint16');
ssNdx = zeros(numel(class_pos_images(clsNdx).ndx)*M, numel(ids),'uint16');

count = 1;
for ii=1:numel(class_pos_images(clsNdx).ndx)  
    this_id = class_pos_images(clsNdx).ndx(ii);    
    
    n = 1;
    for jj=1:numChunk   
        load([matchdir ids{this_id} '_chunk' num2str(jj) '.mat'], 'maxVals','maxNdxs');
        
        if jj==1
            maxSsValMat = zeros(size(maxVals,1),numel(ids),'single');
            maxSsNdxMat = zeros(size(maxNdxs,1),numel(ids),'uint16');
        end
        
        maxSsValMat(:,n:n+size(maxVals,2)-1) = maxVals;
        maxSsNdxMat(:,n:n+size(maxNdxs,2)-1) = uint16(maxNdxs);

        n = n + size(maxVals,2);
    end
    maxSsValMat = maxSsValMat(:,1:numel(ids));
    maxSsNdxMat = maxSsNdxMat(:,1:numel(ids));
    
    [maxSsValMat, sortedImgNdx] = sort(maxSsValMat,2,'descend');  
    posLabelMat = zeros(size(maxSsValMat),'single');
    for jj=1:size(maxSsNdxMat,1)
        maxSsNdxMat(jj,:) = maxSsNdxMat(jj,sortedImgNdx(jj,:));
        posLabelMat(jj,:) = posLabels(sortedImgNdx(jj,:));
    end
    
    sumPosLabel = sum(posLabelMat(:,1:K),2);
    [~,sPLndx] = sort(sumPosLabel,'descend');
    
    posRatio(count:count+M-1,:) =  uint16(posLabelMat(sPLndx(1:M),:));
    imgNdx(count:count+M-1,:) =  uint16(sortedImgNdx(sPLndx(1:M),:));
    ssNdx(count:count+M-1,:) =  uint16(maxSsNdxMat(sPLndx(1:M),:));
    
    count = count + M;
%     img(ii).posRatio = uint16(sum(posLabelMat(:,1:K),2));
%     img(ii).sortedNdx = uint16(sortedNdx);
%     img(ii).maxNdx = maxNdxMat;
    ii
end

% posRatio = [];
% imgNdx = [];
% boxNdx = [];
% for ii=1:numel(class_pos_images(clsNdx).ndx) 
%     posRatio = [posRatio; img(ii).posRatio];
%     imgNdx = [imgNdx; ii*ones(size(img(ii).posRatio))];
%     boxNdx = [boxNdx; (1:numel(img(ii).posRatio))'];
% end
% 
% [posRatio,sortedNdx]= sort(posRatio,'descend');
% imgNdx = imgNdx(sortedNdx);
% boxNdx = boxNdx(sortedNdx);

sumPosLabel = sum(posRatio(:,1:K),2);
[sumPosLabel,sPLndx] = sort(sumPosLabel,'descend');
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
clear clusters;
totClust = 1;
NUMCLUSTERS = 50;
NUMTOPMATCHES = ceil(K/2);
DEBUG_FLAG = 1;
for jj=1:numel(sPLndx)
    if totClust > NUMCLUSTERS
        break;
    end  
    
    thisBoxes = [];
    thisImgNdx = [];    
    for ii=1:NUMTOPMATCHES
        % only if the bounding box comes from a positive image
        if posRatio(sPLndx(jj),ii)==1
            img_id = imgNdx(sPLndx(jj),ii);
            load([featdir ids{img_id} '.mat'],'boxes');
            thisImgNdx = [thisImgNdx; img_id];
            thisBoxes = [thisBoxes; boxes(ssNdx(sPLndx(jj),ii),:)];           
        end
    end    
    clusters(totClust).imNdx = thisImgNdx;
    clusters(totClust).boxes = thisBoxes;    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % check overlap to higher-ranked clusters 
    stopFlag = 0;
    if jj>1
        [stopFlag] = checkClusterOverlap(clusters,ceil(K/20),0.5);            
    end
    if stopFlag == 1
        continue;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    totClust = totClust + 1   
     
    if DEBUG_FLAG==1
        figure(1); clf;
        set(gcf,'Color',[1 1 1]);
    
        for ii=1:min(size(thisImgNdx,1),25)
            imgpath = sprintf(VOCopts.imgpath,ids{thisImgNdx(ii)});
            I = imread(imgpath);   

            % [x1 y1 x2 y2]
            x1 = thisBoxes(ii,1);
            y1 = thisBoxes(ii,2);
            x2 = thisBoxes(ii,3);
            y2 = thisBoxes(ii,4);

            subplot(5,5,ii); axis tight; axis off;
            imshow(I(y1:y2,x1:x2,:));
        end

        pause(0.1);
    end
end
% need to save the clusters..
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

