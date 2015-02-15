function collatePairwiseMatches_chunks(clsNdx)

addpath('/home/yjlee/Downloads/VOCdevkit/VOCcode');
addpath('/home/yjlee/projects/weakVideo/misc');

VOCinit;

% car = 7
% clsNdx = 7;

imgset = 'trainval';
ids = textread(sprintf(VOCopts.imgsetpath,imgset),'%s');
   
basedir = '/home/SSD1/yjlee-data/projects/weakVideo/PASCAL2007/';
load([basedir imgset 'class_pos_images.mat'], 'class_pos_images');
featdir = [basedir 'trainval/pool5_L2norm/'];
matchdir = [basedir 'trainval/pool5_matches_gpu_chunks/'];

numChunk = 400;
numIdPerRound = ceil(numel(ids)/numChunk);
numChunk = ceil(numel(ids)/numIdPerRound);

posLabels = zeros(1,numel(ids));
posLabels(class_pos_images(clsNdx).ndx) = 1;

K = ceil(numel(class_pos_images(clsNdx).ndx)/2);

% clear img;
% for now, due to memory issues,
% only keep at most top M selective search windows per image based on pos/neg ratio
% eventually will only keep even fewer windows (below)
M = 200;
posRatio = zeros(numel(class_pos_images(clsNdx).ndx)*M, numel(ids),'uint16');
imgNdx = zeros(numel(class_pos_images(clsNdx).ndx)*M, numel(ids),'uint16');
ssNdx = zeros(numel(class_pos_images(clsNdx).ndx)*M, numel(ids),'uint16');

count = 1;
for ii=1:numChunk 
    n = 1;
    for jj=1:numChunk  
        load([matchdir 'chunk' num2str(ii) '_' num2str(jj) '.mat'], 'maxVals','maxNdxs','source_imNdx','match_imNdx');

        valid_ndx = find(ismember(source_imNdx,class_pos_images(clsNdx).ndx));
        if isempty(valid_ndx)
            break;
        end
        maxVals = maxVals(valid_ndx,:);
        maxNdxs = maxNdxs(valid_ndx,:);
        
        if jj==1
            maxSsValMat = zeros(size(maxVals,1),numel(ids),'single');
            maxSsNdxMat = zeros(size(maxNdxs,1),numel(ids),'uint16');
        end
        
        maxSsValMat(:,n:n+size(maxVals,2)-1) = maxVals;
        maxSsNdxMat(:,n:n+size(maxNdxs,2)-1) = uint16(maxNdxs);

        n = n + size(maxVals,2);
    end
    if isempty(valid_ndx)
        continue;
    end
    if n>numel(ids)+1
        error('\n\nSOMETHING WEIRD!!!\n\n');
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
    
    source_imNdx = source_imNdx(valid_ndx);
    unique_imNdx = unique(source_imNdx);
    for jj=1:numel(unique_imNdx)
        ndx = find(source_imNdx==unique_imNdx(jj));        
        [~,sPLndx] = sort(sumPosLabel(ndx),'descend');
        
        MM = min(M,numel(ndx));
        posRatio(count:count+MM-1,:) =  uint16(posLabelMat(ndx(sPLndx(1:MM)),:));
        imgNdx(count:count+MM-1,:) =  uint16(sortedImgNdx(ndx(sPLndx(1:MM)),:));
        ssNdx(count:count+MM-1,:) =  uint16(maxSsNdxMat(ndx(sPLndx(1:MM)),:));
        
        count = count + MM;
    end
%     ii
end
posRatio = posRatio(1:count-1,:);
imgNdx = imgNdx(1:count-1,:);
ssNdx = ssNdx(1:count-1,:);

sumPosLabel = sum(posRatio(:,1:K),2);
[sumPosLabel,sPLndx] = sort(sumPosLabel,'descend');
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
clear clusters;
totClust = 1;
NUMCLUSTERS = 100;
NUMTOPMATCHES = ceil(K/2);
NUMBBOXOVERLAP = ceil(K/20);
BBOXOVERLAP = 0.5;
DEBUG_FLAG = 0;
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
        [stopFlag] = checkClusterOverlap(clusters,NUMBBOXOVERLAP,0.5);            
    end
    if stopFlag == 1
        continue;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    totClust = totClust + 1;   
     
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
if ~exist([basedir imgset '/' VOCopts.classes{clsNdx}], 'dir')
    mkdir([basedir imgset '/' VOCopts.classes{clsNdx}]);
end

param.NUMTOPMATCHES = NUMTOPMATCHES;
param.NUMCLUSTERS = NUMCLUSTERS;
param.NUMBBOXOVERLAP = NUMBBOXOVERLAP;
param.BBOXOVERLAP = BBOXOVERLAP;

save([basedir imgset '/' VOCopts.classes{clsNdx} '/' ...
    'maxNumBboxPerCluster_' num2str(NUMTOPMATCHES) '_numCluster_' num2str(NUMCLUSTERS) ...
    '_numBboxOverlap_' num2str(NUMBBOXOVERLAP) '_bboxOverlap_' num2str(BBOXOVERLAP) '.mat'], ...
    'clusters','param');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

