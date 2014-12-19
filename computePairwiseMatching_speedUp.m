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
savedir = [basedir 'trainval/pool5_matches_gpu/'];
if ~exist(savedir,'dir')
    mkdir(savedir);
end

g = gpuDevice();

numChunk = 100;
numIdPerRound = ceil(numel(ids)/numChunk);

for kk=1:numChunk    

    if exist([savedir ids{class_pos_images(clsNdx).ndx(end)} '_chunk' num2str(kk) '.mat'], 'file')
        fprintf('done with chunk %d/%d\n',kk,numChunk);
        continue;
    end
    
    t1 = tic;
    feats = zeros(200000,9216,'single');
    idNdx = zeros(1,200000,'single');
    n = 1;
    for ii=numIdPerRound*(kk-1)+1:numIdPerRound*kk      
        load([featdir ids{ii} '.mat'],'feat');
        feats(n:n+size(feat,1)-1,:) = feat;
        idNdx(n:n+size(feat,1)-1) = ii;
        n = n + size(feat,1);
    end
    gfeat2 = gpuArray(feats(1:n-1,:)');    
%     fprintf('time to load feature: %f\n',toc(t1));
    
    for ii=1:numel(class_pos_images(clsNdx).ndx)  
        this_id = class_pos_images(clsNdx).ndx(ii);
        if exist([savedir ids{this_id} '_chunk' num2str(kk) '.mat'], 'file')
            fprintf('done with im %d/%d, chunk %d/%d\n',ii,numel(class_pos_images(clsNdx).ndx),kk,numChunk);
            continue;
        end

%         th = tic;
        load([featdir ids{this_id} '.mat'],'feat');
        gfeat1 = gpuArray(feat);
%         fprintf('time to load one feature: %f\n',toc(th));
        
        %%%%%%%%%%%%%%%%%%%%
        % GPU
%         t2 = tic;
        sim = gather(gfeat1 * gfeat2);
%         fprintf('time to compute matrix multiplication: %f\n',toc(t2));
        %%%%%%%%%%%%%%%%%%%%

%         t3 = tic;
        maxVals = zeros(size(feat,1),numIdPerRound,'single');
        maxNdxs = zeros(size(feat,1),numIdPerRound,'single');
        n = 1;
        for jj=numIdPerRound*(kk-1)+1:numIdPerRound*kk           
            [maxVals(:,n),maxNdxs(:,n)] = max(sim(:,idNdx==jj),[],2);
            n = n + 1;
        end
%         fprintf('time to find max: %f\n\n',toc(t3)/numIdPerRound);

%         fprintf('done with im %d/%d\n',ii,numel(class_pos_images(clsNdx).ndx));
%         fprintf('time per pair: %f/%d = %f\n',toc(th),numIdPerRound,toc(th)/numIdPerRound);

%         g.FreeMemory
        save([savedir ids{this_id} '_chunk' num2str(kk) '.mat'], 'maxVals','maxNdxs');
    end
    fprintf('done with chunk %d/%d\n',kk,numChunk);
    fprintf('time per pair: %f/%d = %f\n',toc(t1),numel(class_pos_images(clsNdx).ndx)*numIdPerRound,...
        toc(t1)/(numel(class_pos_images(clsNdx).ndx)*numIdPerRound));
end




