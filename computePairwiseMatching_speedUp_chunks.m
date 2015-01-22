clear;

addpath('/home/yjlee/Downloads/VOCdevkit/VOCcode');

VOCinit;

imgset = 'trainval';
ids = textread(sprintf(VOCopts.imgsetpath,imgset),'%s');

basedir = '/home/SSD1/yjlee-data/projects/weakVideo/PASCAL2007/';

% featdir = [basedir 'trainval/pool5/'];
featdir = [basedir 'trainval/pool5_L2norm/'];
savedir = [basedir 'trainval/pool5_matches_gpu_chunks/'];
if ~exist(savedir,'dir')
    mkdir(savedir);
end

g = gpuDevice(1);

numChunk = 400;
numIdPerRound = ceil(numel(ids)/numChunk);
numChunk = ceil(numel(ids)/numIdPerRound);

for mm=1:numChunk
    if exist([savedir 'chunk' num2str(mm) '_' num2str(numChunk) '.mat'], 'file')
        fprintf('done with chunk %d/%d\n',mm,numChunk);
        continue;
    end
    
    if numIdPerRound*(mm-1)+1 > numel(ids)
        break;
    end
    
%     t1 = tic;
    feats1 = zeros(75000,9216,'single');
    idNdx1 = zeros(1,75000,'single');
    n = 1;
    for ii=numIdPerRound*(mm-1)+1:min(numIdPerRound*mm,numel(ids))    
        load([featdir ids{ii} '.mat'],'feat');
        feats1(n:n+size(feat,1)-1,:) = feat;
        idNdx1(n:n+size(feat,1)-1) = ii;
        n = n + size(feat,1);
    end
    gfeat1 = gpuArray(feats1(1:n-1,:)); 
    idNdx1 = idNdx1(1:n-1);
%     fprintf('time to load feature: %f\n',toc(t1));
    
    for nn=mm:numChunk  
        if exist([savedir 'chunk' num2str(mm) '_' num2str(nn) '.mat'], 'file')
            fprintf('done with chunk %d, %d/%d\n',mm,nn,numChunk);
            continue;
        end

        t2 = tic;
        feats2 = zeros(75000,9216,'single');
        idNdx2 = zeros(1,75000,'single');
        n = 1;
        for ii=numIdPerRound*(nn-1)+1:min(numIdPerRound*nn,numel(ids))    
            load([featdir ids{ii} '.mat'],'feat');
            feats2(n:n+size(feat,1)-1,:) = feat;
            idNdx2(n:n+size(feat,1)-1) = ii;
            n = n + size(feat,1);
        end
        gfeat2 = gpuArray(feats2(1:n-1,:)');  
        idNdx2 = idNdx2(1:n-1);
    %     fprintf('time to load feature: %f\n',toc(t2));
        
        %%%%%%%%%%%%%%%%%%%%
        % GPU
%         t2 = tic;
        sim = gather(gfeat1 * gfeat2);
%         fprintf('time to compute matrix multiplication: %f\n',toc(t2));
        %%%%%%%%%%%%%%%%%%%%
        
%         t3 = tic;
        maxVals = zeros(numel(idNdx1),numel(unique(idNdx2)),'single');
        maxNdxs = zeros(numel(idNdx1),numel(unique(idNdx2)),'single');
        n = 1;
        for jj=numIdPerRound*(nn-1)+1:min(numIdPerRound*nn,numel(ids))          
            [maxVals(:,n),maxNdxs(:,n)] = max(sim(:,idNdx2==jj),[],2);
            n = n + 1;
        end
        source_imNdx = idNdx1;
        match_imNdx = unique(idNdx2);
        save([savedir 'chunk' num2str(mm) '_' num2str(nn) '.mat'], 'maxVals','maxNdxs','source_imNdx','match_imNdx');
        
        if mm~=nn       
            maxVals = zeros(numel(idNdx2),numel(unique(idNdx1)),'single');
            maxNdxs = zeros(numel(idNdx2),numel(unique(idNdx1)),'single');
            n = 1;
            for jj=numIdPerRound*(mm-1)+1:min(numIdPerRound*mm,numel(ids))          
                [maxVals(:,n),maxNdxs(:,n)] = max(sim(idNdx1==jj,:),[],1);
                n = n + 1;
            end
            source_imNdx = idNdx2;
            match_imNdx = unique(idNdx1);
            save([savedir 'chunk' num2str(nn) '_' num2str(mm) '.mat'], 'maxVals','maxNdxs','source_imNdx','match_imNdx');
        end        
%         fprintf('time to find max: %f\n\n',toc(t3)/numIdPerRound);

        fprintf('done with chunk %d, %d/%d\n',mm,nn,numChunk);
        fprintf('time per pair: %f/%d = %f\n',toc(t2),numIdPerRound^2,toc(t2)/(numIdPerRound^2));

%         g.FreeMemory        
    end
    fprintf('done with chunk %d/%d\n\n',mm,numChunk);
    try
        fprintf('previous iter: %d/%d %d:%02d\n\n',c(2),c(3),c(4),c(5));
    catch
    end
    c = clock;
    fprintf('%d/%d %d:%02d\n\n',c(2),c(3),c(4),c(5));
end




