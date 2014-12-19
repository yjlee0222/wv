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
savedir = [basedir 'trainval/pool5_matches_gpu2/'];
if ~exist(savedir,'dir')
    mkdir(savedir);
end

g = gpuDevice();

num_gpu_chunk = 100;
num_gpu_IdPerRound = ceil(numel(ids)/num_gpu_chunk);

num_PosClassImgs = numel(class_pos_images(clsNdx).ndx);
num_cpu_chunk = ceil(num_PosClassImgs/200); % roughly 15GB
num_cpu_IdPerRound = ceil(num_PosClassImgs/num_cpu_chunk);


% start loop
cpu_round = 1;

tt = tic;

cpu_feats = zeros(600000,9216,'single');
cpu_idNdx = zeros(1,600000,'single');
n = 1;
for ii=num_cpu_IdPerRound*(cpu_round-1)+1:min(num_cpu_IdPerRound*cpu_round, num_PosClassImgs)      
    this_id = class_pos_images(clsNdx).ndx(ii);
    load([featdir ids{this_id} '.mat'],'feat');
    cpu_feats(n:n+size(feat,1)-1,:) = feat;
    cpu_idNdx(n:n+size(feat,1)-1) = ii;
    n = n + size(feat,1);
end
cpu_feats = cpu_feats(1:n-1,:);
fprintf('time to load feature: %f\n',toc(tt));
    
for gpu_round=1:num_gpu_chunk    

    if exist([savedir ids{class_pos_images(clsNdx).ndx(min(num_cpu_IdPerRound*cpu_round, num_PosClassImgs))}...
            '_cpuchunk' num2str(cpu_round) '_gpuchunk' num2str(gpu_round) '.mat'], 'file')
        fprintf('done with cpu_chunk %d/%d, gpu_chunk %d/%d\n',cpu_round,num_cpu_chunk,gpu_round,num_gpu_chunk);
        continue;
    end
    
    t1 = tic;
    gpu_feats = zeros(200000,9216,'single');
    gpu_idNdx = zeros(1,200000,'single');
    n = 1;
    for ii=num_gpu_IdPerRound*(gpu_round-1)+1:num_gpu_IdPerRound*gpu_round      
        load([featdir ids{ii} '.mat'],'feat');
        gpu_feats(n:n+size(feat,1)-1,:) = feat;
        gpu_idNdx(n:n+size(feat,1)-1) = ii;
        n = n + size(feat,1);
    end
    gfeat2 = gpuArray(gpu_feats(1:n-1,:)'); 
    clear gpu_feats;
    fprintf('time to load feature: %f\n',toc(t1));
    
    for ii=num_cpu_IdPerRound*(cpu_round-1)+1:min(num_cpu_IdPerRound*cpu_round, num_PosClassImgs)   
        this_id = class_pos_images(clsNdx).ndx(ii);
        if exist([savedir ids{this_id} '_cpuchunk' num2str(cpu_round) '_gpuchunk' num2str(gpu_round) '.mat'], 'file')
            fprintf('done with im %d/%d\n',ii-num_cpu_IdPerRound*(cpu_round-1),min(num_cpu_IdPerRound*cpu_round, num_PosClassImgs));
            continue;
        end

%         th = tic;
        gfeat1 = gpuArray(cpu_feats(cpu_idNdx==ii,:));
%         fprintf('time to load one feature: %f\n',toc(th));
        
        %%%%%%%%%%%%%%%%%%%%
        % GPU
%         t2 = tic;
        sim = gather(gfeat1 * gfeat2);
%         fprintf('time to compute matrix multiplication: %f\n',toc(t2));
        %%%%%%%%%%%%%%%%%%%%

%         t3 = tic;        
        maxVals = zeros(numel(find(cpu_idNdx==ii)),num_gpu_IdPerRound,'single');
        maxNdxs = zeros(numel(find(cpu_idNdx==ii)),num_gpu_IdPerRound,'single');
        n = 1;
        for jj=num_gpu_IdPerRound*(gpu_round-1)+1:min(num_gpu_IdPerRound*gpu_round, numel(ids))           
            [maxVals(:,n),maxNdxs(:,n)] = max(sim(:,gpu_idNdx==jj),[],2);
            n = n + 1;
        end
%         fprintf('time to find max: %f\n\n',toc(t3)/numIdPerRound);

%         fprintf('done with im %d/%d\n',ii,numel(class_pos_images(clsNdx).ndx));
%         fprintf('time per pair: %f/%d = %f\n',toc(th),numIdPerRound,toc(th)/numIdPerRound);

%         g.FreeMemory
        save([savedir ids{this_id} '_cpuchunk' num2str(cpu_round) '_gpuchunk' num2str(gpu_round) '.mat'], 'maxVals','maxNdxs');
    end
    fprintf('done with cpu_chunk %d/%d, gpu_chunk %d/%d\n',cpu_round,num_cpu_chunk,gpu_round,num_gpu_chunk);
    fprintf('time per pair: %f/%d = %f\n',toc(t1),num_cpu_IdPerRound*num_gpu_IdPerRound,...
        toc(t1)/(num_cpu_IdPerRound*num_gpu_IdPerRound));
    c = clock;
    fprintf('%d/%d %d:%02d\n\n',c(2),c(3),c(4),c(5));
end




