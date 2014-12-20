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
savedir = [basedir 'trainval/pool5_matches/'];
if ~exist(savedir,'dir')
    mkdir(savedir);
end

count = 0;
th = tic;
for nn=1:numel(class_pos_images(clsNdx).ndx)  
    ii = class_pos_images(clsNdx).ndx(nn);
    if exist([savedir ids{ii} '.mat'], 'file');
        fprintf('done with im %d/%d\n',nn,numel(class_pos_images(clsNdx).ndx));
        continue;
    end
    load([featdir ids{ii} '.mat'],'feat');
    feat1 = feat;
    gfeat1 = gpuArray(feat1);
    
    maxVals = zeros(size(feat1,1),numel(ids),'single');
    maxNdxs = zeros(size(feat1,1),numel(ids),'single');
    for jj=1:numel(ids)
        if ii==jj
            continue;
        end
        t1 = tic;
        savename = [savedir ids{ii} '_' ids{jj} '.mat'];
        if exist(savename,'file')
            load(savename, 'maxVal','maxNdx');
            maxVals(:,jj) = maxVal;
            maxNdxs(:,jj) = maxNdx;
            delete(savename);
            continue;
        end
        fprintf('time to check existing feature: %f\n',toc(t1));
        
        t2 = tic;
        load([featdir ids{jj} '.mat'],'feat');
        feat2 = feat;
        fprintf('time to load feature: %f\n',toc(t2));
        
%         %%%%%%%%%%%%%%%%%%%%
%         % CPU   
%         sim = feat1 * (feat2');
%         %%%%%%%%%%%%%%%%%%%%
        % GPU
        t3 = tic;        
        gfeat2 = gpuArray(feat2');
        sim = gather(gfeat1 * gfeat2);
        fprintf('time to compute matrix multiplication: %f\n',toc(t3));
        %%%%%%%%%%%%%%%%%%%%
        
        t4 = tic;
        [maxVals(:,jj),maxNdxs(:,jj)] = max(sim,[],2);
        [maxVal,maxNdx] = max(sim,[],1);
        fprintf('time to find max: %f\n\n',toc(t4));
        save([savedir ids{jj} '_' ids{ii} '.mat'], 'maxVal','maxNdx');

%         if count == 100            
%             fprintf('done with im %d/%d : %d/%d\n',nn,numel(class_pos_images(clsNdx).ndx),jj,numel(ids));
%             fprintf('time per pair: %f/%d = %f\n',toc(th),count,toc(th)/count);
%             count = 0;
%             th = tic;
%         end
        count = count + 1;
    end
              
    fprintf('done with im %d/%d\n',nn,numel(class_pos_images(clsNdx).ndx));
    fprintf('time per pair: %f/%d = %f\n',toc(th),count,toc(th)/count);
    c = clock;
    fprintf('%d/%d %d:%02d\n\n',c(2),c(3),c(4),c(5));

    count = 0;
    th = tic;

    save([savedir ids{ii} '.mat'], 'maxVals','maxNdxs');
end




