clear;

addpath('/home/yjlee/Downloads/VOCdevkit/VOCcode');

VOCinit;

imgset = 'trainval';
ids = textread(sprintf(VOCopts.imgsetpath,imgset),'%s');

basedir = '/home/SSD1/yjlee-data/projects/weakVideo/PASCAL2007/';
featdir = [basedir 'trainval/pool5/'];
savedir = [basedir 'trainval/pool5_L2norm/'];
if ~exist(savedir,'dir')
    mkdir(savedir);
end

for ii=1:numel(ids)  
    load([featdir ids{ii} '.mat'],'feat','boxes');
    feat = bsxfun(@times, feat, 1./sqrt(sum(feat.*feat,2)));
    save([savedir ids{ii} '.mat'],'feat','boxes');
    
    ii
end