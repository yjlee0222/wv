clear;

VOCinit;
clsNdx = 7;
cls = VOCopts.classes{clsNdx};
imgset = 'trainval';
ids = textread(sprintf(VOCopts.imgsetpath,imgset),'%s');

basedir = '/home/SSD1/yjlee-data/projects/weakVideo/PASCAL2007/';
load([basedir imgset 'class_pos_images.mat'], 'class_pos_images');
K = ceil(numel(class_pos_images(clsNdx).ndx)/2);
NUMCLUSTERS = 100;
NUMTOPMATCHES = ceil(K/2);
NUMBBOXOVERLAP = ceil(K/20);
BBOXOVERLAP = 0.5;
PYRASTRIDE = 16;

file_prefix = [basedir imgset '/' VOCopts.classes{clsNdx} '/' ...
    'maxNumBboxPerCluster_' num2str(NUMTOPMATCHES) '_numCluster_' num2str(NUMCLUSTERS) ...
    '_numBboxOverlap_' num2str(NUMBBOXOVERLAP) '_bboxOverlap_' num2str(BBOXOVERLAP)];

% num_match = 5;
num_match = 20;
load([file_prefix '_pseudo_gt_num_match=' num2str(num_match) '_mirrored.mat'], 'pseudo_gt');

overlap_scores = [];
cluster_size = [];
for ii=1:length(ids)
    if isempty(pseudo_gt.bbox{ii})
        continue;
    end
    
    try
      voc_rec = PASreadrecord(sprintf(VOCopts.annopath, ids{ii}));
    catch
      voc_rec = [];
      ii
    end
    gt_boxes = cat(1, voc_rec.objects(:).bbox);
    gt_classes = {voc_rec.objects(:).class};
    
    match_class_ndx = find(strcmp(gt_classes,cls)==1);    
    if ~isempty(match_class_ndx)
        overlaps = computeOverlap(pseudo_gt.bbox{ii}, gt_boxes(match_class_ndx,:));
        overlap_scores = [overlap_scores; max(overlaps)];
        cluster_size = [cluster_size; pseudo_gt.cluster_size(ii)];
    end
end

fprintf('num_match: %d, mean ov score: %f, # bbox: %d\n',num_match,mean(overlap_scores),numel(overlap_scores));
% overlap_scores(cluster_size<=5) = [];
overlap_scores(cluster_size<=100) = [];
fprintf('num_match: %d, mean ov score: %f, # bbox: %d\n',num_match,mean(overlap_scores),numel(overlap_scores));

% num_match: 5, mean ov score: 0.563553, # bbox: 700
% num_match: 5, mean ov score: 0.615804, # bbox: 584
% num_match: 20, mean ov score: 0.570685, # bbox: 709
% num_match: 20, mean ov score: 0.654206, # bbox: 494

