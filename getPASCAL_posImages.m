clear;

addpath('/home/yjlee/Downloads/VOCdevkit/VOCcode');

VOCinit;

imgset = 'trainval';
ids = textread(sprintf(VOCopts.imgsetpath,imgset),'%s');

for clsNdx=1:20
    class_(clsNdx).ndx = zeros(numel(ids),1);
end

for ii=1:numel(ids)
    rec = PASreadrecord(sprintf(VOCopts.annopath,ids{ii}));
    
    for clsNdx=1:20
        cls = VOCopts.classes{clsNdx};
        
        if ~isempty(find(strcmp(cls,{rec.objects.class}))==1)
            class_(clsNdx).ndx(ii) = 1;
        end    
    end  
end

for clsNdx=1:20
    class_pos_images(clsNdx).ndx = find(class_(clsNdx).ndx==1);
%     numel(find(class_(clsNdx).ndx==1))
end

savedir = '/home/SSD1/yjlee-data/projects/weakVideo/PASCAL2007/';
save([savedir imgset 'class_pos_images.mat'], 'class_pos_images');