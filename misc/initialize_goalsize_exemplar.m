function model = initialize_goalsize_exemplar(I, bbox, cnn_model, init_params)
% Initialize the exemplar (or scene) such that the representation
% which tries to choose a region which overlaps best with the given
% bbox and contains roughly init_params.goal_ncells cells, with a
% maximum dimension of init_params.MAXDIM
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
%
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm
%
% modified by Yong Jae Lee

if ~exist('init_params','var')
    init_params.MAXDIM = 12;
    init_params.goal_ncells = 48;
end

% Create a blank image with the exemplar inside
Ibox = zeros(size(I,1), size(I,2));
Ibox(bbox(2):bbox(4), bbox(1):bbox(3)) = 1;

pyra = deep_pyramid(I, cnn_model);
pyra = deep_pyramid_add_padding(pyra, 0, 0);

% Extract the regions most overlapping with Ibox from each level in the pyramid
[masker,sizer] = get_matching_masks(pyra.feat, Ibox);
% Now choose the mask which is closest to N cells
[targetlvl, mask] = get_ncell_mask(init_params, masker, sizer);

[uu,vv] = find(mask);
curfeats = pyra.feat{targetlvl}(min(uu):max(uu),min(vv):max(vv),:);

model.init_params = init_params;
model.feat_size = size(curfeats);
model.mask = logical(ones(model.feat_size(1),model.feat_size(2)));
model.feats = curfeats;
model.pyra_level = targetlvl;
model.pyra_scale = pyra.scales(targetlvl);
model.pyra_locs = [min(vv) min(uu) max(vv) max(uu)]; % x1, y1, x2, y2

fprintf(1,'initialized with deep_pyramid_size = [%d %d]\n',model.feat_size(1),model.feat_size(2));


% ------------------------------------------------------------------------
function [masker,sizer] = get_matching_masks(f_real, Ibox)
% ------------------------------------------------------------------------
%Given a feature pyramid, and a segmentation mask inside Ibox, find
%the best matching region per level in the feature pyramid

masker = cell(length(f_real),1);
sizer = zeros(length(f_real),2);

for a=1:length(f_real)
%     goods = double(sum(f_real{a}.^2,3)>0);
    masker{a} = max(0.0,min(1.0,imresize(Ibox,[size(f_real{a},1) size(f_real{a},2)])));
    [~,ind] = max(masker{a}(:));
    masker{a} = (masker{a}>.1);% & goods;
    if sum(masker{a}(:))==0
        [aa,bb] = ind2sub(size(masker{a}),ind);
        masker{a}(aa,bb) = 1;
    end
    [uu,vv] = find(masker{a});
    masker{a}(min(uu):max(uu),min(vv):max(vv)) = 1;
    sizer(a,:) = [range(uu)+1 range(vv)+1];
end


% ------------------------------------------------------------------------
function [targetlvl,mask] = get_ncell_mask(init_params, masker, sizer)
% ------------------------------------------------------------------------
%Get a the mask and features, where mask is closest to NCELL cells
%as possible

for ii=1:size(masker)
    [uu,vv] = find(masker{ii});
    if ((max(uu)-min(uu)+1) <= init_params.MAXDIM) && ((max(vv)-min(vv)+1) <= init_params.MAXDIM)
        targetlvl = ii;
        mask = masker{targetlvl};        
        return;
    end
end

fprintf(1,'didnt find a match\n');
%Default to older strategy
ncells = prod(sizer,2);
[~,targetlvl] = min(abs(ncells-init_params.goal_ncells));
mask = masker{targetlvl};

