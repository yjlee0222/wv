function [boxes,feat] = computeSelectiveSearchDeepFeats(im, rcnn_model, boxes)

% compute selective search candidates
% fprintf('Computing candidate regions...');
% th = tic();
if nargin<3
    fast_mode = true;
    im_width = 500;
    boxes = selective_search_boxes(im, fast_mode, im_width);
    % compat: change coordinate order from [y1 x1 y2 x2] to [x1 y1 x2 y2]
    boxes = boxes(:, [2 1 4 3]);
    % fprintf('found %d candidates (in %.3fs).\n', size(boxes,1), toc(th));
end

% extract features from candidates (one row per candidate box)
% fprintf('Extracting CNN features from regions...');
% th = tic();
feat = rcnn_features(im, boxes, rcnn_model);
% feat = rcnn_scale_features(feat, rcnn_model.training_opts.feat_norm_mean);
% fprintf('done (in %.3fs).\n', toc(th));