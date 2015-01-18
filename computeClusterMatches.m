function computeClusterMatches(query_pyra,cnn_model,frame_names,save_dir)

% im = imread(frame_names{1843});
% im = imread(frame_names{25});
% im = imread(frame_names{700});

cluster_ndx = query_pyra.cluster_ndx;
group_ndx = query_pyra.group_ndx;

for ii=1:numel(frame_names)
    im = imread(frame_names{ii});

    h = query_pyra.level_sizes(1,1);
    w = query_pyra.level_sizes(1,2);
    
%     th = tic;   
    pyra = deep_pyramid(im, cnn_model);
    pyra = deep_pyramid_add_padding(pyra, 0, 0);
%     fprintf('deep_pyramid took %.3fs\n', toc(th));
%     th = tic;
    pyra = pyramid2Mat(pyra,h,w,1);
%     fprintf('pyramid2mat took %.3fs\n', toc(th));

    D = query_pyra.Feats'*pyra.featMat;
    [matchVal,matchNdx] = max(D,[],2);
    
    save([save_dir 'cluster' num2str(cluster_ndx) '_group' num2str(group_ndx) '_frame' num2str(ii) '.mat'], ...
        'matchVal','matchNdx');
    
    ii
end