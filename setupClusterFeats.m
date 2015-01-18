function [query_pyra,imgs] = setupClusterFeats(clusters,cluster_ndx,group_ndx,cnn_model,VOCopts,ids)

clear imgs;
group_size = size(clusters(cluster_ndx).group(group_ndx).cluster_instances,1);
for ii=1:group_size
    this_ndx = clusters(cluster_ndx).group(group_ndx).cluster_instances(ii);
    
    img_id = clusters(cluster_ndx).imNdx(this_ndx);
    imgpath = sprintf(VOCopts.imgpath,ids{img_id});
    I = imread(imgpath); 

    thisBox = clusters(cluster_ndx).boxes(this_ndx,:);
    x1 = thisBox(1); y1 = thisBox(2); x2 = thisBox(3); y2 = thisBox(4);

    imgs{ii} = imresize(I(y1:y2,x1:x2,:), ...
        [clusters(cluster_ndx).group(group_ndx).height clusters(cluster_ndx).group(group_ndx).width]);
end

query_pyra = deep_pyramid_batch(imgs, cnn_model);
query_pyra = deep_pyramid_add_padding(query_pyra, 0, 0);

% w = query_pyra.level_sizes(1,2);
% h = query_pyra.level_sizes(1,1);

query_pyra.Feats = zeros(numel(query_pyra.feat{1}),group_size);
for ii=1:numel(query_pyra.feat)
    feat = vec(query_pyra.feat{ii});
    query_pyra.Feats(:,ii) = feat/sqrt(feat'*feat);
end

query_pyra.cluster_ndx = cluster_ndx;
query_pyra.group_ndx = group_ndx;
