function clusters = initializeClustersForMatching(clusters,group_size)

for ii=1:numel(clusters)
    clusters(ii).group = init_cluster_for_matching(clusters(ii),group_size);
end


% ------------------------------------------------------------------------
function group = init_cluster_for_matching(cluster,group_size)
% ------------------------------------------------------------------------

w = cluster.boxes(:,3)-cluster.boxes(:,1)+1;
h = cluster.boxes(:,4)-cluster.boxes(:,2)+1;
bbox_info = [h w w./h h.*w]; % height, width, aspect ratio, area
[bbox_info,sort_ndx] = sortrows(bbox_info,-3);

num_bbox = size(bbox_info,1);
num_groups = ceil(num_bbox/group_size);

clear group;
for ii=1:num_groups
    ndx = (ii-1)*group_size+1:min(ii*group_size,num_bbox);
    
    aspect_ratio = mean(bbox_info(ndx,3));
    areas = bbox_info(ndx,4);
    areas = sort(areas,'ascend');
    scale = sqrt(floor(areas(ceil(numel(ndx)*0.5))/aspect_ratio));
     
    group(ii).aspect_ratio = aspect_ratio;
    group(ii).height = scale;
    group(ii).width = scale*aspect_ratio;
    group(ii).cluster_instances = sort_ndx(ndx);
end

    