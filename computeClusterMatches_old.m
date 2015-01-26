function computeClusterMatches_old(query_pyra,cnn_model,frame_names,save_dir)

cluster_ndx = query_pyra.cluster_ndx;
group_ndx = query_pyra.group_ndx;
h = query_pyra.level_sizes(1,1);
w = query_pyra.level_sizes(1,2);
    
matchVals = zeros(numel(query_pyra.feat),numel(frame_names),'single');
matchNdxs = zeros(numel(query_pyra.feat),numel(frame_names),'uint16');
    
t1 = tic;
for ii=1:numel(frame_names)
    im = imread(frame_names{ii});
    
%     th = tic;   
    pyra = deep_pyramid(im, cnn_model);
    pyra = deep_pyramid_add_padding(pyra, 0, 0);
%     fprintf('deep_pyramid took %.3fs\n', toc(th));
%     th = tic;
    pyra = pyramid2Mat(pyra,h,w,1);
%     fprintf('pyramid2mat took %.3fs\n', toc(th));

    D = query_pyra.Feats'*pyra.featMat;
    [matchVal,matchNdx] = max(D,[],2);

    matchVals(:,ii) = matchVal;
    matchNdxs(:,ii) = uint16(matchNdx);
    
    if mod(ii,100)==1
%         save([save_dir 'cluster' num2str(cluster_ndx) '_group' num2str(group_ndx) '_chunk' num2str(chunk_count) '.mat'], ...
%         'matchVals','matchNdxs');
    
        fprintf('done with %d/%d\n\n',ii,numel(frame_names));        
        hrs_left = (numel(frame_names)-ii)*(toc(t1)/ii)/60/60;
        fprintf('estimated hrs left: %f\n',hrs_left);
    end
end

save([save_dir 'cluster' num2str(cluster_ndx) '_group' num2str(group_ndx) '.mat'], ...
        'matchVals','matchNdxs');