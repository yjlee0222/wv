function computeClusterMatches2(feats,pyra_size,h,w,cnn_model,frame_names,save_prefix)
    
frame_ndxs = 1:4:numel(frame_names);

match_vals = zeros(size(feats,2),numel(frame_ndxs),'single');
match_ndxs = zeros(size(feats,2),numel(frame_ndxs),'uint16');
boxes = zeros(size(feats,2),numel(frame_ndxs),5,'single');
     
gfeat1 = gpuArray(feats');

t1 = tic;
for ii=1:numel(frame_ndxs)
    im = imread(frame_names{frame_ndxs(ii)});
    
%     th = tic;   
    pyra = deep_pyramid(im, cnn_model);
    pyra = deep_pyramid_add_padding(pyra, 0, 0);
%     fprintf('deep_pyramid took %.3fs\n', toc(th));
%     th = tic;
    pyra = pyramid2Mat(pyra,h,w,1);
%     fprintf('pyramid2mat took %.3fs\n', toc(th));

%     tic; 
%     D = feats*pyra.featMat;
%     toc;
    
%     tic; 
    gfeat2 = gpuArray(pyra.featMat);
    D = gather(gfeat1 * gfeat2);
%     toc;
    
    [match_val,match_ndx] = max(D,[],2);

    match_vals(:,ii) = match_val;
    match_ndxs(:,ii) = uint16(match_ndx);
    
    this_box = single([pyra.featPos(2,match_ndx); pyra.featPos(1,match_ndx); ...
            pyra.featPos(2,match_ndx)+pyra_size(:,2)'-1; pyra.featPos(1,match_ndx)+pyra_size(:,1)'-1]');
    boxes(:,ii,:) = [this_box pyra.scales(pyra.featLevel(match_ndx))];  
        
    if mod(ii,10)==1    
        fprintf('\n\ndone with %d/%d\n',ii,numel(frame_ndxs));        
        hrs_left = (numel(frame_ndxs)-ii)*(toc(t1)/ii)/60/60;
        fprintf('estimated hrs left: %f\n\n',hrs_left);
    end
end

save('-v7.3',[save_prefix '_cluster_matches_resize.mat'], 'match_vals','match_ndxs','boxes','frame_ndxs');