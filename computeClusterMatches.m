function computeClusterMatches(feats,h,w,cnn_model,frame_names,save_dir)
    
match_vals = zeros(size(feats,1),numel(frame_names),'single');
match_ndxs = zeros(size(feats,1),numel(frame_names),'uint16');
     
tic;
gfeat1 = gpuArray(feats);
toc;

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

    tic; 
    D1 = feats*pyra.featMat;
    toc;
    
    tic; 
    gfeat2 = gpuArray(pyra.featMat);
    D = gather(gfeat1 * gfeat2);
    toc;
    
    [match_val,match_ndx] = max(D,[],2);

    match_vals(:,ii) = match_val;
    match_ndxs(:,ii) = uint16(match_ndx);
    
    boxes = [pyra.featPos(2,match_ndx); pyra.featPos(1,match_ndx); ...
            pyra.featPos(2,match_ndx)+w-1; pyra.featPos(1,match_ndx)+h-1; pyra.featLevel(match_ndx)]';  
        
    if mod(ii,100)==1    
        fprintf('done with %d/%d\n\n',ii,numel(frame_names));        
        hrs_left = (numel(frame_names)-ii)*(toc(t1)/ii)/60/60;
        fprintf('estimated hrs left: %f\n',hrs_left);
    end
end

save([save_dir 'cluster_matches.mat'], 'match_vals','match_ndxs');