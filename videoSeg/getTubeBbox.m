function tube_bbox = getTubeBbox(dense_seg)
                  
dense_seg = rgb2gray(dense_seg);            
unique_labels = double(unique(dense_seg));
unique_labels(unique_labels==0) = []; % remove background seg           

tube_bbox = zeros(numel(unique_labels),5);
for ii=1:numel(unique_labels)
    temp_img = (dense_seg==unique_labels(ii));                
    CC = bwconncomp(temp_img,8); 
    seg_size = cellfun(@numel,CC.PixelIdxList,'UniformOutput',true);

    [max_val, max_ndx] = max(seg_size);
    % if max connected component is too small, just use all pixels
    if max_val<=10
        [y,x] = find(temp_img==1);
    else
        [y,x] = ind2sub(size(dense_seg),CC.PixelIdxList{max_ndx});
    end        

    tube_bbox(ii,:) = [min(x) min(y) max(x) max(y) unique_labels(ii)];
end