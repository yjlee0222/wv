function tracks = createTubesFromSparseSeg(sparse_seg_dir,sample_rate)

clear tracks;
sparse_segs = dir([sparse_seg_dir '*.ppm']);    
    
se1 = strel('square',2);
se2 = strel('square',5);

for ii=1:numel(sparse_segs)           
    sparse_img = imread([sparse_seg_dir sparse_segs(ii).name]);
    [nr,nc,~] = size(sparse_img);
    
    gray_sparse_img = rgb2gray(sparse_img);
    gray_sparse_img(gray_sparse_img==255) = 0;

    sparse_img_small = zeros([floor(size(gray_sparse_img)/8) 5]);  
    rows = sample_rate/2+1:sample_rate:sample_rate/2+1+sample_rate*(size(sparse_img_small,1)-1);
    cols = sample_rate/2+1:sample_rate:sample_rate/2+1+sample_rate*(size(sparse_img_small,2)-1);
    count = 1;
    for jj=-2:1:2
        sparse_img_small(:,:,count) = gray_sparse_img(rows+jj,cols+jj,:);
        count = count + 1;
    end
    sparse_img_small = max(sparse_img_small,[],3);

    unique_labels = unique(sparse_img_small);
    unique_labels(unique_labels==0) = [];

    final_sparse_img_small = zeros(size(sparse_img_small));
    label_num = 1;
    labels = [];
    for jj=1:numel(unique_labels)
        temp_img = (sparse_img_small==unique_labels(jj));
        temp_img = imdilate(temp_img,se1);
        temp_img = imerode(temp_img,se1);

        CC = bwconncomp(temp_img,8); 
        temp_img = zeros(size(temp_img));
        for kk=1:CC.NumObjects
            if numel(CC.PixelIdxList{kk})<=1
                continue;
            end
            temp_img(CC.PixelIdxList{kk}) = 1;
        end
        ndx = find(temp_img);
        if isempty(ndx)
            continue;
        end
        temp_img = imdilate(temp_img,se2);                                               

        CC = bwconncomp(temp_img,8);  
        num_pixels = zeros(CC.NumObjects,1);
        for kk=1:CC.NumObjects
            num_pixels(kk) = numel(CC.PixelIdxList{kk});
        end
        [~,max_ndx] = max(num_pixels);
        intersect_pixels = intersect(CC.PixelIdxList{max_ndx},ndx);

        final_sparse_img_small(intersect_pixels) = label_num;
        labels = [labels; unique_labels(jj)];
        label_num = label_num + 1;
    end
    final_sparse_img = upsample(final_sparse_img_small,sample_rate);
    final_sparse_img = upsample(final_sparse_img',sample_rate)';

    unique_labels = unique(final_sparse_img);
    unique_labels(unique_labels==0) = [];

    bbox_info = zeros(numel(unique_labels),5);
    for jj=1:numel(unique_labels)
        [yy,xx] = find(final_sparse_img==unique_labels(jj));
        x1 = max(1,min(xx)-2);
        y1 = max(1,min(yy)-2);
        x2 = min(max(xx)+2,nc);
        y2 = min(max(yy)+2,nr);    

        bbox_info(jj,:) = [x1 y1 x2 y2 labels(jj)];                
    end    

    tracks(ii).bbox_info = bbox_info;
end