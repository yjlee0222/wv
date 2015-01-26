function pyramid = pyramid2Mat2(pyramid,h,w,normalize)
% converts pyramid to a 2D matrix
% h = height of patch in deep pyramid space
% w = width of patch in deep pyramid space
%
% modified Tomasz Malisiewicz code

numFeats = zeros(1,numel(pyramid.feat));
for ii=1:numel(pyramid.feat)
    [fr,fc,fd] = size(pyramid.feat{ii});
    numFeats(ii) = (fr-h+1)*(fc-w+1);
end
featMat = zeros(h*w*fd,sum(numFeats),'single');
% featPos = zeros(2,sum(numFeats),'uint16');
% featLevel = zeros(1,sum(numFeats),'uint16');

count = 1;
for ii=1:numel(pyramid.feat)
    s = size(pyramid.feat{ii});
    NW = s(1)*s(2);
    ppp = reshape(1:NW,s(1),s(2));
    curf = reshape(pyramid.feat{ii},NW,fd);
    b = im2col(ppp,[h w]);
%     offsets{i} = b(1,:);
%     offsets{i}(end+1,:) = i;
    
    for jj = 1:size(b,2)
        featMat(:,count) = reshape(curf(b(:,jj),:),[],1);
        count = count + 1;
    end
%     [uus{i},vvs{i}] = ind2sub(s,offsets{i}(1,:));
end

if normalize==1
%     X = bsxfun(@minus, featMat(:,1:count-1), mean(featMat(:,1:count-1),1));
    X = featMat(:,1:count-1);
    pyramid.featMat = bsxfun(@times, X, 1./sqrt(sum(X.*X,1)));
else
    pyramid.featMat = featMat(:,1:count-1);
end
% pyramid.featPos = featPos(:,1:count-1);
% pyramid.featLevel = featLevel(:,1:count-1);

