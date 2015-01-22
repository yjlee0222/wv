function pyramid = pyramid2Mat(pyramid,h,w,normalize)
% converts pyramid to a 2D matrix
% h = height of patch in deep pyramid space
% w = width of patch in deep pyramid space

numFeats = zeros(1,numel(pyramid.feat));
for ii=1:numel(pyramid.feat)
    [fr,fc,fd] = size(pyramid.feat{ii});
    numFeats(ii) = (fr-h+1)*(fc-w+1);
end
featMat = zeros(h*w*fd,sum(numFeats),'single');
featPos = zeros(2,sum(numFeats),'uint16');
featLevel = zeros(1,sum(numFeats),'uint16');


% count = 1;
% for ii=1:numel(pyramid.feat)
%     [fr,fc,~] = size(pyramid.feat{ii});
%     for nn=1:fr-h+1
%         for mm=1:fc-w+1            
%             featMat(:,count) = vec(pyramid.feat{ii}(nn:nn+h-1,mm:mm+w-1,:));
%             featPos(:,count) = uint16([nn mm]');
%             featLevel(count) = ii;
%             count = count + 1;
%         end
%     end
% end

count = 1;
for ii=1:numel(pyramid.feat)
    if numFeats(ii)<=0
        continue;
    end
    [thisFeatMat,thisFeatPos] = extract_pyra_feat(pyramid.feat{ii},h,w,numFeats(ii));        

    featMat(:,count:count+numFeats(ii)-1) = thisFeatMat;
    featPos(:,count:count+numFeats(ii)-1) = thisFeatPos;
    featLevel(count:count+numFeats(ii)-1) = ii;
    count = count + numFeats(ii);
end


if normalize==1
%     X = bsxfun(@minus, featMat(:,1:count-1), mean(featMat(:,1:count-1),1));
    X = featMat(:,1:count-1);
    pyramid.featMat = bsxfun(@times, X, 1./sqrt(sum(X.*X,1)));
else
    pyramid.featMat = featMat(:,1:count-1);
end
pyramid.featPos = featPos(:,1:count-1);
pyramid.featLevel = featLevel(:,1:count-1);

