function match = retrievekBestMatches(detector,pyramid,patchSize,k,overlap)
% retrieves top k matches per scale and optionally performs non-max suppression

D = detector.hog'*pyramid.featMat;

[matchVal,matchNdx] = sort(D,'descend');
matchNdx = matchNdx(1:k);
score = matchVal(1:k);
featPos = pyramid.featPos(:,matchNdx);    
scale = pyramid.featScale(matchNdx);

if (exist('overlap','var')) && (k>1)
    % non-max suppression
    x1 = featPos(2,:).*pyramid.sBin+1; y1 = featPos(1,:).*pyramid.sBin+1;
    boxes = [x1' y1' x1'+patchSize-1 y1'+patchSize-1 score']; % x1 y1 x2 y2 score

    [~,pick] = esvm_nms(boxes, overlap);
    score = score(pick);
    featPos = featPos(:,pick);    
    scale = scale(pick);
end
imPos = featPos.*pyramid.sBin+1;
    
match = struct('score', num2cell(score), 'imPos', num2cell(imPos,1), ...
    'featPos', num2cell(featPos,1), 'scale', num2cell(scale));