function [stopFlag] = checkClusterOverlap(clusters,numPatchThresh,ovThresh)

if nargin==1
    numPatchThresh = 5;
    ovThresh = 0.25;
end

thisNdx = numel(clusters);
for ii=1:thisNdx-1
    imgIntersect = intersect(clusters(ii).imNdx,clusters(thisNdx).imNdx);
    intCount = 0;
    stopFlag = 0;
    if numel(imgIntersect) > numPatchThresh
        for kk=1:numel(imgIntersect)
            ndx1 = (clusters(ii).imNdx==imgIntersect(kk));
            ndx2 = (clusters(thisNdx).imNdx==imgIntersect(kk));
            overlaps = computeOverlap(clusters(ii).boxes(ndx1,:), clusters(thisNdx).boxes(ndx2,:));
            intCount = intCount + numel(find(max(overlaps,[],1)>ovThresh));
            if intCount > numPatchThresh
                stopFlag = 1;
                break;
            end
        end
    end
    if stopFlag == 1
        break;
    end
end