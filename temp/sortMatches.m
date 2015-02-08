function sortMatches(param)

matchImgNames = [];
allYears = zeros(param.numTrainImages,1);
count = 1;
for matchDec = 1:numel(param.decRange)    
    d = param.trainImages{matchDec};
    matchImgNames = [matchImgNames; d];    
    for ii=1:numel(d)
        allYears(count) = d(ii).year; 
        count = count + 1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% keep info about all detectors (sampled patches) in all query images
count = 1;
matchYears = zeros(param.numTopMatches, param.numTrainImages/10*param.numPatchesPerScale*numel(param.scales),'uint16');
matchNdx = zeros(param.numTopMatches, param.numTrainImages/10*param.numPatchesPerScale*numel(param.scales),'uint16');
keepInfo = zeros(param.numTopMatches*2+2, param.numTrainImages/10*param.numPatchesPerScale*numel(param.scales),'uint16');
keepScales = zeros(param.numTopMatches, param.numTrainImages/10*param.numPatchesPerScale*numel(param.scales),'single');
for queryDec = 1:numel(param.decRange)
    queryImgNames = param.trainImages{queryDec};
    for queryImg=1:10:numel(queryImgNames)
        finalMatchScores = [];
        finalMatchScales = [];
        finalMatchYpos = [];
        finalMatchXpos = [];
        
        for matchDec = 1:numel(param.decRange) 
            savename = [param.matchdir 'queryDec=' num2str(queryDec) '_queryImg=' num2str(queryImg) '_matchDec=' num2str(matchDec) '.mat'];
            load(savename,'matchScores','matchScales','matchYpos','matchXpos');
            finalMatchScores = [finalMatchScores; matchScores];
            finalMatchScales = [finalMatchScales; matchScales];
            finalMatchYpos = [finalMatchYpos; matchYpos];
            finalMatchXpos = [finalMatchXpos; matchXpos];
        end

        % get info about top 50 matches across entire dataset for each detector
        [~,sortedNdx] = sort(finalMatchScores,1,'descend');
        for jj=1:size(sortedNdx,2)  
            matchYears(:,count) = allYears(sortedNdx(1:param.numTopMatches,jj));  
            matchNdx(:,count) = sortedNdx(1:param.numTopMatches,jj);  
            keepInfo(1:2,count) = [queryDec queryImg]';
            keepInfo(3:param.numTopMatches+2,count) = finalMatchYpos(sortedNdx(1:param.numTopMatches,jj),jj);
            keepInfo(param.numTopMatches+3:param.numTopMatches*2+2,count) = finalMatchXpos(sortedNdx(1:param.numTopMatches,jj),jj);
            keepScales(:,count) = finalMatchScales(sortedNdx(1:param.numTopMatches,jj),jj);
            count = count + 1;
        end      
    end
end
matchYears(:,count:end) = [];
matchNdx(:,count:end) = [];
keepInfo(:,count:end) = [];
keepScales(:,count:end) = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% compute entropy based on year distribution
yearDist = 1920:1999;
matchHist = histc(matchYears,yearDist,1);
matchAvgHist = cellfun(@(x) conv(x,ones(5,1),'same'),num2cell(matchHist,1),'UniformOutput',0);
matchAvgHist = cell2mat(matchAvgHist);
matchNormHistYear = bsxfun(@times,matchAvgHist,1./sum(matchAvgHist,1));

buff = log2(matchNormHistYear);
buff(isinf(buff)) = 0;
ent_year = -1*sum(matchNormHistYear.*buff,1)/log2(numel(yearDist));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% find peak response year per query
nn = 0;
clear decCluster;
for thisYear = param.decRange
    nn = nn + 1;
    
    [~,maxYear] = max(matchNormHistYear,[],1);
    maxYear = yearDist(maxYear);
    [~,matchSortedNdx] = sort(ent_year,'ascend');
    maxYear = maxYear(matchSortedNdx);
    keepNdx = find((maxYear>=thisYear)&(maxYear<thisYear+10));
    matchSortedNdx = matchSortedNdx(keepNdx);
 
    clear clusters;
    totClust = 0;
    for jj=1:size(matchSortedNdx,2)
        if totClust >= param.numClustersPerDecade
            break;
        end
        queryYear = keepInfo(1,matchSortedNdx(jj));
        queryImg = keepInfo(2,matchSortedNdx(jj));
        thisMatchYpos = keepInfo(3:param.numTopMatches+2,matchSortedNdx(jj));
        thisMatchXpos = keepInfo(param.numTopMatches+3:param.numTopMatches*2+2,matchSortedNdx(jj));
        thisMatchScales = keepScales(:,matchSortedNdx(jj));

        boxes = single([thisMatchXpos thisMatchYpos thisMatchXpos+param.initPatchSize thisMatchYpos+param.initPatchSize]);
        clusters(jj).boxes = bsxfun(@times, boxes, 1./thisMatchScales);
        clusters(jj).imNdx = matchNdx(:,matchSortedNdx(jj));
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % check overlap to higher-ranked clusters 
        stopFlag = 0;
        if jj>1
            [stopFlag] = checkClusterOverlap(clusters);            
        end
        if stopFlag == 1
            continue;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        totClust = totClust + 1;
            
        if param.DEBUG_FLAG==1
            figure(1); clf;
            set(gcf,'Color',[1 1 1]);
        end
        for ii=1:param.numTopMatches
            matchImgNdx = matchNdx(ii,matchSortedNdx(jj));
            imname = [param.trainimgdir matchImgNames(matchImgNdx).name];    
                        
            year = getYear(imname);
            if year~=allYears(matchImgNdx)
                error('something wrong\n');
            end
            
            decCluster{nn,totClust}(ii).imname = imname;
            decCluster{nn,totClust}(ii).y = thisMatchYpos(ii);
            decCluster{nn,totClust}(ii).x = thisMatchXpos(ii);
            decCluster{nn,totClust}(ii).scale = thisMatchScales(ii);
            decCluster{nn,totClust}(ii).year = year;
            
            if param.DEBUG_FLAG==1 && ii<20
                subplot(5,5,ii); axis tight; axis off;
                set(gca,'FontSize',12,'FontWeight','Bold');

                I = imread(imname);
                y = thisMatchYpos(ii);
                x = thisMatchXpos(ii);
                I = imresize(I,thisMatchScales(ii));
                imshow(I(y:y+param.initPatchSize-1,x:x+param.initPatchSize-1,:));
                title([num2str(year)]);
            end
        end
        if param.DEBUG_FLAG==1
            subplot(5,5,21:25);
            set(gca,'FontSize',15,'FontWeight','Bold');
            bar(yearDist,50*matchNormHistYear(:,matchSortedNdx(jj)));
            pause;
        end        
    end
end
save('-v7',[param.clusterdir 'decCluster.mat'],'decCluster');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
