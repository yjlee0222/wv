clear;

addpath(genpath('/home/yjlee/projects/weakVideo/external/nrsfm_nips2014/'));
addpath('/home/yjlee/Downloads/FastVideoSegment/Code/External/sundaramECCV2010');

datadir = '/home/SSD1/yjlee-data/projects/weakVideo/YouTube-Objects/car/data/';

nclusters=20;

d = dir(datadir);
d = d(3:end);

th=tic;
for ii=1:1%numel(d)
    if isdir([datadir d(ii).name])
        dd = dir([datadir d(ii).name '/shots']);
        dd = dd(3:end);
        
        for jj=1:1%numel(dd)
            video_dir = [datadir d(ii).name '/shots/' dd(jj).name '/'];
            imnames = get_file_list(video_dir,'jpg');
        
            ddd = dir([datadir d(ii).name '/shots/' dd(jj).name '/*.jpg']);
            
            
            flow_dir=setdir([video_dir 'flow/']);            
            ts=cell2struct(num2cell([1:length(imnames)]),'t',1);
            imnames = cell2struct([struct2cell(imnames); struct2cell(ts')], ...
                [fieldnames(imnames); fieldnames(ts)], 1);

            [para]=get_para(video_dir, 'jpg',length(imnames),4);

            disp('compute dense trajectories');
% %             computeFlowLDOF(imnames,flow_dir,para)
%             computeFlowLDOF_gpu(imnames,flow_dir)
            load([flow_dir 'flows.mat'],'forward_flow','backward_flow');

            tr=linkFlowLDOF(imnames,para,forward_flow,backward_flow);

            lens=get_tr_lengths(tr);
            tr=tr(lens>=5);
            if 0
                plot_trajectory_labels(tr, ones(length(tr),1), imnames(1:5:end),1,[],0,4);
            end

            disp('compute trajectory affinities');
            [Atr]=computeTrAffinities(tr,para,0);

            disp('compute trajectory clustering');
            [Vtr,Str] = ncut(Atr,para.nv);

            binsoltr = getbinsol(Vtr(:,1:nclusters));
            tr_labels = full(sum(binsoltr.*repmat(1:nclusters, size(binsoltr,1),1),2)')';
            h=plot_trajectory_labels(tr,tr_labels,imnames(1:10:end),nclusters,[],1,5);

        end
    end
end
toc(th);
