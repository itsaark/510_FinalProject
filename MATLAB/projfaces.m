% Math ML Winter 2020
% Final Project on Sparse Subspace Clustering (2013 Elhamifar,Vidal)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load Benchmark Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear 
close all
load YaleBCrop025.mat

nClust = 2;
s = [];

for ii = 1:nClust
    s = [s ii*ones(1,64)];
end

tic
    idx = Ind{nClust};
    for j = 1:size(idx,1)
        X = [];
        for p = 1:nClust
            X = [X Y(:,:,idx(j,p))];
        end        
        [D,N] = size(X);
    end
toc

disp("norm of X - XC");
C = sparseADMM(X, alpha=1, rho=800);
norm(X-X*C)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%CREATE SIMILARITY GRAPH
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Normalize by column max, and create similarity graph
%C = C ./max(C);
C = normc(C);
W = abs(C) + abs(C)';

%Threshold W in rows and columns to top q values (or median+c)

%MEDIAN
t = 0.05; %threshold value over median
W2 = W.*(W> (median(W(:))+t) ) + 1e-6; %add constant so eig doesn't get Inf/NaN warning message

%KEEP q
%q = fix(size(W,1)/max(s)); %Setting q to a percentage of N
%W2 = threshk(W, q*6 );
figure
imagesc(W);
figure
imagesc(W2);
title('Thresholded Affinity matrix');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SPECTRAL CLUSTERING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%USAGE>> [estLabels,Y] = mySpectralClustering(W,K,normalized)
[estLabels, _] = mySpectralClustering(W,2,1);
[estLabels2, _] = mySpectralClustering(W2,2,1);

s = s';
[err estLabels] = missRate(s, estLabels);
disp(['err =' num2str(err)])
[err2 estLabels2] = missRate(s, estLabels2);
disp(['err2(thresholding) =' num2str(err2)])


figure
plot(s,'g'); hold on
plot(estLabels2,'r');
legend('true', 'predicted');
