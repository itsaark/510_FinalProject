function [estLabels,Y] = mySpectralClustering(W,K,normalized);
% Inputs:
%   W: weighted adjacency matrix of size N x N
%   K: number of output clusters
%   normalized: 1 for normalized Laplacian, 0 for unnormalized
% Outputs:
%   estLabels: estimated cluster labels
%   Y: transformed data matrix of size K x N

pkg load statistics

degMat = diag(sum(W,1));
L = degMat - W;

if normalized == 0
    [V,D] = eig(L);
    [~,inds] = sort(diag(D),'ascend');
    Y = V(:,inds(1:K))';
    
    estLabels = kmeans(Y',K,'Replicates',100);
else
    % invert degree matrix
    degInv = diag(1./diag(degMat));
    Ln = degInv*L;

    [V,D] = eig(Ln);
    [~,inds] = sort(diag(D),'ascend');
    Y = V(:,inds(1:K))';
    
    estLabels = kmeans(Y',K,'Replicates',100);
end
