% Math ML Winter 2020
% Final Project on Sparse Subspace Clustering (2013 Elhamifar,Vidal)

clear
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%CREATE SYNTHETIC DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Setting 1-d and 2-d subspaces in 3D space (for visualization)
%D = 3; %Dimension of ambient space (set to 30 for sparsity)
D = 30;
d1 = 1; d2 = 1; %d1 and d2: dimension of subspace 1 and 2
N1 = 20; N2 = 20; %N1 and N2: number of points in subspace 1 and 2
X1 = randn(D,d1) * randn(d1,N1); %Generating N1 points in a d1 dim. subspace
X2 = randn(D,d2) * randn(d2,N2); %Generating N2 points in a d2 dim. subspace
X = [X1 X2];
s = [1*ones(1,N1) 2*ones(1,N2)];


%Adding on a plane
d3 = 2; N3 = 20;
X3 = randn(D,d3) * randn(d3,N3); 
X = [X X3];
s = [s 3*ones(1,N3)];

if D == 3
    figure
    scatter3(X1(1,:),X1(2,:),X1(3,:),100,'r')
    hold all
    scatter3(X2(1,:),X2(2,:),X2(3,:),100,'b')
    scatter3(X3(1,:),X3(2,:),X3(3,:),100,'g')
end
disp("norm of X - XC");
C = sparseADMM(X, alpha=800, rho = 800);
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
t = 0.2; %threshold value over median
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
[estLabels, _] = mySpectralClustering(W,max(s),1);
[estLabels2, _] = mySpectralClustering(W2,max(s),1);

s = s';
[err estLabels] = missRate(s, estLabels);
disp(['err =' num2str(err)])
[err2 estLabels2] = missRate(s, estLabels2);
disp(['err2(thresholding) =' num2str(err2)])


figure
plot(s,'g'); hold on
plot(estLabels2,'r');
legend('true', 'predicted');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ADD NOISE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vars = linspace(0, 0.003, 20); % noise variances
N = size(X,2);
errs = [];


for i = 1:1:length(vars)
    var = vars(i);
    %Xn = X + max(X(:))*repmat(sqrt(var)*randn(1,N),D,1);
    Xn = X + sqrt(var)*randn(D,N);
    C1 = sparseADMM(Xn, alpha=800, rho = 800);
    C1 = normc(C1);
    W = abs(C1) + abs(C1)';
    W2 = W.*(W> (median(W(:))+t) ) + 1e-6;
    [estLabels, _] = mySpectralClustering(W2,max(s),1);
    [err estLabels] = missRate(s, estLabels);
    errs = [errs err];
end

figure
plot(vars, errs);
title('Performance vs noise level');
xlabel('var');
ylabel('classification error');

%BUGGY FUNCTION
function Wth = threshk(W, q)
%Keeps q largest values per row, column
%Input
%W = affinity matrix NxN
%q = (approx) largest number of values to keep per row, column
%Output
%Wth = thresholded W matrix
    N = size(W,1);
    Wth = W;
%Sort rows
    [_, ind] = sort(W,2);
    Wth(ind<N-q+1) = 0; %Only keep largest q values
%Sort columns
    [_, ind] = sort(W,1);
    Wth(ind<N-q+1) = 0; %Only keep largest q values
end
