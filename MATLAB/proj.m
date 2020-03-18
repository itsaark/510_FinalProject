% Math ML Winter 2020
% Final Project on Sparse Subspace Clustering (2013 Elhamifar,Vidal)

clear
close all

%Setting 1-d and 2-d subspaces in 3D space (for visualization)
D = 3; %Dimension of ambient space (increase for sparsity)
d1 = 1; d2 = 1; %d1 and d2: dimension of subspace 1 and 2
N1 = 20; N2 = 20; %N1 and N2: number of points in subspace 1 and 2
X1 = randn(D,d1) * randn(d1,N1); %Generating N1 points in a d1 dim. subspace
X2 = randn(D,d2) * randn(d2,N2); %Generating N2 points in a d2 dim. subspace

d3 = 2; N3 = 50;
X3 = randn(D,d3) * randn(d3,N3); 
X = [X1 X2 X3];
s = [1*ones(1,N1) 2*ones(1,N2) 3*ones(1,N3)];

figure
scatter3(X1(1,:),X1(2,:),X1(3,:),100,'r')
hold all
scatter3(X2(1,:),X2(2,:),X2(3,:),100,'b')
scatter3(X3(1,:),X3(2,:),X3(3,:),100,'g')

disp("norm of X - XC");
C = sparseADMM(X);
norm(X-X*C)