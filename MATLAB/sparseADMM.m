% Math ML 2020
% E. Elhamifar and R. Vidal, “Sparse subspace clustering..."
% Algorithm 2 for sparse coefficient matrix

function C = sparseADMM(Y,alpha= 800, rho = 800, maxIter = 200, eps=2e-4)
%------------------------------------------------------------------
%Referencing http://khoury.neu.edu/home/eelhami/codes.htm
% This function takes a DxN matrix of N data points in a D-dimensional 
% space and returns a NxN coefficient matrix of the sparse representation 
% of each data point in terms of the rest of the points
%Inputs
% Y: DxN data matrix
%constants
% alpha(1,2) -> to calculate lambda(1,2)
% rho -> >0
% thr1: stopping threshold for the coefficient error ||Z-C||
% thr2: stopping threshold for the linear system error ||Y-YZ||
% maxIter: maximum number of iterations of ADMM
% C: NxN sparse coefficient matrix

%Set constants
if (length(alpha) == 1)
    alp_e = alpha(1);
    alp_z = alpha(1);
elseif (length(alpha) == 2)
    alp_e = alpha(1);
    alp_z = alpha(2);
end

%rho = alp_e;

YtY = Y'*Y;
mu_e = min(max(abs(Y),[],1));
mu_z = min(max(abs(YtY - diag(diag(YtY))),[],2)); %max each row

lam_e = alp_e / mu_e;
lam_z = alp_z / mu_z;

%initialize C, A, E, d, Delta
[D, N] = size(Y);
C = zeros(N,N);
A = zeros(N,N);
E = zeros(D,N);
d = zeros(N,1);
Delta = zeros(N,N);

c1 = lam_z*YtY+rho*eye(N,N)+rho*ones(N,N);
for i = 1:1:maxIter
    E_prev = E;
    A_prev = A;

   %update A
   %A = inv(c1)*(lam_z * (YtY - Y'*E) + rho*(ones(N,N) + C) - ones(N,1)*d' - Delta); %Paper code
   
   A = inv(c1)* (lam_z*(YtY)+rho*(C-Delta/rho)+rho*ones(N,1)*(ones(1,N)-d'/rho)); %(updated? authors code, slightly better)
   
   %update C
   C = shrinkThresh(A+Delta/rho, 1/rho);
   C = C - diag(diag(C));
   
   %update E
   E = shrinkThresh( (Y - Y*A), lam_e/lam_z);
   
   %update little delta, d
   at_ones = A'*ones(N,1) - 1;
   d = d + rho*(at_ones);
   
   %update Delta
   Delta = Delta + rho*(A-C);
   
   %Set break threshold
   stop = [max(at_ones(:)), max((A-C)(:)), max((A-A_prev)(:)), max((E-E_prev)(:))];
   if max(stop) <= eps
       break
   end
end
end

function s = shrinkThresh(v,c)
    %Inputs
    %v = input matrix
    %c = thresholding value
    s = max(0,(abs(v) - c*ones(size(v)))) .* sign(v);
end