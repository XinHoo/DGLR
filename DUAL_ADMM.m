function X = DUAL_ADMM(params)

if ~exist('omp','file')
    error('OMP Package missing!');
end

if isfield(params,'Y')
    Y = params.Y;
else
    error('Input data matrix Y missing!');
end
if isfield(params,'D')
    D = params.D;
else
    error('Input dictionary D missing!');
end

if isfield(params,'T')
    T = params.T;
else
    error('Sparsity constraint T missing!');
end

if isfield(params,'Lc')
    Lc = params.Lc;
else
    error('Manifold Laplacian Lc missing!');
end

if isfield(params,'beta')
    beta = params.beta;
else
    error('Regularizaion coefficient beta missing!');
end

if isfield(params,'iternum')
    iternum = params.iternum;
else
    iternum = 25;
end

if isfield(params,'rho')
    rho = params.rho;
else
    rho = 1;
end



M = size(Y,2); 
K = size(D,2);
U = zeros(K,M);
Z = U;
A=D'*D+rho*eye(size(D'*D))+alpha*Lr;%
B=beta*Lc;
C=D'*Y+rho*(Z-U);
%ADMM
for i = 1:iternum 
    %求 Sylvester 方程 AX + XB = C 的 X 解 
    %X = sylvester(full(D'*D+rho*eye(size(D'*D))+alpha*Lr),full(beta*Lc),full(D'*Y+rho*(Z-U)));
    X = sylvester(full(A),full(B),full(C));
    Z = SpProj(X+U,T);
    U = U+X-Z;
end


end

function Z = SpProj(XU,T)
Z = zeros(size(XU));
for j=1:size(Z,2)
    [~,ind] = sort(abs(XU(:,j)),'descend');
    Z(ind(1:T),j) = XU(ind(1:T),j);
end
end
