function patch_out = covSparse(patch_in)
    
    patch_in  = double(patch_in);
    %patch_in_mean = repmat(mean( patch_in, 2 ), 1, size(patch_in, 2));
    
   %patch_in = patch_in - patch_in_mean;
    
  %  patch_in_cov= patch_in * patch_in'/(size(patch_in, 2)-1);
    [U, S, V] = svd(patch_in);

options.WeightMode = 'HeatKernel';options.t = 1;
Wc = constructW(patch_in', options);
Wr = constructW(patch_in, options);
Lc = graph_laplacian(Wc); 
Lr = graph_laplacian(Wr);
params = struct();params.Y = patch_in';
params.D = U;
params.T = 3;
params.alpha = 1;params.Lr = Lr;
params.beta =1;
params.Lc = Lc;
params.Z = omp(U,patch_in',3);
X = GRSC_ADMM(params);
 
    patch_in = U * X *V';
    patch_out = patch_in;
    

end
