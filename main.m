% Read Image
img_path = './Test_Images/';
files = dir([img_path, '*.png']);

psnr_file = fopen('psnr.txt', 'w');
tic
for i = 1 : length(files)
    
    cim = imread([img_path, files(i).name]);
    
    nim = imnoise(cim, 'gaussian', 0, 15^2/255^2); % Add Nois
    dnim = uint8(GBsimple(nim)); % Group-Based dual graph
    save([files(i).name, '_15.mat'], 'cim', 'nim', 'dnim');
    
    fprintf(psnr_file, files(i).name);
    fprintf(psnr_file, ' 15 nim: %.3f dB  dnim: %.3f dB \n', ...
    psnr(nim, cim), psnr(dnim, cim)),ssim(dnim,cim); 
end
fclose(psnr_file);
toc
