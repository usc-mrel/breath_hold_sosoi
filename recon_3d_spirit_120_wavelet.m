addpath ./mfile/

ccc

%% use GPU or not
ifGPU = 1;

%% recon parameters
calib_size  = [30, 30, 20];
kernel_size = [5, 5, 5];
tyk_param   = 0.02;
niter       = 100;
lambda      = 2;
lambda_wavelet = 0.002;
fov_recon = [480, 480] * 1.25; %[mm]
coil_drop = [];

%% raw and recon data dir
all_mat = dir('./raw_data/*out_in*120*.mat');
if ~isfolder('./recon_data')
    mkdir recon_data
end

%% recon script
for ii = 1:length(all_mat)

file_path = fullfile(all_mat(ii).folder,all_mat(ii).name);
load(file_path)

nos_one = kspace_info.user_nKzEncoding * kspace_info.user_interleaves;

res = [kspace_info.user_ResolutionX, kspace_info.user_ResolutionY];
matrix_size = round(fov_recon ./ res / 2) * 2; 
matrix_keep = round([kspace_info.user_FieldOfViewX, kspace_info.user_FieldOfViewX] ./ res / 2) * 2; 
sz = kspace_info.user_nKzEncoding;

imsize = [matrix_size, sz];

kspace = single(kspace) * 1e3;

nsample = size(kspace, 1);
nos     = size(kspace, 2);
ncoil   = size(kspace, 3);
nkxy    = kspace_info.user_interleaves;

%% drop coil
if exist('coil_drop')
    kspace(:, :, coil_drop) = [];
    ncoil = size(kspace, 3);
end

%% svd on coil
if  ncoil > 6
    kspace = reshape(kspace, [nsample * nos, ncoil]);
    [~, ~, v] = svd(kspace, 0);
    kspace = kspace * v;
    ncoil = 6;
    kspace = kspace(:, 1:ncoil);
    kspace = reshape(kspace, [nsample, nos, ncoil]);
end

%% kspace trajectory
kx = kspace_info.kx_meas * imsize(1);
ky = kspace_info.ky_meas * imsize(2);

kz = kspace_info.RFIndex + 1;
w = kspace_info.DCF(:, 1);

view_order = kspace_info.viewOrder;

%% drop dummy arms in begining
narm_drop = kspace_info.user_nDummyPulses;
kspace(:, 1:narm_drop, :) = [];

kz(:, 1:narm_drop) = [];
view_order(1:narm_drop) = [];
nos = size(kspace, 2);

%% pick spiral out
kx_out = kx(1:nsample/2, :);
ky_out = ky(1:nsample/2, :);
kspace_out = kspace(1:nsample/2, :, :);
DCF_out = kspace_info.DCF(1:nsample/2, :);
nsample_out = size(kspace_out, 1);

%% pick spiral in
kx_in = kx(nsample/2+1:end, :);
ky_in = ky(nsample/2+1:end, :);
kspace_in = kspace(nsample/2+1:end, :, :);
DCF_in = kspace_info.DCF(nsample/2+1:end, :);
nsample_in = size(kspace_in, 1);

clear kspace kx ky

%% sort kspace into 3d
nframe = 1;

kspace_3d_out = zeros(nsample_out, nkxy, sz, nframe, ncoil, 'single');
kspace_3d_in  = zeros(nsample_in,  nkxy, sz, nframe, ncoil, 'single');

% put the spirals at correcte location. This should also work for random
% order.
for i = 1:nos
    slice_idx = kz(i);
    frame_idx = ceil(i/nos_one);
    kxy_idx   = view_order(i);

    kspace_3d_out(:,kxy_idx,slice_idx,frame_idx,:) = kspace_out(:,i,:);
    kspace_3d_in (:,kxy_idx,slice_idx,frame_idx,:) = kspace_in (:,i,:);
end
%% estimate 3d SPIRIT kernel
%% slice by slice NUFFT
image_3d_out = zeros([imsize, ncoil], 'single');
image_3d_in  = zeros([imsize, ncoil], 'single');

% all the kz prtition has the same sample pattern, so we only need one

N_out = NUFFT.init(kx_out, ky_out, 1, [4,4], matrix_size(1), matrix_size(1));
N_out.W = DCF_out(:, 1);
N_in = NUFFT.init(kx_in, ky_in, 1, [4,4], matrix_size(1), matrix_size(1));
N_in.W = DCF_in(:, 1);
for islice = 1:sz
    image_3d_out(:, :, islice, :) = NUFFT.NUFFT_adj(kspace_3d_out(:, :, islice, :), N_out);
    image_3d_in (:, :, islice, :) = NUFFT.NUFFT_adj(kspace_3d_in (:, :, islice, :), N_in );
end

%% Cartesian kspace (only using kspace out for calibration)
kspace_cart_3d = fftshift2(fft2(fftshift2(image_3d_out)));

%% calibration kspace 

calib_range_x =  matrix_size(1)/2 - calib_size(1)/2 : matrix_size(1)/2 + calib_size(1)/2 - 1;
calib_range_y =  matrix_size(2)/2 - calib_size(2)/2 : matrix_size(2)/2 + calib_size(2)/2 - 1;
calib_range_z =  sz/2 - calib_size(3)/2 : sz/2 + calib_size(3)/2 - 1;
kspace_calib  = kspace_cart_3d(calib_range_x, calib_range_y, calib_range_z, :);

%% sensitivity map (only using kspace out for sens map)
kspace_sens = kspace_calib .* hamming(calib_size(1)) .* hamming(calib_size(2))' .* permute(hamming(calib_size(3)), [3, 2, 1]);
im_sens = zeros([matrix_size, sz, ncoil]);
im_sens(calib_range_x, calib_range_y, calib_range_z, :) = kspace_sens;
im_sens = fftshift3(ifft3(fftshift3(im_sens)));
sens_map = get_sens_map(permute(im_sens, [1, 2, 3, 5, 4]), '3D');

%% calibrate kernel
kernel = calibSPIRiT(kspace_calib, kernel_size, ncoil, tyk_param);

%% image space kernel
%% takes too much memory. Consider other approach.
% ok now we do coil compression and reduce the recon FOV, try to make it
% work
image_kernel = zeros([matrix_size, sz, ncoil, ncoil]);
kernel_pos_x = round(matrix_size(1)/2 - kernel_size(1)/2 + 1 : matrix_size(1)/2 + kernel_size(1)/2);
kernel_pos_y = round(matrix_size(2)/2 - kernel_size(2)/2 + 1 : matrix_size(2)/2 + kernel_size(2)/2);
kernel_pos_z = round(sz/2 - kernel_size(3)/2 + 1 : sz/2 + kernel_size(3)/2);
image_kernel(kernel_pos_x, kernel_pos_y, kernel_pos_z, :, :) = kernel(end:-1:1, end:-1:1, end:-1:1, :, :);
image_kernel = fftshift3(ifft3(fftshift3(image_kernel))) .* prod([matrix_size, sz]);

%% fft along slice 
image_3d_out = fftshift(image_3d_out, 3);
image_3d_out = ifft(image_3d_out, [], 3);
image_3d_out = fftshift(image_3d_out, 3);

image_3d_in = fftshift(image_3d_in, 3);
image_3d_in = ifft(image_3d_in, [], 3);
image_3d_in = fftshift(image_3d_in, 3);

%% iterative SPIRIT
clear kspace_cart_3d

image_out = cgNUSPIRiT_3d_l1_wavelet(kspace_3d_out, image_3d_out, N_out, image_kernel, niter, lambda, lambda_wavelet, ifGPU);
image_in  = cgNUSPIRiT_3d_l1_wavelet(kspace_3d_in , image_3d_in , N_in , image_kernel, niter, lambda, lambda_wavelet, ifGPU);

image_out = sum(image_out .* conj(squeeze(sens_map)), 4);
image_out = fliplr(rot90(image_out, -1));
image_out = crop_half_FOV(image_out, matrix_keep);

image_in = sum(image_in .* conj(squeeze(sens_map)), 4);
image_in = fliplr(rot90(image_in, -1));
image_in = crop_half_FOV(image_in, matrix_keep);
%% save
save(sprintf('./recon_data/%s_spirit_wavelet.mat', all_mat(ii).name(1:end-4)), 'image_out', 'image_in', 'kspace_info')

end





