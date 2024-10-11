function [x_old, norm, s_norm, f_norm] = cgNUSPIRiT_3d_l1_wavelet(kspace, x0, N, kernel, niter, lambda, lambda_wavelet, ifGPU)

% Implementation of image-domain SPIRiT reconstruction from arbitrary
% k-space. The function is based on Jeff Fessler's nufft code and LSQR 
% 
% Inputs: 
%       kData   - k-space data matrix it is 3D corresponding to [readout,interleaves,coils] 
%       x0      - Initial estimate of the coil images
%       NUFFTOP - nufft operator (see @NUFFT class)
%       GOP     - SPIRiT Operator (See @SPIRiT)
%       nIter   - number of LSQR iterations
%       lambda  - ratio between data consistency and SPIRiT consistency (1 is recommended)
%
% Outputs:
%       res - reconstructed coil images
%       FLAG,RELRES,ITER,RESVEC,LSVEC - See LSQR documentation
%
% See demo_nuSPIRiT for a demo on how to use this function.
%
% (c) Michael Lustig 2006, modified 2010

% Midified by Ye Tian for 3D non-Cartesian SPIRiT
% phye1988@gmail.com

%% 
N.W     = sqrt(N.W);
kspace  = kspace .* N.W;
sx      = size(x0, 1);
sy      = size(x0, 2);
sz      = size(x0, 3);
x0      = x0 * sqrt(sx * sy * sz);

undersample_mask = logical(abs(kspace(1, :, :, :, 1)));

%%
imSize      = size(x0);
dataSize    = size(kspace);
x_old       = x0;
step        = 0.35;
lambda_wavelet   = max(abs(x0(:))) * lambda_wavelet;

if ifGPU
    x_old   = gpuArray(x_old);
    N.S     = gpuArray(N.S);
    y1      = gpuArray(zeros(dataSize, 'single'));
    kernel  = gpuArray(single(kernel));
end

s_norm = zeros([1, niter]);
f_norm = zeros([1, niter]);
norm   = zeros([1, niter]);

for i = 1:niter
    tic
    %% spirit update
    s_update = squeeze(sum(kernel .* x_old, 4)) - x_old;
    
    s_norm(i)= sum(abs(s_update(:)).^2) * lambda;
    
    s_update = sum(conj(kernel) .* permute(s_update, [1, 2, 3, 5, 4]), 5) - s_update;
    
    %% fidelity update
    f_update = fftshift(x_old, 3);
    f_update = fft(f_update, [], 3);
    f_update = fftshift(f_update, 3);
    
    for iz = 1:dataSize(3)
        y1(:, :, iz, :, :) = NUFFT.NUFFT(f_update(:, :, iz, :), N);
    end
    y1 = y1 .* undersample_mask / sqrt(prod(imSize(1:3))) .* (N.W);
    y1 = kspace - y1;
    
    f_norm(i)= sum(abs(y1(:)).^2);
    
    for iz = 1:imSize(3)
        f_update(:, :, iz, :) = NUFFT.NUFFT_adj(y1(:, :, iz, :), N);
    end
    f_update = fftshift(f_update, 3);
    f_update = ifft(f_update, [], 3);
    f_update = fftshift(f_update, 3);
    f_update = f_update * sqrt(prod(imSize(1:3)));
    
    %% 3d wavelet
    clear wavelet_decom
    for icoil = 1:size(x_old, 4)
        wavelet_decom{icoil} = wavedec3(x_old(:, :, :, icoil), 2, 'db1');
    end
    n_wavelet_comp = length(wavelet_decom{1}.dec);
    
    for iwavelet = 1:n_wavelet_comp
        temp = wavelet_decom{1}.dec{iwavelet};
        for icoil = 2:size(x_old, 4)
            temp = cat(4, temp, wavelet_decom{icoil}.dec{iwavelet});
        end
        
        abs_temp = abs(sos(temp));
        unit_temp = temp ./ (abs_temp + eps);
        
        res_temp = abs_temp - lambda_wavelet;
        res_temp(res_temp < 0) = 0;
        temp = res_temp .* unit_temp;
        
        for icoil = 1:size(x_old, 4)
            wavelet_decom{icoil}.dec{iwavelet} = temp(:, :, :, icoil);
        end
        
    end
    
    for icoil = 1:size(x_old, 4)
        image_update(:, :, :, icoil) = waverec3(wavelet_decom{icoil});
    end
    wavelet_update = image_update - x_old;
%     tv_update = compute_sTV_3D_yt(circshift(x_old, [0, 0, 10, 0]), tv_weight, eps('single'));
%     tv_update = circshift(tv_update, [0, 0, -10, 0]);
    
    %% cg update
%     x_new = x_old - s_update * 0.1 + f_update * 0.1;
%     x_old = x_new;
    norm(i) = f_norm(i) + s_norm(i);
    
    update_term = (f_update - s_update * lambda + wavelet_update);
    
    if i > 1
        beta = update_term(:)'*update_term(:)/(update_term_old(:)'*update_term_old(:)+eps('single'));
        update_term = update_term + beta*update_term_old;
        if norm(i) > norm(i-1)
            step = step / 2;
            step
        end
    end
    
    x_old = x_old + update_term * step;
    update_term_old = update_term;
    
    
    figure(1)
    imagesc(abs(x_old(:,:,40)))
    axis image
    title(i)
    drawnow
    
    figure(2)
    clf
    loglog(norm)
    hold on
    plot(f_norm)
    plot(s_norm)

    toc
end
x_old = gather(x_old);