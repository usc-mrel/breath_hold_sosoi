function [AtA, A, kernel] = dat2AtA(data, kSize)

% [AtA,A,kernel] = dat2AtA(data, kSize)
%
% Function computes the calibration matrix from calibration data. 
%
% (c) Michael Lustig 2013


sx = size(data, 1);
sy = size(data, 2);
sz = size(data, 3);
nc = size(data, 4);

nkx = sx - (kSize(1) - 1);
nky = sy - (kSize(2) - 1);
nkz = sz - (kSize(3) - 1);

nkernel = nkx * nky * nkz;
kernel = zeros([nkernel, prod(kSize), nc]);
i = 1;
for ix = 1:nkx
    for iy = 1:nky
        for iz = 1:nkz
            kernel(i, :, :) = reshape(data(ix:ix+kSize(1)-1, iy:iy+kSize(2)-1, iz:iz+kSize(3)-1, :), [1, prod(kSize), nc]);
            i = i + 1;
        end
    end
end


[tsx, tsy, tsz] = size(kernel);
A = reshape(kernel, [tsx, tsy*tsz]);

AtA = A'*A;

kernel = AtA;
kernel = reshape(kernel, [kSize, nc, tsy*tsz]);
