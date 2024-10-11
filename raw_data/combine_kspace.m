%% combine kspace
% github only allows file size up to 25Mb, so I split the kspace data into
% parts and combine them using this code. This is a simple trick. 

kspace = [];
for i = 1:14
    load(sprintf('./usc_disc_yt_2022_12_09_124944_lung_3d_stack_of_out_in_120_slices_2mm_iso_int_120_part_%02d.mat', i), "kspace_")
    kspace = cat(2, kspace, kspace_);
end

load('./usc_disc_yt_2022_12_09_124944_lung_3d_stack_of_out_in_120_slices_2mm_iso_int_120_kspace_info.mat')

save('./usc_disc_yt_2022_12_09_124944_lung_3d_stack_of_out_in_120_slices_2mm_iso_int_120.mat', "kspace", "kspace_info")

!rm *part* *info*