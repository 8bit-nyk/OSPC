clc;clear;


%image_folder_path = '/home/aub/datasets/Pcalib/basler_dataset2_desk_240104';
%exposure_file_path = '/home/aub/datasets/Pcalib/basler_dataset2_desk_240104/exposure_data.txt'; 
crf_file_path = "/home/aub/Dev/OSPC/outputs/crf_ospc_d435.txt";
image_folder_path = '/home/aub/datasets/Pcalib/d435/d435_vignetteSweep/d435_vignetteSweep/';
exposure_file_path = '/home/aub/datasets/Pcalib/d435/d435_vignetteSweep/d435_vignetteSweep_values.txt'; 
 


in= 0:1/1000:1 ;
output = estimate_vignette_refactored(image_folder_path, exposure_file_path,crf_file_path);
%groundtruth = readmatrix('/home/aub/datasets/Pcalib/basler_vignette_240105/vignette_results.txt','Range','A1:B200');

figure;
%plot(groundtruth(:,1),groundtruth(:,2),'Xb',in,output, 'red');
plot(in,output, 'red');