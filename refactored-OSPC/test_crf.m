%CLEAR
clear all;
% Define paths

%image_folder_path = '/home/aub/datasets/TUM/sequence_30_og/images/';
%exposure_file_path = '/home/aub/datasets/TUM/sequence_30_og/times.txt'; 
 
% image_folder_path = '/home/aub/datasets/TUM-MonoVO/all_calib_sequences/calib_narrow_sweep1/images/';
% exposure_file_path = '/home/aub/datasets/TUM-MonoVO/all_calib_sequences/calib_narrow_sweep1/times.txt'; 
image_folder_path = '/home/aub/datasets/Pcalib/d435/d435_exposureSweep/d435_exposureSweep/';
exposure_file_path = '/home/aub/datasets/Pcalib/d435/d435_exposureSweep/d435_exposureSweep_values.txt'; 

output = ospc_CRF(image_folder_path, exposure_file_path);

%groundtruth = readmatrix('/home/aub/datasets/Pcalib/basler1300/basler1300_macbeth_e7/macbeth-Basler1300-exp2-res.csv','Range','A2:B100');
groundtruth = readmatrix('/home/aub/datasets/Pcalib/d435/d435-macbeth/images-exp8/macbeth-res8-2.csv');
%groundtruth = readmatrix('/home/aub/datasets/TUM/sequence_30_og/pcalib.txt');

normalized_x = (0:255) / 255;
%{
figure;
plot(normalized_x,groundtruth,'green');
title('Ground Truth CRF');
ylabel('Normalized Brightness');
xlabel('Normalized Irradiance');
legend('GT','Location', 'best');
grid on;


figure;
plot(normalized_x,output_ref,'red');
title('OSPC CRF');
ylabel('Normalized Irradiance');
xlabel('Normalized Brightness');
legend('OSPC_ref','Location', 'best');
grid on;      
%}
output255 = output * 255;
%Define path to save output data:
output_path = '/home/aub/Dev/OSPC/outputs/crf_ospc.txt';

% Check if the file exists
if ~exist(output_path, 'file')
    % If the file doesn't exist, create it
    fid = fopen(output_path, 'w');
    fclose(fid);
end
writematrix(output255, output_path, 'Delimiter', '\t');
figure;
plot(groundtruth(:,2),groundtruth(:,1) ,'Xb',normalized_x,output,'red');
%plot(normalized_x,groundtruth,'blue',normalized_x,output255,'red');

title('CRF estimation');
ylabel('Normalized Irradiance');
xlabel('Normalized Brightness');
legend('GT','OSPC','Location', 'best');
grid on;