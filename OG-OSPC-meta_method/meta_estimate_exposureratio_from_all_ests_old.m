clc 
clear 

%% read crf
crf_truth=dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_50\pcalib.txt' );

figure(1)
plot(crf_truth)


%% get corresponding points %0.95 0.9376 0.9376
num_image1=67; %105 1  35 185 1585
num_image2=66; %106 36 36 186 1586
%% best proove that it is caused by outliers  
% worked        1-20 1-21   (there is no outliers)
% did not work  1-19   1-21 (there is outliers) 
%%
I1 = im2double(imread('C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_50\images\000' + string(num_image1) + ".jpg" )); %0 102
I2 = im2double(imread('C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_50\images\000' + string(num_image2) + ".jpg" )); %15 128
%I1 = rgb2gray (im2double(imread('C:\Users\User\Desktop\102_m.jpg'))); 
%I2 = rgb2gray (im2double(imread('C:\Users\User\Desktop\108_m.jpg'))); 
figure
imshow(I1)
figure
imshow(I2)

%% exposures
temp_exposures = dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_50\times.txt' );
exposures_truth = temp_exposures(:,3);
image1_exposure = exposures_truth(num_image1 + 1);
image2_exposure = exposures_truth(num_image2 + 1);

% exposure ratio
exposure_ratio = (image1_exposure/image2_exposure);

%get radius for  image
% calculate a radius map
radius_map_f1 = zeros(size(I1));
center_I1 = size(I1)/2;
for i = 1 :1:size(I1,1)
    for j = 1:1:size(I1, 2)
        
        radius_map_f1(i,j) = sqrt(   (i - center_I1(1))^2 + (j - center_I1(2))^2 );       
    end
end

radius_map_f1=radius_map_f1/ max(radius_map_f1, [], 'all');


%% feature extraction 

N_corr=500; % deleted it in the function , it is taking them all 
draw=1;
[X_image1,X_image2]=meta_sift_corr_fn_photo(I1,I2,N_corr,draw);
%% calculate distance
x_sub_sq=( X_image1(1,:)- X_image2(1,:) ).^2;
y_sub_sq=( X_image1(2,:)- X_image2(2,:) ).^2;
d= x_sub_sq + y_sub_sq ;
dist= (d).^0.5;

num_corr_points=size(X_image1(1,:),2);

for corr=1:num_corr_points
   
    M_f1(1,corr)=I1(round(X_image1(2,corr)),round(X_image1(1,corr))); %% check why inverted 
    M_f2(1,corr)=I2(round(X_image2(2,corr)),round(X_image2(1,corr)));
%%%%%%%%%%%%%%%%%%%%%% why some values > 1 
    R_f1(1,corr)= radius_map_f1(round(X_image1(2,corr)),round(X_image1(1,corr)));
    R_f2(1,corr)= radius_map_f1(round(X_image2(2,corr)),round(X_image2(1,corr)));

end


%% filling

corr_pts=num_corr_points; % changing this drastically changes the result thus outliers play a major role , so remove them

%num_corr_points
for i=1:corr_pts
    
    [ M1 , M2,R1,R2]=deal( M_f1(i), M_f2(i),double(R_f1(i)),double(R_f2(i)) );

    [c0,c1,c2]=deal(0,-0.093920939562920,1.093920939562920);
    [v1,v2,v3]=deal(-1.885783024583895,3.243195251823905,-2.111217842250675); 

    invf1= c0 + M1*c1 + M1^2*c2; %+ M1^3*X(4) + M1^4*X(5) + M1^5*X(6) + M1^6*X(7) + M1^7*X(8) + M1^8*X(9) + M1^9*X(10); 
    invf2= c0 + M2*c1 + M2^2*c2;
    Vf1=1 + v1*R1^2 + v2*R1^4 + v3*R1^6;
    Vf2=1 + v1*R2^2 + v2*R2^4 + v3*R2^6;

    exp_est_list(i)=(invf1/invf2)/(Vf1/Vf2);

end

mean(exp_est_list)
median(exp_est_list)






