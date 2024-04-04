%{ 

- read images  
- exposures 
- if r ~=1
    - if r >1 
        flip 
      else 
      
    - mean of each image 
    - latent exposure 
    - 

%} 

clc 
clear 
true=[];
intensity_energy=[];
reflectance_energy=[];
scene_radiance=[];
 
% image1_list = 0:3238;                   
% image2_list = 1:3239;
image1_list = 0:100;                   



for pair=1:size(image1_list,2) %%%%%%%%%%%%%%%%% change

    image1_num=get_num_image(image1_list(pair)) ;




    image1=im2double(imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\images\" + image1_num + ".jpg"));

    %% you did not normalizeeeeeeeeeeeeeeeeeeeeeee previously now fixed

    I1=image1;


    %% exposures
    temp_exposures = dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\times.txt' );
    exposures_truth = temp_exposures(:,3);
    image1_exposure = exposures_truth(image1_list(pair) + 1);     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55



    reflectance=image1/mean(image1(:));


    true   = [true image1_exposure ];
    intensity_energy =  [intensity_energy    sum(image1(:))];
    reflectance_energy=[ reflectance_energy  sum(reflectance(:))];
    
    scene_radiance=[ scene_radiance       mean(image1(:))]


end


figure('Name',"true_vs_intensity_energy")
scatter(true,intensity_energy)

figure('Name',"true_vs_reflectance_energy")
scatter(true,reflectance_energy)

figure('Name',"true_vs_intensity/reflectance_energy")
scatter(true,intensity_energy./reflectance_energy)

figure('Name',"true_vs_scene_Radiance")
scatter(true,scene_radiance)





