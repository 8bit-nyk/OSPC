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
latent=[];
 
% image1_list = 0:3238;                   
% image2_list = 1:3239;
image1_list = 0:100 %1796;                   
image2_list = 1:101 %1797;


for pair=1:size(image1_list,2) %%%%%%%%%%%%%%%%% change

image1_num=get_num_image(image1_list(pair)) ;
image2_num=get_num_image(image2_list(pair)) ;



image1=im2double(imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\images\" + image1_num + ".jpg"));
image2=im2double(imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\images\" + image2_num + ".jpg"));

%% you did not normalizeeeeeeeeeeeeeeeeeeeeeee previously now fixed

I1=image1;
I2=image2;

%% exposures
temp_exposures = dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\times.txt' );
exposures_truth = temp_exposures(:,3);
image1_exposure = exposures_truth(image1_list(pair) + 1);                  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
image2_exposure = exposures_truth(image2_list(pair) + 1);


% exposure ratio
exposure_ratio = (image1_exposure/image2_exposure);




if exposure_ratio < 1 & exposure_ratio >0.9 %~= 1 
    
% 	if exposure_ratio > 1
%         temp=image1;
%         image1=image2;
%         image2=temp;
%         exposure_ratio=1/exposure_ratio ;
%     end
    h=size(image1,1);
    w=size(image1,2);
    image1_mid=image1(  round(h/2-10):round(h/2+10)    ,  round(w/2-10):round(w/2+10)    )
    
    true   =   [true exposure_ratio ];
    latent =   [latent  mean(image1(:))/mean(image2(:))  ];
    

end

end


figure
scatter(latent,true)


c=polyfit(latent,true,1)
estimated=polyval(c,latent);
hold on
scatter(latent,estimated)
error=abs(estimated-true);







