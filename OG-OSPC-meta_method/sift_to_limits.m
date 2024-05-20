%% Experiment one : 
%{ 
-import image
-get exposure
-get inverse crf 
-fit V 
-get corr
-remove v and e from each pair
-check if the L value is the same 
%}
clear all 
clc

%% import image 
image1_list= [ 0 ] ;
image2_list= [ 9 ] ;
pair=1;
image1_num=get_num_image(image1_list(pair)) ;
image2_num=get_num_image(image2_list(pair)) ;

image1=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\images\" + image1_num + ".jpg");
image2=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\images\" + image2_num + ".jpg");
I1=image1;
I2=image2;
%% get exposure 
temp_exposures = dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\times.txt' );
exposures_truth = temp_exposures(:,3);
image1_exposure = exposures_truth(image1_list(pair) + 1);                  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
image2_exposure = exposures_truth(image2_list(pair) + 1);
% exposure ratio
exposure_ratio = (image1_exposure/image2_exposure);
%% get crf and in crf 
crf_truth=dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\pcalib.txt' );
in=0:1:255;
%crf
crf_truth_fun=polyfit(in,crf_truth,15);
val=polyval(crf_truth_fun,in);
%inv crf
inv_truth_fun=polyfit(crf_truth,in,15);
val_inv=polyval(inv_truth_fun,crf_truth);

figure
plot(in,crf_truth)
hold on 
plot(in,val)
hold on
plot(crf_truth,val_inv)

%% fit V
vignette_truth = imread('C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\vignette.png');
vignette_truth_norm = im2double(vignette_truth);

% raduis map
radius_map_f1 = zeros(size(I1));
center_I1 = size(I1)/2;
for i = 1 :1:size(I1,1)
    for j = 1:1:size(I1, 2)
        
        radius_map_f1(i,j) = sqrt(   (i - center_I1(1))^2 + (j - center_I1(2))^2 );       
    end
end

radius_map_f1=radius_map_f1; %/ max(radius_map_f1, [], 'all'); %% why normalize?!


% function vignette truth
eps = 0.005;
radius_map_norm_sub = radius_map_f1(center_I1(1):size(image1,1), center_I1(2):size(image1,2));
vignette_truth_norm_sub = vignette_truth_norm(center_I1(1):size(image1,1), center_I1(2):size(image1,2));
figure
rad_counter = 0;
for rad = 0:radius_map_f1(1,1)/1000:radius_map_f1(1,1)
    [rad_row, rad_col] = find(radius_map_norm_sub < (rad + eps) & radius_map_norm_sub > (rad - eps), 1);
    if size(rad_row) ~= 0 & size(rad_col) ~=0
        rad_counter = rad_counter + 1;
        radius_input(1,rad_counter) = radius_map_norm_sub(rad_row, rad_col);
        vignette_output(1, rad_counter) = vignette_truth_norm_sub(rad_row, rad_col);
        plot(radius_input(1, rad_counter), vignette_output(1, rad_counter), '+');
        hold on
    end
end

vig_pol_order = 4;
vign_truth_fun = polyfit(radius_input, vignette_output, vig_pol_order);
vign_truth_eval = polyval(vign_truth_fun, radius_input);

plot(radius_input, vign_truth_eval)
hold on
scatter(0,0)

%% get corr 
N_corr=200 ;  % deleted it in the function , it is taking them all 
draw=1 ;
[X_image1,X_image2] = meta_sift_corr_fn_photo(I1,I2,N_corr,draw);
x_sub_sq1=( X_image1(1,:)- X_image2(1,:) ).^2;
y_sub_sq1=( X_image1(2,:)- X_image2(2,:) ).^2;
d1= x_sub_sq1 + y_sub_sq1 ;
dist1= (d1).^0.5;

num_corr_points=size(X_image1(1,:),2);



for corr=1:num_corr_points
   
    M_f1(1,corr)=double(I1(round(X_image1(2,corr)),round(X_image1(1,corr)))); %% check why inverted 
    M_f2(1,corr)=double(I2(round(X_image2(2,corr)),round(X_image2(1,corr))));
%%%%%%%%%%%%%%%%%%%%%% why some values > 1 
    R_f1(1,corr)= double(radius_map_f1(round(X_image1(2,corr)),round(X_image1(1,corr))));
    R_f2(1,corr)= double(radius_map_f1(round(X_image2(2,corr)),round(X_image2(1,corr))));
 
end


%% remove v and e 

I_f1 = polyval(inv_truth_fun,im2double(M_f1)); %% why im2double is not normalizing
I_f2 = polyval(inv_truth_fun,im2double(M_f2)); 

V_f1=polyval(vign_truth_fun, R_f1);
V_f2=polyval(vign_truth_fun, R_f2);

L_f1= I_f1 ./ (V_f1*image1_exposure); 
L_f2= I_f2 ./ (V_f2*image2_exposure); 

L_per=(L_f1./L_f2)*100;



%% 
%% remove  V 
% % % I_f1_nov=polyval(inv_truth_fun,im2double(I1(:))*255)./(polyval(vign_truth_fun, radius_map_f1(:))); 
% % % I_f2_nov=polyval(inv_truth_fun,im2double(I2(:))*255)./(polyval(vign_truth_fun, radius_map_f1(:))); 
% % % 
% % % M_f1_nov=polyval(crf_truth_fun,I_f1_nov)    ; 
% % % M_f2_nov=polyval(crf_truth_fun,I_f2_nov)   ;
% % % M_f1_nov(M_f1_nov>255 ) = 255;
% % % M_f2_nov(M_f2_nov>255 ) = 255;
% % % 
% % % image1_nov=reshape(M_f1_nov,size(I1));
% % % image2_nov=reshape(M_f2_nov,size(I2));
% % % imshow(uint8(image1_nov))
% % % imshow(uint8(image2_nov))

% rpeat 
%%%%%%%%%%%                       za3baraaaaaaaaaaaaaaaaaaaaaaaaaaa 



image1_nov=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\temp-rec-corr\" + image1_list(pair) + ".png");
image2_nov=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\temp-rec-corr\" + image2_list(pair)+ ".png");

%%%

% raduis map
radius_map_f1_s = zeros(size(image1_nov));
center_I1_s = size(image1_nov)/2;
for i = 1 :1:size(image1_nov,1)
    for j = 1:1:size(image1_nov, 2)
        
        radius_map_f1_s(i,j) = sqrt(   (i - center_I1_s(1))^2 + (j - center_I1_s(2))^2 );       
    end
end

radius_map_f1_s=radius_map_f1_s; %/ max(radius_map_f1, [], 'all'); %% why normalize?!

N_corr=200 ; % deleted it in the function , it is taking them all 
draw=1 ;
[X_image1_s,X_image2_s] = meta_sift_corr_fn_photo(image1_nov,image2_nov,N_corr,draw);

x_sub_sq2=( X_image1_s(1,:)- X_image2_s(1,:) ).^2;
y_sub_sq2=( X_image1_s(2,:)- X_image2_s(2,:) ).^2;
d2= x_sub_sq2 + y_sub_sq2 ;
dist2= (d2).^0.5;

% tol=20;
%  X_image1= X_image1(:,dist2<tol)
%  X_image2= X_image2(:,dist2<tol)

num_corr_points=size(X_image1_s(1,:),2);


%%%%%%%%%%%%%%% REMOVE saturateeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed 
for corr=1:num_corr_points
   
    M_f1_s(1,corr)=double(image1_nov(round(X_image1_s(2,corr)),round(X_image1_s(1,corr)))); %% check why inverted 
    M_f2_s(1,corr)=double(image2_nov(round(X_image2_s(2,corr)),round(X_image2_s(1,corr))));
%%%%%%%%%%%%%%%%%%%%%% why some values > 1 
    R_f1_s(1,corr)= double(radius_map_f1_s(round(X_image1_s(2,corr)),round(X_image1_s(1,corr))));
    R_f2_s(1,corr)= double(radius_map_f1_s(round(X_image2_s(2,corr)),round(X_image2_s(1,corr))));
 
end


%% remove v and e 

I_f1_s = polyval(inv_truth_fun,im2double(M_f1_s)); 
I_f2_s = polyval(inv_truth_fun,im2double(M_f2_s)); 

V_f1_s=polyval(vign_truth_fun, R_f1_s); % does this change by size ?
V_f2_s=polyval(vign_truth_fun, R_f2_s);

L_f1_s= I_f1_s ./ (V_f1_s*image1_exposure); 
L_f2_s= I_f2_s ./ (V_f2_s*image2_exposure); 

L_per_S=(L_f1_s./L_f2_s)*100;

%% 
%%  


[N_start_img,N_last,N_corr,draw,draw_final] = deal(image1_list(pair),image2_list(pair),2000,0,1);
[X_image1_ss,X_image2_ss]=best_tracker(N_start_img,N_last,N_corr,draw,draw_final); 



num_corr_points=size(X_image1_ss(1,:),2);

for corr=1:num_corr_points
   
    M_f1_ss(1,corr)=I1(round(X_image1_ss(2,corr)),round(X_image1_ss(1,corr))); %% check why inverted 
    M_f2_ss(1,corr)=I2(round(X_image2_ss(2,corr)),round(X_image2_ss(1,corr)));
%%%%%%%%%%%%%%%%%%%%%% why some values > 1 
    R_f1_ss(1,corr)= radius_map_f1(round(X_image1_ss(2,corr)),round(X_image1_ss(1,corr))); %% raduis map similar to the first 
    R_f2_ss(1,corr)= radius_map_f1(round(X_image2_ss(2,corr)),round(X_image2_ss(1,corr)));
    
 
end


%% filling

corr_pts=num_corr_points;


%% remove v and e 

I_f1_ss = polyval(inv_truth_fun,im2double(M_f1_ss)); 
I_f2_ss = polyval(inv_truth_fun,im2double(M_f2_ss)); 

V_f1_ss=polyval(vign_truth_fun, R_f1_ss);  %% this changes by size right???????????
V_f2_ss=polyval(vign_truth_fun, R_f2_ss);

L_f1_ss= I_f1_ss ./ (V_f1_ss*image1_exposure); 
L_f2_ss= I_f2_ss ./ (V_f2_ss*image2_exposure); 

L_per_ss=(L_f1_ss./L_f2_ss)*100;




























figure('Name',"L_per")
histogram(L_per)
figure('Name',"L_per_s")
histogram(L_per_S)
figure('Name',"L_per_ss")
histogram(L_per_ss)






