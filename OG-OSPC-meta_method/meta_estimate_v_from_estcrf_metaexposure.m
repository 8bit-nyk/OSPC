
%% Clear and flush caches
clc 
clear 

%%
% Specify the datset you are using and initialize adjustable parameters
%datasetFormat = "tum";
datasetFormat = "vrl";
%window = 200;
%step = 60;
window = 7;
step = 1;
t=0.3;
if datasetFormat == "tum"
    image1_list = 0:0+window ;
elseif datasetFormat == "vrl"
    %image1_list = 1:1+window ;
    image1_list = 0:0+window ;


end
image2_list=  step:step+window;

%Initialize variables
A_all=[];
l_all=[];
A=[];
l=[];
R1_list=[];
counter=1;
M1_list=[];
M2_list=[];
R1_list=[];
R2_list=[];


%% Read and display Ground trugh CRF and Vignette
crf_truth=readmatrix('/home/aub/datasets/sequence_30_og/pcalib.txt' ); 
vignette_truth = imread('/home/aub/datasets/sequence_30_og/vignette.png');
vignette_truth_norm = im2double(vignette_truth);

% Generate input values in the range [0, 1] with fine granularity (1/255)
input_range = 0:1/255:1;

% Fit a polynomial of degree 10 to approximate the ground truth Camera Response Function (CRF)
coefficients_crf_fitting = polyfit(input_range, crf_truth/255, 10);
p = coefficients_crf_fitting;
% Calculate the CRF values using the fitted polynomial
fitted_crf = polyval(coefficients_crf_fitting, input_range);

% Fit a polynomial of degree 15 to approximate the inverse of the ground truth CRF
coefficients_inverse_fitting = polyfit(crf_truth/255, input_range, 15);

% Calculate the inverse CRF values using the fitted polynomial
fitted_inverse_crf = polyval(coefficients_inverse_fitting, crf_truth/255);

% Plot the ground truth CRF, fitted CRF, and inverse fitted CRF
figure('Name', 'Camera Response Function (CRF) Approximation');
plot(input_range, crf_truth/255, 'b', 'LineWidth', 2);
hold on;
plot(input_range, fitted_crf, 'r--', 'LineWidth', 2);
plot(crf_truth/255, fitted_inverse_crf, 'g:', 'LineWidth', 2);

% Set plot labels and legend
xlabel('Input Intensity (Normalized)');
ylabel('Irradiance (Normalized)');
title('Ground Truth CRF and Polynomial Approximations');
legend('Ground Truth CRF', 'Fitted CRF', 'Fitted Inverse CRF', 'Location', 'best');

% Customize the plot appearance
grid on;
box on;
set(gca, 'FontSize', 12);
set(gcf, 'Color', 'w');


%% get corresponding points %0.95 0.9376 0.9376



for pair=1:size(image1_list,2) %%%%%%%%%%%%%%%%% change
%input images for selected dataset
    if datasetFormat == "tum"
        image1_num=get_num_image(image1_list(pair)) ;
        image2_num=get_num_image(image2_list(pair)) ;
        image1=im2double(imread("/home/aub/datasets/sequence_30_og/images/" + image1_num + ".jpg"));
        image2=im2double(imread("/home/aub/datasets/sequence_30_og/images/" + image2_num + ".jpg"));
        I1=image1;
        I2=image2;
        temp_exposures = readmatrix( '/home/aub/datasets/sequence_30_og/times.txt');
    elseif datasetFormat == "vrl"
        image1_num=image1_list(pair) ;
        image2_num=image2_list(pair) ;
        %/home/aub/datasets/Pcalib/V-unit-testing
        image1=im2double(imread("/home/aub/datasets/Pcalib/V-unit-testing/image_" + image1_num + ".jpg"));
        image2=im2double(imread("/home/aub/datasets/Pcalib/V-unit-testing/image_" + image2_num + ".jpg"));
        I1=im2gray(image1);
        I2=im2gray(image2);
        temp_exposures = readmatrix('/home/aub/datasets/Pcalib/V-unit-testing/exposure_values.txt' );
    end
%get exposre values from third column and calculate exposure ratio
    %exposures_truth = temp_exposures(:,3);
    exposures_truth = temp_exposures(:,2);

    image1_exposure = exposures_truth(image1_list(pair) + 1);        
    image2_exposure = exposures_truth(image2_list(pair) + 1);
    exposure_ratio = (image1_exposure/image2_exposure);


    %% calculate a radius map
    radius_map_f1 = zeros(size(I1));
    center_I1 = size(I1)/2;
    for i = 1 :1:size(I1,1)
        for j = 1:1:size(I1, 2)
            
            radius_map_f1(i,j) = sqrt(   (i - center_I1(1))^2 + (j - center_I1(2))^2 );       
        end
    end
    
    radius_map_f1=radius_map_f1 /  max(radius_map_f1, [], 'all'); %% why normalize?!


    %% feature extraction 
    X_image1=[];
    X_image2=[];
    draw=0;
    [X_image1,X_image2]=meta_sift_corr_fn_photo(I1,I2,draw);
    
    %% calculate distance
    M_f1=[];
    M_f2=[];
    R_f1=[];
    R_f2=[];
    region_f1_points=[];
    theta=[];
    
    [num_blocks_x,num_blocks_y,num_rows,num_cols]=deal(15,15,size(I1,1),size(I1,2));
    num_regions=num_blocks_x*num_blocks_y;
    region_image=create_region_img(num_blocks_x,num_blocks_y,num_rows,num_cols);

    %% remove huge distance since it effects the mean
    x_sub_sq=( X_image1(1,:)- X_image2(1,:) ).^2;
    y_sub_sq=( X_image1(2,:)- X_image2(2,:) ).^2;
    d= x_sub_sq + y_sub_sq ;
    dist= (d).^0.5;
    index_dist=dist < mean(dist) + 2*std(dist)  ;   %% it was 2 try it before cleaning as well
    X_image1_nod=X_image1(:,index_dist);
    X_image2_nod=X_image2(:,index_dist);
    
    % draw_points(image1,image2, X_image1_nod , X_image2_nod,num_blocks_y,num_blocks_x )   

    num_corr_points=size(X_image1_nod(1,:),2);
    
    for corr=1:num_corr_points   %% round or round
       
         theta( 1,corr )=rad2deg(atan2(round(X_image2_nod(2,corr)-X_image1_nod(2,corr)),round(X_image2_nod(1,corr)-X_image1_nod(1,corr))));
         region_f1_points(1,corr)=region_image( round(X_image1_nod(2,corr)) , round(X_image1_nod(1,corr)) );
        
    end 
    %% filter
    [draw,ratio_std]=deal(0,0.1);
    [ X_image1_clean, X_image2_clean] = direction_filter( X_image1_nod,X_image2_nod,num_regions,region_f1_points,theta,I1,I2,draw,ratio_std );
   
    num_corr_points_clean=size(X_image1_clean,2);
    
    for corr=1:num_corr_points_clean   %% round or round
       
        M_f1(1,corr)=I1(round(X_image1_clean(2,corr)),round(X_image1_clean(1,corr))); %% check why inverted 
        M_f2(1,corr)=I2(round(X_image2_clean(2,corr)),round(X_image2_clean(1,corr)));
       
        R_f1(1,corr)= radius_map_f1(round(X_image1_clean(2,corr)),round(X_image1_clean(1,corr)));
        R_f2(1,corr)= radius_map_f1(round(X_image2_clean(2,corr)),round(X_image2_clean(1,corr)));
    
    end
    % 
    %draw_points(image1,image2, X_image1_clean( :,abs(R_f1-R_f2)>t ),X_image2_clean( :,abs(R_f1-R_f2)>t ),num_blocks_y,num_blocks_x )   
    
   
    %% filling
    
    corr_pts=num_corr_points_clean; % changing this drastically changes the result thus outliers play a major role , so remove them
    [Y,E] = discretize(R_f1,0:0.1:1);
    
    for i=1:corr_pts
            
        [ M1 , M2,R1,R2]=deal( double(M_f1(i)), double(M_f2(i)),double(R_f1(i)),double(R_f2(i)) );
        
        bin_number=Y(i);
        length_bin=sum(Y==bin_number);
        
        invf1=polyval(p,M1)    ;  
        invf2=polyval(p,M2)    ; 
        
        
        if M1==0 || M2 ==0 || M1==1 || M2 == 1 || M1==M2   
        continue 
        end
        
        if abs(R2-R1)<t
            continue 
        end
        
        
        %R1_list=[R1_list R1];
        %R2_list=[R2_list R2];
        % 
        % M1_list=[M1_list M1];
        % M2_list=[M2_list M2];
        
        
        J=(invf1*image2_exposure)/(invf2*image1_exposure); 
        a = R1^2 - J * R2^2;
        b = R1^4 - J * R2^4;
        c = R1^6 - J * R2^6;
        
        A(counter,:)=[ a b c];
        
        l(counter,:)=  ( J - 1 ) ;
        counter=counter + 1;
    
    end
    
    % A_all=[ A_all ; A ]; 
    % l_all=[ l_all ; l ];
end 
A_all=A;
l_all=l;

n=3;
cvx_begin
    variable x(n)
  
    minimize(norm( A_all*x - l_all ,1) )
    %minimize(huber(A_all*x,l_all))

cvx_end

%% M estimater
V_m = robustfit(A_all,l_all)

%% pesudo method 
V = inv(A_all'*A_all)*A_all'*l_all;

%% plot vignette
vignette_image = zeros(size(I1));
for i = 1:1:size(vignette_image,1)
    for j = 1:1:size(vignette_image, 2) 

       temp=1 + x(1)*radius_map_f1(i,j)^2 + x(2)*radius_map_f1(i,j)^4 + x(3)*radius_map_f1(i,j)^6;
       
       if temp  >= 0 
       vignette_image(i,j) = temp;   
        end
    end
end 

figure

imshow(vignette_image)
figure
% in= 0:0.1:1 ;
in= 0:radius_map_f1(1,1)/1000:radius_map_f1(1,1) ;
plot( in, 1 + V(1)*in.^2 +  V(2)*in.^4 + V(3)*in.^6 );
figure
plot( in, 1 + x(1)*in.^2 +  x(2)*in.^4 + x(3)*in.^6 );
hold on
figure('Name','vignette truth vs estimate')
% function vignette truth
eps = 0.005;
radius_map_norm_sub = radius_map_f1(center_I1(1):size(image1,1), center_I1(2):size(image1,2));
vignette_truth_norm_sub = vignette_truth_norm(center_I1(1):size(image1,1), center_I1(2):size(image1,2));

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
plot(radius_input, vign_truth_eval,'g')
hold on
scatter(0,0)



%% analysis 
%% 
%% remove v and e 


vig_pol_order = 4;
vign_truth_fun = polyfit(radius_input, vignette_output, vig_pol_order);
vign_truth_eval = polyval(vign_truth_fun, radius_input);

figure('Name',"convex_norm1")
plot( in, 1 + x(1)*in.^2 +  x(2)*in.^4 + x(3)*in.^6,'r','LineWidth',2 );
hold on
plot(radius_input, vign_truth_eval,'g','LineWidth',2)
xlabel('Radius')
ylabel('Vignette factor')
axis([0 1 0 1])


figure('Name',"pesudo")
plot( in, 1 + V(1)*in.^2 +  V(2)*in.^4 + V(3)*in.^6,'r','LineWidth',2 );
hold on
plot(radius_input, vign_truth_eval,'g','LineWidth',2)
xlabel('Radius')
ylabel('Vignette factor')
axis([0 1 0 1])


tune_const = [3 4.685 6];

for i = 1:length(tune_const)
    Vb1 = robustfit(A_all,l_all,'bisquare',tune_const(i),'off');
    VB_mat1(i,:) = Vb1;
end


figure('Name',"robust")
plot( in, 1 + VB_mat1(2,1)*in.^2 +  VB_mat1(2,2)*in.^4 + VB_mat1(2,3)*in.^6,'r','LineWidth',2 );
hold on
plot(radius_input, vign_truth_eval,'g','LineWidth',2)
xlabel('Radius')
ylabel('Vignette factor')
axis([0 1 0 1])

% calculate  RMSE 
% using L2 
error_L2= sqrt(sum((vign_truth_eval - (1 + V(1)*in.^2 +  V(2)*in.^4 + V(3)*in.^6) ).^2)/size(vign_truth_eval,2))
% using L1 
error_L1= sqrt(sum((vign_truth_eval - (1 + x(1)*in.^2 +  x(2)*in.^4 + x(3)*in.^6) ).^2)/size(vign_truth_eval,2))
% using m estimator
error_m= sqrt(sum((vign_truth_eval - (1 + VB_mat1(2,1)*in.^2 +  VB_mat1(2,2)*in.^4 + VB_mat1(2,3)*in.^6) ).^2)/size(vign_truth_eval,2))


