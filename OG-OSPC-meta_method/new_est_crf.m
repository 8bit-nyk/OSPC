%% Clear and flush caches
clc 
clear 

%%
% Specify the datset you are using and initialize adjustable parameters
%datasetFormat = "tum";
datasetFormat = "vrl";
window = 100;
step = 20;
if datasetFormat == "tum"
    image1_list = 0:0+window;
elseif datasetFormat == "vrl"
    image1_list = 1:1+window; 
end
image2_list=  step:step+window;
t=0.1; %001;% threshold on displacement

%Initialize variables
A_all=[];
l_all=[];
A=[];
l=[];
R1_list=[];
length_matchpts=[1]; %since first time start at 1
truth=[];
M1_list=[];
M2_list=[];
R1_list=[];
R2_list=[];
counter=1; 
r_all=[];
pair_all=[];

%%
% Figure 1: Ground Truth CRF and Polynomial Fitting

% Read and preprocess ground truth CRF data
%crf_truth = readmatrix('/home/aub/datasets/sequence_30_og/pcalib.txt');

crf_truth = readmatrix('/home/aub/datasets/crf_dataset_20230822/crf_response.txt');
crf_truth_norm = crf_truth / 255;

% Create input values for the polynomial fitting
input_values = 0:1/767:1;

% Perform polynomial fitting to approximate the CRF
degree_crf_fitting = 10;
coeff_crf_fitting = polyfit(input_values, crf_truth_norm, degree_crf_fitting);
fitted_crf = polyval(coeff_crf_fitting, input_values);

% Perform polynomial fitting to approximate the inverse of the CRF
degree_inverse_fitting = 15;
coeff_inverse_fitting = polyfit(crf_truth_norm, input_values, degree_inverse_fitting);
fitted_inverse_crf = polyval(coeff_inverse_fitting, crf_truth_norm);

% Plot the ground truth CRF and the fitted polynomials
figure('Name', 'Ground Truth CRF and Fitted Polynomials');
plot(input_values, crf_truth_norm, 'b', 'LineWidth', 2);
hold on;
plot(input_values, fitted_crf, 'r--', 'LineWidth', 2);
plot(crf_truth_norm, fitted_inverse_crf, 'g:', 'LineWidth', 2);

% Set plot labels and legend
xlabel('Normalized Input Values');
ylabel('Normalized CRF Values');
title('Ground Truth CRF and Fitted Polynomials');
legend('Ground Truth CRF', 'Fitted CRF', 'Fitted Inverse CRF', 'Location', 'best');
grid on;
hold off;


%% get corr points
for pair=1:size(image1_list,2) 
%input images for selected dataset
    if datasetFormat == "tum"
        image1_num=get_num_image(image1_list(pair)) ;
        image2_num=get_num_image(image2_list(pair)) ;
        image1=im2double(imread("/home/aub/datasets/sequence_30_og/images/" + image1_num + ".jpg"));
        image2=im2double(imread("/home/aub/datasets/sequence_30_og/images/" + image2_num + ".jpg"));
        I1=image1;
        I2=image2;
        temp_exposures = readmatrix('/home/aub/datasets/sequence_30_og/times.txt' );
    elseif datasetFormat == "vrl"
        image1_num=image1_list(pair) ;
        image2_num=image2_list(pair) ;
       % 
        %image1=im2double(imread("/home/aub/datasets/Pcalib/d435-dataset-mv2/Frame-"+ image1_num + ".jpg"));
        %image2=im2double(imread("/home/aub/datasets/Pcalib/d435-dataset-mv2/Frame-"+ image2_num + ".jpg"));
        image1=im2double(imread("/home/aub/datasets/crf_dataset_20230822/frame_"+ image1_num + ".png"));
        image2=im2double(imread("/home/aub/datasets/crf_dataset_20230822/frame_"+ image2_num + ".png"));
        I1=rgb2gray(image1);
        I2=rgb2gray(image2);
        %temp_exposures = readmatrix('/home/aub/datasets/Pcalib/d435_mv1_exposure_values.csv' );
        temp_exposures = readmatrix('/home/aub/datasets/crf_dataset_20230822/frame_data.txt' );

    end
%get exposre values from third column and calculate exposure ratio
    %exposures_truth = temp_exposures(:,3);
    exposures_truth = temp_exposures(:,2);
    image1_exposure = exposures_truth(image1_list(pair) + 1);        
    image2_exposure = exposures_truth(image2_list(pair) + 1);
    exposure_ratio = (image1_exposure/image2_exposure);
    r=exposure_ratio;

%get radius for  image and calculate a radius map
    radius_map_f1 = zeros(size(I1));
    center_I1 = size(I1)/2;
    for i = 1:1:size(I1,1)
        for j = 1:1:size(I1, 2)
            
            radius_map_f1(i,j) = sqrt(   (i - center_I1(1))^2 + (j - center_I1(2))^2 );       
        end
    end
    
    radius_map_f1=radius_map_f1 /  max(radius_map_f1, [], 'all'); %why normalize?!
    
  %accept only high diffecrence in exposure ratio of the exposure pair
    if   r<0.93 || r>1.05
        truth=[truth r];
    
        %% feature extraction
        pair_all=[pair_all image1_num];
        X_image1=[];
        X_image2=[];
        %N_corr=200;         % deleted it in the function , it is taking them all 
        draw=0;
        feature_method="sift" % feature method experiment 
  
        if feature_method=="sift"
            tic
            [X_image1,X_image2]=meta_sift_corr_fn_photo(I1,I2,draw);
            toc
        else
         %TRIAL using other feature detectors
            tic
            % sift 
             points1_ = detectSIFTFeatures(I1);
             points2_ = detectSIFTFeatures(I2);
            %harris
             %points1_ = detectHarrisFeatures(I1);
             %points2_ = detectHarrisFeatures(I2);
            %surf
             %points1_ = detectSURFFeatures(I1);
             %points2_ = detectSURFFeatures(I2);
            %orb
             %points1_ =detectORBFeatures(I1);
             %points2_ =detectORBFeatures(I2);
            %Brisk
             %points1_ = detectBRISKFeatures(I1);
             %points2_ = detectBRISKFeatures(I2);
            % Fast 
             %points1_ = detectFASTFeatures(I1);
             %points2_ = detectFASTFeatures(I2);
            %Strongest 
             %N=3000;
             %points1 = selectStrongest(points1_,N);
             %points2 = selectStrongest(points2_,N);

        %Extract the features
            [f1,vpts1] = extractFeatures(I1,points1_);
            [f2,vpts2] = extractFeatures(I2,points2_);
            %Retrieve the locations of matched points
        
            indexPairs = matchFeatures(f1,f2) ;
        
            matchedPoints1 = vpts1(indexPairs(:,1));
            matchedPoints2 = vpts2(indexPairs(:,2));
            matchedPointsf1_r=round(matchedPoints1.Location);
            matchedPointsf2_r=round(matchedPoints2.Location);
            X_image1=matchedPointsf1_r';
            X_image2=matchedPointsf2_r';
            toc
        end
        %%end of feature extraction

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
    
        %draw_points(image1,image2, X_image1_nod , X_image2_nod,num_blocks_y,num_blocks_x )   
    
    
    
        num_corr_points=size(X_image1_nod(1,:),2);
    
        for corr=1:num_corr_points   %% round or round
    
             theta( 1,corr )=rad2deg(atan2(round(X_image2_nod(2,corr)-X_image1_nod(2,corr)),round(X_image2_nod(1,corr)-X_image1_nod(1,corr))));
             region_f1_points(1,corr)=region_image( round(X_image1_nod(2,corr)) , round(X_image1_nod(1,corr)) );
    
        end 
        %%
        [draw,ratio_std]=deal(0,0.1);
        [ X_image1_clean , X_image2_clean ] = direction_filter( X_image1_nod,X_image2_nod,num_regions,region_f1_points,theta,I1,I2,draw,ratio_std );
    
        %get pads
    %     X_image1_clean=neighbours(X_image1_clean0);
    %     X_image2_clean=neighbours(X_image2_clean0);
    %     X_image1_clean=X_image1_clean0 ; %%%%%%%%%%%%%%%% change this 
    %     X_image2_clean=X_image2_clean0 ;
    
    
        %draw_points(image1,image2,X_image1_clean , X_image2_clean,num_blocks_y,num_blocks_x )   ;
    
    
        % _points(image1,image2, X_image1_clean,X_image2_clean,num_blocks_y,num_blocks_x )   
    
        num_corr_points_clean=size(X_image1_clean,2);
    
        for corr=1:num_corr_points_clean   %% round or round
    
            M_f1(1,corr)=I1(round(X_image1_clean(2,corr)),round(X_image1_clean(1,corr))); %% check why inverted 
            M_f2(1,corr)=I2(round(X_image2_clean(2,corr)),round(X_image2_clean(1,corr)));
            %%%%%%%%%%%%%%%%%%%%%% why some values > 1 
            R_f1(1,corr)= radius_map_f1(round(X_image1_clean(2,corr)),round(X_image1_clean(1,corr)));
            R_f2(1,corr)= radius_map_f1(round(X_image2_clean(2,corr)),round(X_image2_clean(1,corr)));
    
        end
    
        %draw_points(image1,image2, X_image1_clean( :,abs(R_f1-R_f2)<t ),X_image2_clean( :,abs(R_f1-R_f2)<t ),num_blocks_y,num_blocks_x )   
    
    
        %% filling
    
        corr_pts=num_corr_points_clean; % changing this drastically changes the result thus outliers play a major role , so remove them
        [Y,E] = discretize(M_f1,0:0.5:1);
        
        
        r_all=[r_all r];
        for i=1:corr_pts
    
            [ M1 , M2,R1,R2]=deal( double(M_f1(i)), double(M_f2(i)),double(R_f1(i)),double(R_f2(i)) );
    
    
            
            if  M1 == 1 || M2 == 1 || M1 == 0 || M2 == 0 || M1==M2 || (M1<M2 & r>1) || (M1>M2 & r<1) %|| (R1 > 0.2 & R2>0.2) %!!!!!depends  %|| dist(i)>115 %|| mag1(i)>40 || mag2(i)>40 ||  abs(M1-M2)>90 % abs(M1-M2)<10   %|| M1>M2 %|| dist(i)>50
                continue 
            end
    
            if abs(R2-R1)>t %related to V
                continue 
            end
            M1_list=[M1_list M1];
            M2_list=[M2_list M2];
           
          
            row=[ (1 - r) (M1 - r*M2)   (M1^2- r*M2^2)];
    
            A(counter,:)=row;
    
            k=1;
            counter=counter + 1;
    
        end
 end
 length_matchpts=[ length_matchpts size(M1_list,2) ];
end 
%loop end 



%% solve using convex optomization method 
A_all=A;
n=3;
cvx_begin

    variable x(n)
    %b=zeros(corr_pts,1)
    minimize( norm( A*x ,2) );
    %minimize( (A_all*x)'*(A_all*x) );
    
    subject to 
       sum(x)==1;
       x(1)==0;
       
cvx_end

%% solve using eigen method 
[V,D]=eig(A_all'*A_all);

% [U,S,V]=svd(A_all);  
% x=V(:,1);

%% solve using lagrange 
d1=[1;0];
E=[1 1 1;
    1 0 0];
Z=[0 0;
    0 0];
M=[2*A_all'*A_all   E';
    E               Z];
b1=[0;0;0;d1];

sol_fast=(M'*M)^-1*M'*b1;
   

%% Plotting Camera Response Function (CRF) Results

% Define the input range for the CRF plot
input_range = 0:1/255:1;

% Calculate the sum of coefficients and find the ratio for normalization
sum_coefficients = sum(x);
normalization_ratio = 1 / sum_coefficients;

% Calculate the CRF output using the fitted polynomial
crf_output = (x(1) + x(2) * input_range + x(3) * input_range.^2) * 255;

% Create a new figure for the plot
figure('Name', 'Camera Response Function (CRF) Results');

% Plot the fitted CRF
plot(input_range, crf_output * normalization_ratio, 'r', 'LineWidth', 2);
hold on;
%Plot Inverse fitted CRF

% Plot the ground truth CRF
plot(input_range, crf_truth, 'g', 'LineWidth', 2);

% Set plot labels and legend
xlabel('Input Intensity');
ylabel('Irradiance');
title('Fitted CRF vs. Ground Truth CRF');
legend('Fitted CRF', 'Inverse CRF', 'Ground Truth CRF', 'Location', 'best');

% If you have a fast method to compare, plot it as well
if exist('sol_fast', 'var')
    crf_output_fast = (sol_fast(1) + sol_fast(2) * input_range + sol_fast(3) * input_range.^2) * 255;
    hold on;
    plot(input_range, crf_output_fast, 'b--', 'LineWidth', 2);
    legend('Fitted CRF', 'Ground Truth CRF', 'Fast Method', 'Location', 'best');
end

% Customize the plot appearance
grid on;
box on;
set(gca, 'FontSize', 12);
set(gcf, 'Color', 'w');

%input=0:1/255:1; 
%input=0:1:255;

%final=sum(x);maps
%ratio=1/final;
%output=  (x(1) + x(2)*input +  x(3)*input.^2)*255; %+ x(4)*input.^3 +  x(5)* input.^4% + x(6)*input.^5 +  x(7)* input.^6 + x(8)*input.^7 +  x(9)* input.^8 +  x(10)* input.^9 +  x(11)* input.^10;
%figure('Name','Results')
%plot(input,output*ratio,'r','LineWidth',2)
%hold on
%plot(input,crf_truth,'g','LineWidth',2)
%xlabel('Intensity')
%ylabel('Irradiance')
%hold on 
%crf_output_fast=  (sol_fast(1) + sol_fast(2)*input +  sol_fast(3)*input.^2)*255
%plot(input,out_fast,'b','LineWidth',2)



%% calculate RMSE errors 
crf_255=crf_output_fast*255 
error_feature= sqrt(sum( ( crf_truth/255 - ( crf_output_fast )/255 ).^2)/size(crf_truth ,2))




%% now try to estimate exposure back
% for every keyframe 
    % loop for every matched point 
    %estimate k 
%{    
estimated_list_mean=[] 
estimated_list_median=[]
counter=1
% you need to fix here
 for i=2:size(length_matchpts,2)  %since you aded 1 %loop over keyframes
     for j=length_matchpts(i-1):length_matchpts(i)
         
        [ M1 , M2]=deal( M1_list(j), M2_list(j) );
        if M1>0.6 || M2>0.6
            continue
        else 
            %p is the truth
            invf1= polyval(p,M1) %+ M1^3*X(4) + M1^4*X(5) + M1^5*X(6) + M1^6*X(7) + M1^7*X(8) + M1^8*X(9) + M1^9*X(10); 
            invf2= polyval(p,M2)
            %invf1= sol_fast(1) + sol_fast(2)*M1 +  sol_fast(3)*M1^2
            %invf2= sol_fast(1) + sol_fast(2)*M2 +  sol_fast(3)*M2^2
            %Vf1=polyval(vign_truth_fun, R1);
            %Vf2=polyval(vign_truth_fun, R2);

            exp_est_list(counter)=(invf1/invf2) %/(Vf1/Vf2);
            counter=counter+1
        end    
     end
     
     estimated_list_mean = [ estimated_list_mean  mean(exp_est_list) ]
     estimated_list_median =  [   estimated_list_median        median(exp_est_list)]
     exp_est_list=[];
     counter=1;
 
 end
 
 
figure
plot(truth,'LineWidth',2)
hold on

%plot(estimated_list_mean,'LineWidth',1)
%hold on
plot(estimated_list_median,'LineWidth',2)
 %} 
     



