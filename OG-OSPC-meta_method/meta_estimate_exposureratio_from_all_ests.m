%% get corr pts 

clc 
clear all

%% read crf
crf_truth=dlmread('/home/aub/datasets/sequence_30_og/pcalib.txt' );

% in=0:1:255;
% p=polyfit(in,crf_truth,10);

in=0:1/255:1;
p=polyfit(in,crf_truth/255,10);

val=polyval(p,in);
inv_truth_fun=polyfit(crf_truth/255,in,15);
val_inv=polyval(inv_truth_fun,crf_truth/255);


figure
plot(in,crf_truth/255)
hold on 
plot(in,val)
hold on
plot(crf_truth/255,val_inv)

vignette_truth = imread('/home/aub/datasets/sequence_30_og/vignette.png');

vignette_truth_norm = im2double(vignette_truth);


%% get corresponding points %0.95 0.9376 0.9376

A_all=[];
l_all=[];
A=[];
l=[];
R1_list=[];


image1_list = 1:200; %35 %105; %66 %:1000 ; %[ 20  21  21 ]
image2_list=  2:201; %36 %106; %67 %:1001 ;  %[ 22  22  23 ]   
t=0.1 %0001;


M1_list=[];
M2_list=[];
R1_list=[];
R2_list=[];
counter=1; 
r_all=[];
pair_all=[];

estimated_list_mean = [ ] ;
estimated_list_median =  [ ] ;
truth=[] ;

%get radius for  image
% calculate a radius map
image1_num=get_num_image(image1_list(1)) ;
I1=im2double(imread("/home/aub/datasets/sequence_30_og/images/" + image1_num + ".jpg"));

radius_map_f1 = zeros(size(I1));
center_I1 = size(I1)/2;
for i = 1 :1:size(I1,1)
    for j = 1:1:size(I1, 2)

        radius_map_f1(i,j) = sqrt(   (i - center_I1(1))^2 + (j - center_I1(2))^2 );       
    end
end

radius_map_f1=radius_map_f1 /  max(radius_map_f1, [], 'all'); %% why normalize?!



eps = 0.005;
radius_map_norm_sub = radius_map_f1(center_I1(1):size(I1,1), center_I1(2):size(I1,2));
vignette_truth_norm_sub = vignette_truth_norm(center_I1(1):size(I1,1), center_I1(2):size(I1,2));

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



for pair=1:size(image1_list,2) %%%%%%%%%%%%%%%%% change

    image1_num=get_num_image(image1_list(pair)) ;
    image2_num=get_num_image(image2_list(pair)) ;

    % image1=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_40\images\" + image1_num + ".png");
    % image2=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_40\images\" + image2_num + ".png");


    image1=im2double(imread("/home/aub/datasets/sequence_30_og/images/" + image1_num + ".jpg"));
    image2=im2double(imread("/home/aub/datasets/sequence_30_og/images/" + image2_num + ".jpg"));

    %% you did not normalizeeeeeeeeeeeeeeeeeeeeeee previously now fixed

    % I1=correct_avg(image1,image1_left1,image1_left2,image1_left3);
    % I2=correct_avg(image2,image2_right1,image2_right2,image2_right3);
    I1=image1;
    I2=image2;




    %% exposures
    temp_exposures = dlmread('/home/aub/datasets/sequence_30_og/times.txt' );
    exposures_truth = temp_exposures(:,3);
    image1_exposure = exposures_truth(image1_list(pair) + 1);                  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
    image2_exposure = exposures_truth(image2_list(pair) + 1);

    % exposure ratio
    exposure_ratio = (image1_exposure/image2_exposure);
    r=exposure_ratio;
    %get radius for  image
    % calculate a radius map
    radius_map_f1 = zeros(size(I1));
    center_I1 = size(I1)/2;
    for i = 1 :1:size(I1,1)
        for j = 1:1:size(I1, 2)

            radius_map_f1(i,j) = sqrt(   (i - center_I1(1))^2 + (j - center_I1(2))^2 );       
        end
    end

    radius_map_f1=radius_map_f1 /  max(radius_map_f1, [], 'all'); %% why normalize?!


    %% feature extraction
    pair_all=[pair_all image1_num];
    X_image1=[];
    X_image2=[];
    N_corr=200;         % deleted it in the function , it is taking them all 
    draw=0;
    feature_method="sift"
    tol=10; % needs modificaton
    if feature_method=="sift"
        [X_image1,X_image2]=meta_sift_corr_fn_photo(I1,I2,N_corr,draw);
    else
         %% using other feature detectors
% 

    tic

    % sift 
%     points1_ = detectSIFTFeatures(I1);
%     points2_ = detectSIFTFeatures(I2);
    %harris
%     points1_ = detectHarrisFeatures(I1);
%     points2_ = detectHarrisFeatures(I2);
%    surf
%     points1_ = detectSURFFeatures(I1);
%     points2_ = detectSURFFeatures(I2);

    % orb
    points1_ =detectORBFeatures(I1);
    points2_ =detectORBFeatures(I2);

    % Brisk
%     points1_ = detectBRISKFeatures(I1);
%     points2_ = detectBRISKFeatures(I2);
    % Fast 
%     points1_ = detectFASTFeatures(I1);
%     points2_ = detectFASTFeatures(I2);


    % strong 
    % N=3000;
    % points1 = selectStrongest(points1_,N);
    % points2 = selectStrongest(points2_,N);
%    Extract the features
    [f1,vpts1] = extractFeatures(I1,points1_);
    [f2,vpts2] = extractFeatures(I2,points2_);
    %Retrieve the locations of matched points

    indexPairs = matchFeatures(f1,f2) ;

    matchedPoints1 = vpts1(indexPairs(:,1));
    matchedPoints2 = vpts2(indexPairs(:,2));
    X_image1=matchedPoints1'
    X_image2=matchedPoints2'
    end
    %[N_start_img,N_last,N_corr,draw,draw_final] = deal(image1_list(pair),image2_list(pair),2000,0,0);
    %[X_image1,X_image2]=best_tracker(N_start_img,N_last,N_corr,draw,draw_final);


    %% calculate distance
    % x_sub_sq=( X_image1(1,:)- X_image2(1,:) ).^2;
    % y_sub_sq=( X_image1(2,:)- X_image2(2,:) ).^2;
    % d= x_sub_sq + y_sub_sq ;
    % dist= (d).^0.5;


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

    
    
    
    %% filling

    corr_pts=num_corr_points_clean ; % changing this drastically changes the result thus outliers play a major role , so remove them

    %num_corr_points
    counter=1
    exp_est_list=[]

    for i=1:corr_pts



        [ M1 , M2,R1,R2]=deal( M_f1(i), M_f2(i),double(R_f1(i)),double(R_f2(i)) );
        if M1>0.6 || M2>0.6
            continue
        else 


            invf1= polyval(p,M1); %+ M1^3*X(4) + M1^4*X(5) + M1^5*X(6) + M1^6*X(7) + M1^7*X(8) + M1^8*X(9) + M1^9*X(10); 
            invf2= polyval(p,M2);
            Vf1=polyval(vign_truth_fun, R1);
            Vf2=polyval(vign_truth_fun, R2);

            exp_est_list(counter)=(invf1/invf2)/(Vf1/Vf2);
            counter=counter+1;
        end

    end

estimated_list_mean = [ estimated_list_mean  mean(exp_est_list) ];

estimated_list_median =  [   estimated_list_median        median(exp_est_list)];
truth=[truth exposure_ratio]    
    
    

    
   
end





figure
plot(truth,'LineWidth',2)
hold on

%plot(estimated_list_mean,'LineWidth',1)
%hold on
plot(estimated_list_median,'LineWidth',2)




