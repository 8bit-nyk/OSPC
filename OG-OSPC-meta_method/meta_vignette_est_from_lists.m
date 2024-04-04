clc
clear all

%% inv_crf
% read
inv_crf_truth = dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\pcalib.txt' );

% polyfit 
inv_crf_order = 4;
inv_crf_truth_input = 0:1/255:1;
inv_crf_truth_fun = polyfit(inv_crf_truth_input, inv_crf_truth/255, inv_crf_order);
inv_crf_eval = polyval(inv_crf_truth_fun, inv_crf_truth_input);
figure('Name','inv_crf_truth')
plot(inv_crf_truth_input, inv_crf_eval)
hold on
plot(inv_crf_truth_input, inv_crf_truth/255)
hold off 

%% images lists
% read 
num_image1 = 500*ones(1,10); 
num_image2 = 510:520; 

%% exposures 
times_list = dlmread('C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\times.txt');
exposures_list = times_list(:,3);

%% vignette
vignette_truth = imread('C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\vignette.png');
vignette_truth_norm = im2double(vignette_truth);

% calculate a radius map
% NOTE: center is shifted by half a pixel, since we have even numbers of
% rows and cols
radius_map = zeros(size(vignette_truth_norm));
center_I1 = size(vignette_truth_norm)/2;
for i = 1 :1:size(vignette_truth_norm,1)
    for j = 1:1:size(vignette_truth_norm, 2)       
        radius_map(i,j) = sqrt((i - center_I1(1))^2 + (j - center_I1(2))^2);       
    end
end

% normalize radius map
radius_map_norm = radius_map/max(radius_map, [], 'all');

%% features detection, extraction and matching
A_all=[];
B_all=[];

for img_pair = 1:size(num_image1, 2)
    image1 = imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\images\00" + num2str(num_image1(img_pair)) + ".jpg");
    image2 = imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\images\00" + num2str(num_image2(img_pair)) + ".jpg");

    % normalize
    image1_normalized = im2double(image1);
    image2_normalized = im2double(image2);

    image1_exposure = exposures_list(num_image1(img_pair) + 1);
    image2_exposure = exposures_list(num_image2(img_pair) + 1);

    % exposure ratio
    exposure_ratio = (image1_exposure/image2_exposure);

    % sift 
%     points1 = detectSIFTFeatures(image1_normalized);
%     points2 = detectSIFTFeatures(image2_normalized);
    % orb 
    points1 = detectORBFeatures(image1_normalized);
    points2 = detectORBFeatures(image2_normalized);
    
    % strongest points
    % N=3000;
    % points1 = selectStrongest(points1,N);
    % points2 = selectStrongest(points2,N);
    
    % Extract the features
    [f1, vpts1] = extractFeatures(image1_normalized, points1);
    [f2, vpts2] = extractFeatures(image2_normalized, points2);
    
    %Retrieve the locations of matched points
    indexPairs = matchFeatures(f1, f2);
    matchedPoints1 = vpts1(indexPairs(:,1));
    matchedPoints2 = vpts2(indexPairs(:,2));
    
    % display matches
    figure("Name", "Matches with outliers " + num2str(num_image1(img_pair)) + " with " + num_image2(img_pair)); 
    showMatchedFeatures(image1_normalized, image2_normalized, matchedPoints1, matchedPoints2);
    legend('matched points 1', 'matched points 2');
    
    % rounding
    matchedPointsf1_r = round(matchedPoints1.Location);
    matchedPointsf2_r = round(matchedPoints2.Location);
    
    % eliminate outliers
    x_sub_sq = (matchedPointsf1_r(:,1) - matchedPointsf2_r(:,1)).^2;
    y_sub_sq = (matchedPointsf1_r(:,2) - matchedPointsf2_r(:,2)).^2;
    d_vec = x_sub_sq + y_sub_sq ;
    dist_vec_sq = (d_vec).^0.5;
    tol = 100;
    matchedPointsf1_r = matchedPointsf1_r(dist_vec_sq<tol,:);
    matchedPointsf2_r = matchedPointsf2_r(dist_vec_sq<tol,:);
    % draw 
    figure("Name", "Matches after removing outliers " + num2str(num_image1(img_pair)) + " with " + num_image2(img_pair)) ; clf;
    imagesc(cat(2, image1_normalized, image2_normalized));
    
    xa = matchedPointsf1_r(:,1)' ;
    xb = matchedPointsf2_r(:,1)' + size(image1_normalized,2) ;
    ya = matchedPointsf1_r(:,2)' ;
    yb = matchedPointsf2_r(:,2)' ;
    
    hold on ;
    h = line([xa ; xb], [ya ; yb]) ;
    set(h,'linewidth', 1, 'color', 'b') ;
    hold off

    % system
    num_corr_points = size(matchedPointsf1_r(:,1),1);
    
    for corr = 1:num_corr_points
        % intensities
        m1 = image1_normalized(matchedPointsf1_r(corr,2), matchedPointsf1_r(corr,1)); 
        m2 = image2_normalized(matchedPointsf2_r(corr,2), matchedPointsf2_r(corr,1));
    
        % radius
        r1(corr, img_pair) = radius_map_norm(matchedPointsf1_r(corr,2), matchedPointsf1_r(corr,1));
        r2(corr, img_pair) = radius_map_norm(matchedPointsf2_r(corr,2), matchedPointsf2_r(corr,1));
    
        % irradiances
        i1 = polyval(inv_crf_truth_fun, m1);
        i2 = polyval(inv_crf_truth_fun, m2);
        
        col1 = (i1/i2) * r2(corr, img_pair)^2 - exposure_ratio * r1(corr, img_pair)^2;
        col2 = (i1/i2) * r2(corr, img_pair)^4 - exposure_ratio * r1(corr, img_pair)^4;
        col3 = (i1/i2) * r2(corr, img_pair)^6 - exposure_ratio * r1(corr, img_pair)^6;
        
        A(corr,:) = [col1 col2 col3];  
        B(corr,:) = exposure_ratio - i1/i2;
    end
    
    A_all = [A_all; A];
    B_all = [B_all; B];
    A=[]; 
    B=[];

end


%% solve
% n=3;
% cvx_begin
%     variable V(n)
%    
%     minimize( norm(A*V - B) ); %try with and without square
%     %minimize( power(norm(A*x ),2) )
%     %minimize(power(2,norm(A*x,2)))
%     %minimize( (A*x)'*(A*x) )
% % with constraint 
%     subject to 
%     sum(V) > 0;
% cvx_end

%evector_min=x;
%[V,D]=eig(A'*A)

V = inv(A_all'*A_all)*A_all'*B_all;

%% plot vignette truth vs estimate

% estimate image
vignette_est_image = ones(size(image1_normalized));
for i = 1:1:size(vignette_est_image,1)
    for j = 1:1:size(vignette_est_image, 2) 
       vignette_est_image(i,j) = 1 + V(1) * radius_map_norm(i,j)^2 + V(2) * radius_map_norm(i,j)^4 + V(3) * radius_map_norm(i,j)^6;       
    end
end 

figure('Name', 'vignette estimate image')
imshow(vignette_est_image)

figure('Name','vignette truth vs estimate')
% function vignette truth
eps = 0.005;
radius_map_norm_sub = radius_map_norm(center_I1(1):size(image1_normalized,1), center_I1(2):size(image1_normalized,2));
vignette_truth_norm_sub = vignette_truth_norm(center_I1(1):size(image1_normalized,1), center_I1(2):size(image1_normalized,2));

rad_counter = 0;
for rad = 0:0.01:1
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

% function vignette estimate
vign_est_fun = [V(3) 0 V(2) 0 V(1) 0 1]; % descending order
vign_est_eval = polyval(vign_est_fun, radius_input);
plot(radius_input, vign_est_eval)
hold off 

%% radius histograms
% figure('Name','rad 1 histogram')
% histogram(r1)
% hold off
% figure('Name','rad 2 histogram')
% histogram(r2)
% hold off