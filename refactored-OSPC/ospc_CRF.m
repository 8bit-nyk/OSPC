
function output = ospc_CRF(image_folder_path, exposure_file_path)
    % Initialize parameters and variables
    [window, step, t, image1_list, image2_list] = initializeParameters();

    % Process image pairs
    [A] = processImagePairs(image_folder_path, exposure_file_path,t, image1_list, image2_list);
    
    % Compute the Camera Response Function (CRF) parameters
    output = computeCRF(A);
    %output=1;
end

function [window, step, t, image1_list, image2_list] = initializeParameters()
    % Specify and initialize adjustable parameters
    window = 200;
    step = 100;
    image1_list = 0:0+window; % 0:0+window
    image2_list = step:step+window;
    t = 0.01; % Threshold on displacement
end

function [I1, I2, r] = loadImagePairAndExposureRatio(image_folder_path, exposure_file_path, image1_num, image2_num)
    % Load images
    %image1 = im2double(imread(strcat(image_folder_path, "/image_", int2str(image1_num), ".png")));
    %image2 = im2double(imread(strcat(image_folder_path, "/image_", int2str(image2_num), ".png")));
    %image1 = im2double(imread(strcat(image_folder_path, "/image_", int2str(image1_num), ".jpg")));
    %image2 = im2double(imread(strcat(image_folder_path, "/image_", int2str(image2_num), ".jpg")));
    %image1=im2double(imread( image_folder_path + "/frame_" + image1_num + ".png"));
    %image2=im2double(imread( image_folder_path + "/frame_" + image2_num + ".png"));
    %image1=im2double(imread( image_folder_path + "/Frame-" + image1_num + ".png"));
    %image2=im2double(imread( image_folder_path + "/Frame-" + image2_num + ".png"));
   
    %Read form TUM dataset
    image1_num=get_num_image(image1_num) ;
    image2_num=get_num_image(image2_num) ;
    image1=im2double(imread(image_folder_path + image1_num + ".jpg"));
    image2=im2double(imread(image_folder_path + image2_num + ".jpg"));
  
    I1 = im2gray(image1);
    I2 = im2gray(image2);
    
    % Load exposure values and calculate exposure ratio
     temp_exposures = readmatrix(exposure_file_path);
    % image1_exposure = temp_exposures(str2num(image1_num) + 1, 2);  
    % image2_exposure = temp_exposures(str2num(image2_num) + 1, 2);
    % %Read from TUM exposure file format
    image1_exposure = temp_exposures(str2num(image1_num) + 1, 3);
    image2_exposure = temp_exposures(str2num(image2_num) + 1, 3);
    r = image1_exposure / image2_exposure;
    
end

function radius_map = computeRadiusMap(I)
    % Compute a radius map for the given image
    radius_map = zeros(size(I));
    center_I = round(size(I) / 2);
    for i = 1:size(I,1)
        for j = 1:size(I,2)
            radius_map(i,j) = sqrt((i - center_I(1))^2 + (j - center_I(2))^2);
        end
    end
    radius_map = radius_map / max(radius_map, [], 'all');
   
end

function [A] = processImagePairs(image_folder_path, exposure_file_path, t, image1_list, image2_list)
    A = [];
    length_matchpts=[1];
    % Process each image pair
    for pair = 1:size(image1_list, 2)
        % Load images and calculate exposure ratio
        [I1, I2, r] = loadImagePairAndExposureRatio(image_folder_path, exposure_file_path, image1_list(pair), image2_list(pair));
        
        % Compute radius map
        radius_map_f1 = computeRadiusMap(I1);
       % Save the radius map as an image
        %imwrite(radius_map_f1, sprintf('outputs/images_ref/radius_map_pair_%d.png', pair));


        % Process image pair if exposure ratio is within the desired range
        if r < 0.92 || r > 1.08
            % str = sprintf("Ratio accepted in Refactored");
            % disp(str);
            [A_pair, M1_list, ~] = processSelectedImagePair(I1, I2, r, radius_map_f1, t);
            A = [A; A_pair];
        end
        length_matchpts =[ length_matchpts size(M1_list,2) ];
        
    end
    save("outputs/len_mpoints_ref.mat","length_matchpts");

end

function [A, M1_list, M2_list] = processSelectedImagePair(I1, I2, r, radius_map, t)
    % Feature matching using SIFT
    draw =0;
    ratio_std = 0.1;  % Adjust this as needed
    [X_image1, X_image2] = sift(I1, I2,draw);
   
    % Remove outliers based on distance
    [X_image1, X_image2] = removeOutliers(X_image1, X_image2);
    
    % Directional filtering
    [num_blocks_x, num_blocks_y, num_rows, num_cols] = deal(15, 15, size(I1, 1), size(I1, 2));
    region_image = create_region_img(num_blocks_x, num_blocks_y, num_rows, num_cols);
    num_regions = num_blocks_x * num_blocks_y;
    
    num_corr_points=size(X_image1(1,:),2);

    for corr=1:num_corr_points   %% round or round

         theta( 1,corr )=rad2deg(atan2(round(X_image2(2,corr)-X_image1(2,corr)),round(X_image2(1,corr)-X_image1(1,corr))));
         region_f1_points(1,corr)=region_image( round(X_image1(2,corr)) , round(X_image1(1,corr)) );

    end 
   % theta = rad2deg(atan2(X_image2(2, :) - X_image1(2, :), X_image2(1, :) - X_image1(1, :));
   % region_f1_points = region_image(sub2ind(size(region_image), round(X_image1(2, :)), round(X_image1(1, :))));
    %draw =1;
    [X_image1, X_image2] = direction_filter(X_image1, X_image2, num_regions, region_f1_points, theta, I1, I2, draw, ratio_std);
    
    % Process the matched points
    [A, M1_list, M2_list] = processMatchedPoints(I1, I2, X_image1, X_image2, r, radius_map, t);
    
end

function [X_image1, X_image2] = removeOutliers(X_image1, X_image2)
    distances = sqrt(sum((X_image1 - X_image2).^2, 1));
    index_dist = distances < mean(distances) + 2*std(distances);
    X_image1 = X_image1(:, index_dist);
    X_image2 = X_image2(:, index_dist);
end

function [A, M1_list, M2_list] = processMatchedPoints(I1, I2, X_image1, X_image2, r, radius_map, t)
    A = [];
    M1_list = [];
    M2_list = [];
    counter =1;
    %iterate over clean Ximage_1 after direction filtering
    for i = 1:size(X_image1, 2)
        x_M1 = round(X_image1(2, i));
        y_M1 = round(X_image1(1, i));
        x_M2 = round(X_image2(2, i));
        y_M2 = round(X_image2(1, i));

        M1 = double(I1(x_M1, y_M1));
        M2 = double(I2(x_M2, y_M2));
        R1 = radius_map(x_M1, y_M1);
        R2 = radius_map(x_M2, y_M2);  % Adjust if there's a separate radius_map for I2

        if M1 == 1 || M2 == 1 || M1 == 0 || M2 == 0 || M1 == M2 || ...
           (M1 < M2 && r > 1) || (M1 > M2 && r < 1) || abs(R2 - R1) > t
            continue;
        end
        
        M1_list = [M1_list, M1];
        M2_list = [M2_list, M2];
        %A = [A; (1 - r), (M1 - r*M2), (M1^2 - r*M2^2)];
        row=[ (1 - r) (M1 - r*M2)   (M1^2- r*M2^2)];
        
        A(counter,:)=row;   
        counter=counter + 1;
    end
end

function output = computeCRF(A)
    % Compute the Camera Response Function (CRF) parameters
    if isempty(A) 
        error('The matrix A should not be empty.');
    end
   % Define the original matrix A
    A_all = A;
    save("outputs/debug_ref.mat","A_all");
    % Define the number of elements in vector x
    n = 3;
    
    % Create an optimization problem
    cvx_begin
        variable x(n)
    
        % Define the objective function to minimize the 2-norm of A*x
        minimize(norm(A_all * x, 2));
    
        % Define constraints
        subject to 
           sum(x) == 1;
           x(1) == 0;
           x >= 0; % Ensure non-negative values for x
    cvx_end
    
    % Define the input range for the CRF plot
    input_range = 0:1/255:1;%input_range = 0:1:255
    % Calculate the CRF output using the fitted polynomial
    output = x(1) + x(2) * input_range + x(3) * input_range.^2;

    
    % Calculate the normalization ratio for coefficients 
    %USed for error calculation TODO(compares to groundtruth)
    %sum_coefficients = sum(x);
    %normalization_ratio = 1 / sum_coefficients;


    % Calculate the CRF output using the fitted polynomial
    %output = x(1) + x(2) * (input_range * 255) + x(3) * (input_range * 255).^2;  % Scale input_range to 0-255
    % Set negative values to 0
    %output(output < 0) = 0;
    %output2 = output * 255;
    %writematrix (output,'/home/aub/Dev/Matlab/outputs/crf4.txt');

 
end
