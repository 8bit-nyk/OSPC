
function output = estimate_vignette_refactored(image_folder_path, exposure_file_path,crf_file_path)
      % Initialize parameters and variables
    [window, step, t, image1_list, image2_list] = initializeParameters();
    %crf_coefficients = loadCRF(crf_file_path);
    [A,l,radius_map,I1] = processImagePairs(image_folder_path, exposure_file_path,crf_file_path, t, image1_list, image2_list);
    output = computeV_convex(A,l,radius_map,I1);
end

function [window, step, t, image1_list, image2_list] = initializeParameters()
    % Specify and initialize adjustable parameters
    window = 100;
    step = 20;
    image1_list = 0:0+window; % 0:0+window
    image2_list = step:step+window;
    t = 0.01; % Threshold on displacement
end

function crf_coefficients = loadCRF(crf_file_path)
    %Load crf 
    crf = readmatrix(crf_file_path);
    input_range = 0:1/255:1;
    % Fit a polynomial of degree 10 to approximate the ground truth Camera Response Function (CRF)
    crf_coefficients = polyfit(input_range, crf, 10);
end



function [A,l,radius_map,I1] = processImagePairs(image_folder_path, exposure_file_path, crf_file_path, t, image1_list, image2_list)
    A = [];
    l=[];
    %length_matchpts=1;
    % Process each image pair
    for pair = 1:size(image1_list, 2)
        % Load images and calculate exposure ratio
        image1_num=image1_list(pair) ;
        image2_num=image2_list(pair) ;
        [I1, I2, r,image1_exposure,image2_exposure] = loadImagePairAndExposureRatio(image_folder_path, exposure_file_path, image1_num, image2_num);
        
        % Compute radius map
        radius_map = computeRadiusMap(I1);
        %save("outputs/r_map_ref_" + image1_list(pair) + ".mat","radius_map");
        [A_pair, l_pair] = processSelectedImagePair(I1, I2,image1_exposure,image2_exposure,radius_map,t,crf_file_path);
        A = [A; A_pair];
        l = [l, l_pair];
        %length_matchpts=[ length_matchpts size(M1_list,2) ]
    end
    l=l';
end

function [I1, I2, r,image1_exposure,image2_exposure] = loadImagePairAndExposureRatio(image_folder_path, exposure_file_path, image1_num, image2_num)
    % Load images
    %image1 = im2double(imread(strcat(image_folder_path, "/image_", int2str(image1_num), ".jpg")));
    %image2 = im2double(imread(strcat(image_folder_path, "/image_", int2str(image2_num), ".jpg")));
    %image1=im2double(imread( image_folder_path + "/frame_" + image1_num + ".png"));
    %image2=im2double(imread( image_folder_path + "/frame_" + image2_num + ".png"));
    
    %Read TUM dataset format
    image1_num=get_num_image(image1_num) ;
    image2_num=get_num_image(image2_num) ;
    image1=im2double(imread(image_folder_path + image1_num + ".png"));
    image2=im2double(imread(image_folder_path + image2_num + ".png"));
    
    I1 = im2gray(image1);
    I2 = im2gray(image2);
    % Load exposure values and calculate exposure ratio
    temp_exposures = readmatrix(exposure_file_path);
    image1_exposure = temp_exposures(str2num(image1_num) +1, 2);
    image2_exposure = temp_exposures(str2num(image2_num) +1, 2);
    %Read TUM dataset format
    % image1_exposure = temp_exposures(str2num(image1_num) + 1, 3);
    % image2_exposure = temp_exposures(str2num(image2_num) + 1, 3);
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

function [A, l] = processSelectedImagePair(I1, I2,image1_exposure,image2_exposure,radius_map,t,crf_file_path)
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
    %save("outputs/theta_ref.mat","theta");
    %save("outputs/region_pts_ref.mat","region_f1_points");
    %draw =0;
    [X_image1, X_image2] = direction_filter(X_image1, X_image2, num_regions, region_f1_points, theta, I1, I2, draw, ratio_std);
    %imwrite(X_image1, "outputs/X_image1_"+ image1_num +".png");
    %imwrite(X_image2, "outputs/X_image2_"+ image2_num +".png");
    %Process the matched points
    [A, l] = processMatchedPoints(I1, I2,image1_exposure,image2_exposure, X_image1,X_image2,radius_map,t,crf_file_path);
   
end

function [X_image1, X_image2] = removeOutliers(X_image1, X_image2)
    distances = sqrt(sum((X_image1 - X_image2).^2, 1));
    index_dist = distances < mean(distances) + 2*std(distances);
    X_image1 = X_image1(:, index_dist);
    X_image2 = X_image2(:, index_dist);
end


function [A, l] = processMatchedPoints(I1, I2,image1_exposure,image2_exposure, X_image1,X_image2,radius_map,t,crf_file_path)
    
    A=[];
    l=[];
    num_corr_points = size(X_image1, 2);
    counter = 1;
    crf_coeff = loadCRF(crf_file_path);

    for corr = 1:num_corr_points
        x1 = round(X_image1(1, corr));
        y1 = round(X_image1(2, corr));
        x2 = round(X_image2(1, corr));
        y2 = round(X_image2(2, corr));

        M1 = double(I1(y1, x1));
        M2 = double(I2(y2, x2));
        R1 = radius_map(y1, x1);
        R2 = radius_map(y2, x2);

        if M1 == 0 || M2 == 0 || M1 == 1 || M2 == 1 || M1 == M2 || abs(R2 - R1) < t
            continue;
        end
        invf1 = polyval(crf_coeff, M1);
        invf2 = polyval(crf_coeff, M2);

        J = (invf1 * image2_exposure) / (invf2 * image1_exposure);
        a = R1^2 - J * R2^2;
        b = R1^4 - J * R2^4;
        c = R1^6 - J * R2^6;

        A(counter, :) = [a, b, c];
        l(counter) = J - 1;
        counter = counter + 1;
    end


end

function output = computeV_convex(A,l,radius_map,I1)
    in= 0:radius_map(1,1)/1000:radius_map(1,1) ;
    n=3;
    cvx_begin
        variable x(n)
        minimize(norm( A*x - l ,1) )
    cvx_end
   output = 1 + x(1)*in.^2 +  x(2)*in.^3 + x(3)*in.^4;
     vignette_image = zeros(size(I1));
    for i = 1:1:size(vignette_image,1)
        for j = 1:1:size(vignette_image, 2) 
    
           temp=1 + x(1)*radius_map(i,j)^2 + x(2)*radius_map(i,j)^4 + x(3)*radius_map(i,j)^6;
           
           if temp  >= 0 
           vignette_image(i,j) = temp;   
            end
        end
    end 

    % Resize the vignette_image to the target resolution (width x height)
    %resized_vignette_image_1280 = imresize(vignette_image, [1024, 1280]);
    resized_vignette_image_640 = imresize(vignette_image, [480, 640]);
    % Convert resized_vignette_image to 16-bit grayscale before writing
    %vignette_image_16bit_1280 = uint16(resized_vignette_image_1280 * 65535); % Scale values to 0-65535
    vignette_image_16bit_640 = uint16(resized_vignette_image_640 * 65535); % Scale values to 0-65535
    % Define the output file path for the 16-bit grayscale image
    %outputFilePath_1280 = 'outputs/vig_1280_TUMwide2.png';
    outputFilePath_640 = 'outputs/vig_d435.png';
    % Write the vignette_image as a 16-bit grayscale image with the specified resolution
    %imwrite(vignette_image_16bit_1280, outputFilePath_1280,  'BitDepth', 16);
    imwrite(vignette_image_16bit_640, outputFilePath_640,  'BitDepth', 16);
    %imwrite(vignette_image_16bit, outputFilePath, 'png'); % Save as PNG

    figure('Name',"Vignette Image")
    imshow(vignette_image)

    figure('Name','vignette with x')
   plot( in, output );
    

    hold on
 
end

