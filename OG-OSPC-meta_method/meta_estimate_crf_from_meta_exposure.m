
%{
Here we will use meta exposure to get the crf
data: only two images having:

                     - very small displacements 
                     - exposure ratio different that 1 
                     - good range of intensities  [0 ..... 1]
                     - no outliers 


%}



clc 
clear 

%% read crf
crf_truth=dlmread('/home/aub/datasets/sequence_30_og/pcalib.txt' );

figure(1)
plot(crf_truth)


%% get corresponding points %0.95 0.9376 0.9376

num_image1="00035"; %105 1  35 185 1585
num_image2="00036"; %106 36 36 186 1586

%distance below 10  as lower bound, still works at d 250 
I1 = im2double(imread("/home/aub/datasets/sequence_30_og/images/" + num_image1 + ".jpg" )); %0 102
I2 = im2double(imread("/home/aub/datasets/sequence_30_og/images/" + num_image2 + ".jpg" )); %15 128


%I1 = rgb2gray (im2double(imread('C:\Users\User\Desktop\102_m.jpg'))); 
%I2 = rgb2gray (im2double(imread('C:\Users\User\Desktop\108_m.jpg'))); 
figure
imshow(I1)
figure
imshow(I2)

%% exposures
temp_exposures = dlmread('/home/aub/datasets/sequence_30_og/times.txt' );
exposures_truth = temp_exposures(:,3);
image1_exposure = exposures_truth(str2double(num_image1) + 1);
image2_exposure = exposures_truth(str2double(num_image2) + 1);

% exposure ratio
exposure_ratio = (image1_exposure/image2_exposure);


%% feature extraction  and matching 
feature_method="sif"
tol=10;
if feature_method=="sift"
    N_corr=500; % deleted it in the function , it is taking them all 
    draw=0;
    tic
    [X_image1,X_image2,limit]=meta_sift_corr_fn_photo(I1,I2,N_corr,draw);
    toc
    % calculate distance
    x_sub_sq=( X_image1(1,:)- X_image2(1,:) ).^2;
    y_sub_sq=( X_image1(2,:)- X_image2(2,:) ).^2;
    d= x_sub_sq + y_sub_sq ;
    dist= (d).^0.5;
    %% tol 
    X_image1=X_image1(:,dist<tol);
    X_image2=X_image2(:,dist<tol);
    figure ; clf ;
    imagesc(cat(2, I1, I2)) ;


    xa = X_image1(1,:) ;
    xb = X_image2(1,:) + size(I1,2) ;
    ya = X_image1(2,:) ;
    yb = X_image2(2,:) ;
    
    hold on ;
    h = line([xa ; xb], [ya ; yb]) ;
    set(h,'linewidth', 1, 'color', 'r') ;
    
    
    num_corr_points=size(X_image1(1,:),2);
    for corr=1:num_corr_points

        M_f1(1,corr)=I1(round(X_image1(2,corr)),round(X_image1(1,corr))); %% check why inverted since y,x row col
        M_f2(1,corr)=I2(round(X_image2(2,corr)),round(X_image2(1,corr)));

    end

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
%    points1_ = detectSURFFeatures(I1);
%     points2_ = detectSURFFeatures(I2);

    % orb
    points1_ =detectORBFeatures(I1);
    points2_ =detectORBFeatures(I2);

    % Briskz
%     points1_ = detectBRISKFeatures(I1);
%     points2_ = detectBRISKFeatures(I2);
    % Fast 
%     points1_ = detectFASTFeatures(I1);
%     points2_ = detectFASTFeatures(I2);


    % strong 
%     N=3000;
%     points1 = selectStrongest(points1_,N);
%     points2 = selectStrongest(points2_,N);
%    Extract the features
    [f1,vpts1] = extractFeatures(I1,points1_);
    [f2,vpts2] = extractFeatures(I2,points2_);
    %Retrieve the locations of matched points

    indexPairs = matchFeatures(f1,f2) ;

    matchedPoints1 = vpts1(indexPairs(:,1));
    matchedPoints2 = vpts2(indexPairs(:,2));
    %display

    figure; showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2);
    legend('matched points 1','matched points 2');

    matchedPointsf1_r=round(matchedPoints1.Location);
    matchedPointsf2_r=round(matchedPoints2.Location);
    toc
    %% calculate distance
    x_sub_sq=( matchedPointsf1_r(:,1)- matchedPointsf2_r(:,1) ).^2;
    y_sub_sq=( matchedPointsf1_r(:,2)- matchedPointsf2_r(:,2) ).^2;
    d= x_sub_sq + y_sub_sq ;
    dist= (d).^0.5;
   
    matchedPointsf1_r=matchedPointsf1_r(dist<tol,:);
    matchedPointsf2_r=matchedPointsf2_r(dist<tol,:);
    % draw 
    figure(2) ; clf ;
    imagesc(cat(2, I1, I2)) ;

    xa = matchedPointsf1_r(:,1)' ;
    xb = matchedPointsf2_r(:,1)' + size(I1,2) ;
    ya = matchedPointsf1_r(:,2)' ;
    yb = matchedPointsf2_r(:,2)' ;

    hold on ;
    h = line([xa ; xb], [ya ; yb]) ;
    set(h,'linewidth', 1, 'color', 'b') ;

    num_corr_points=size(matchedPointsf1_r(:,1),1);


    for corr=1:num_corr_points 

        M_f1(1,corr)=I1(matchedPointsf1_r(corr,2),matchedPointsf1_r(corr,1)); %% check why inverted 
        M_f2(1,corr)=I2(matchedPointsf2_r(corr,2),matchedPointsf2_r(corr,1));

    end


    num_corr_points=size(M_f1,2);
end

%% filling 
corr_pts=num_corr_points; % changing this drastically changes the result thus outliers play a major role , so remove them

%% filling
%try r estimate 
r=exposure_ratio;
%% bins 
[Y,E] = discretize(M_f1,10);   

counter=1
for corr_pt =  1:corr_pts
    
% each iter adds one equation from one pair of corresponding points    
[M1,M2 ] = deal(double(M_f1(corr_pt)), double(M_f2(corr_pt)));
bin_number=Y(corr_pt);
length_bin=sum(Y==bin_number);
    %% found a mistake
     %if M1 ~= 1 || M2 ~=1 || M1 ~= 0 || M2 ~= 0
     if  M1 == 1 || M2 == 1 ||M1 == 0 || M2 == 0 || M1==M2 || M1>M2
     else
        %% no weight 
        row=[ (1 - r) (M1 - r*M2)   (M1^2- r*M2^2)  ];    %(M1^3- r*M2^3)  (M1^4- r*M2^4)]; %(M1^5- r*M2^5) (M1^6- r*M2^6) (M1^7- r*M2^7) (M1^8- r*M2^8) (M1^9- r*M2^9)];
        %% intensity weight 
        % if bin_number==1
        %row=(1/length_bin)*[ (1 - r) (M1 - r*M2)   (M1^2- r*M2^2)  ];
        % else
        %row=(1/length_bin)*[ (1 - r) (M1 - r*M2)   (M1^2- r*M2^2)  ];
        % end
        %% intensity exponential 
        %row=(1/exp(length_bin))*[ (1 - r) (M1 - r*M2)   (M1^2- r*M2^2)  ];    %(M1^3- r*M2^3)  (M1^4- r*M2^4)]; %(M1^5- r*M2^5) (M1^6- r*M2^6) (M1^7- r*M2^7) (M1^8- r*M2^8) (M1^9- r*M2^9)];

        %% idistance + intensity weight 

        %row=(1/length_bin)*double((1/dist(corr_pt)))*[ (1 - r) (M1 - r*M2)   (M1^2- r*M2^2)  ];    %(M1^3- r*M2^3)  (M1^4- r*M2^4)]; %(M1^5- r*M2^5) (M1^6- r*M2^6) (M1^7- r*M2^7) (M1^8- r*M2^8) (M1^9- r*M2^9)];
        %row=( 1/( length_bin + dist(corr_pts) ))*[ (1 - r) (M1 - r*M2)   (M1^2- r*M2^2)  ];    %(M1^3- r*M2^3)  (M1^4- r*M2^4)]; %(M1^5- r*M2^5) (M1^6- r*M2^6) (M1^7- r*M2^7) (M1^8- r*M2^8) (M1^9- r*M2^9)];

        A(counter,:)=row;
        %F(corr_pt)= mult;
        k=1;
        counter=counter + 1;
      end
end



%% solve 

n=3;
cvx_begin

    variable x(n)
    %b=ones(corr_pts,1)
    %minimize( norm(A*x ) );
    minimize( (A*x )'*(A*x) );
% with constraint 
    subject to 
    
        sum(x) == 1;
        
        x(1)==0; % this might be very helpful , we can get rid of it if we
        %have good data (balanced )
        
cvx_end

H=A'*A;
[V,D] = eig(H);
evector_min=V(:,1)./(V(1,1)+V(2,1)+V(3,1));


%[0;0.694524887880286;0.305475112119714]
%[0;0.071993869034820;0.928006130965180]

%% plot 
%plot 
input=0:1/255:1;

% output= evector_min(1,1) + evector_min(2,1)*input +  evector_min(3,1)* input.^2 %+ evector_min(4,1)* input.^3 + evector_min(5,1)* input.^4 + evector_min(6,1)* input.^5 + evector_min(7,1)* input.^6 + evector_min(8,1)* input.^7 + evector_min(9,1)* input.^8 + evector_min(10,1)* input.^9 + evector_min(11,1)* input.^10);
%including 1-c2-c3 manually 
output=  x(1) + x(2)*input +  x(3)* input.^2;  %+  x(4)*input.^3 +  x(5)* input.^4 %+ x(6)*input.^5 +  x(7)* input.^6 + x(8)*input.^7 +  x(9)* input.^8 +  x(10)* input.^9;
% output= evector_min(1) + evector_min(2)*input +  evector_min(3)* input.^2
%output=   ( 1-evector_min(1,1) - evector_min(2,1) ) + evector_min(1,1)*input +  evector_min(2,1)* input.^2 ;
%output=  (x(1) + x(2)*input +  x(3)* input.^2); %./  (x(1) + x(2)*1 +  x(3)* 1^2) ;
%plot(input,output)
figure('Name','j')
plot(input,output)
hold on

plot(input,crf_truth/255)


%% polyfit 

order=3
p = polyfit(input,crf_truth/255,order) ;

f1 = polyval(p,input);
plot(input,f1)
hold on
plot(input,crf_truth/255)


%{
surf is acting better than orb since I think better distribution of points
 although orb is giving more corresponding points but many of them are at
 the same place 

orb is not improving once we balance the data
surf improved when we balance the data

try removing v and then do the experiment 

if i took points in a v free region will this improve?
surf and sift are better at detecting pixels with 10% intensity than orb

adding a weight is having a better effect than adding the degree of
polynomial 


%}