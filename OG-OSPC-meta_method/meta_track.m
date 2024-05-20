clear all
clc


%% should have same exposure 
I1=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\images\000" + '20' + ".jpg");
I2=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\images\000" + '21' + ".jpg");
%I3=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\calib_narrowGamma_sweep3\000" + 32 + ".jpg");

I1_old=I1;
    % orb
    points1_ =detectORBFeatures(I1);
    points2_ =detectORBFeatures(I2);
    %points3_ =detectORBFeatures(I3);

%    Extract the features
    [f1,vpts1] = extractFeatures(I1,points1_);
    [f2,vpts2] = extractFeatures(I2,points2_);
    %[f3,vpts3] = extractFeatures(I3,points3_);
    %Retrieve the locations of matched points

    indexPairs12 = matchFeatures(f1,f2) ;
%     indexPairs13 = matchFeatures(f1,f3) ;

    matchedPoints1_12 = vpts1(indexPairs12(:,1));
    matchedPoints2_12 = vpts2(indexPairs12(:,2));
    
%     matchedPoints1_13 = vpts1(indexPairs13(:,1));
%     matchedPoints3_13 = vpts3(indexPairs13(:,2));
%     
    %display
%     figure; showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2);
%     legend('matched points 1','matched points 2');

    matchedPointsf1_r_12=round(matchedPoints1_12.Location); %%
    matchedPointsf2_r_12=round(matchedPoints2_12.Location);
    
%     matchedPointsf1_r_13=round(matchedPoints1_13.Location); %%
%     matchedPointsf3_r_13=round(matchedPoints3_13.Location);
    
    %% take the average corr points of I1 and I2
    num_corr_points12=size(matchedPointsf1_r_12(:,1),1);

    for corr=1:num_corr_points12 

        M_f1_12(1,corr)=I1(matchedPointsf1_r_12(corr,2),matchedPointsf1_r_12(corr,1)); %% check why inverted 
        M_f2_12(1,corr)=I2(matchedPointsf2_r_12(corr,2),matchedPointsf2_r_12(corr,1));

    end
     
    M_avg_12=  ( M_f1_12 + M_f2_12 )./2 ; 
    
    %% edit the image by the average 
    for i=1:1:size(M_avg_12,2)
    I1(matchedPointsf1_r_12(i,2),matchedPointsf1_r_12(i,1))=M_avg_12(1,i);
    end 
    
    
    
%     %% take the average corr points of I1 and I3
%     num_corr_points13=size(matchedPointsf1_r_13(:,1),1);
% 
% 
%     for corr=1:num_corr_points13 
% 
%         M_f1_13(1,corr)=I1(matchedPointsf1_r_13(corr,2),matchedPointsf1_r_13(corr,1)); %% check why inverted 
%         M_f2_13(1,corr)=I2(matchedPointsf3_r_13(corr,2),matchedPointsf3_r_13(corr,1));
% 
%     end
%     
%     %% take the average of the two averages 
%     
%     M_avg_13=  ( M_f1_13 + M_f2_13 )./2 ; 
%     
%     
%     %% average of average 
%     M_average_12_13=( M_avg_12 + M_avg_13  ) ./2 ; 
%     
%     
    
    
    