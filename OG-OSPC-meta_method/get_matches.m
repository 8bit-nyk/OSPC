function [matchedPointsf1_r,matchedPointsf2_r]=get_matches(I1,I2)
        %% orb
%         points1_ =detectORBFeatures(I1);
%         points2_ =detectORBFeatures(I2);
%     %    Extract the features
%         [f1,vpts1] = extractFeatures(I1,points1_);
%         [f2,vpts2] = extractFeatures(I2,points2_);
%         %[f3,vpts3] = extractFeatures(I3,points3_);
%         %Retrieve the locations of matched points
% 
%         indexPairs12 = matchFeatures(f1,f2) ;
% 
%         matchedPoints1_12 = vpts1(indexPairs12(:,1));
%         matchedPoints2_12 = vpts2(indexPairs12(:,2));
% 
%         matchedPointsf1_r=round(matchedPoints1_12.Location); 
%         matchedPointsf2_r=round(matchedPoints2_12.Location);

        
        
        %% sift 
        N_corr=200; %% to be changed 
        draw=0;
        [X_image1,X_image2,limit] = meta_sift_corr_fn_photo(I1,I2,N_corr,draw)
        
        matchedPointsf1_r=round(X_image1(1:2,:))';
        matchedPointsf2_r=round(X_image2(1:2,:))';
