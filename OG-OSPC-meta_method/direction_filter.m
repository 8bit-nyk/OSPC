function [ X_image1_clean , X_image2_clean ] = direction_filter( X_image1,X_image2,num_regions,region_f1_points,theta,Ia,Ib,draw,ratio_std ) %ratio_std

    X_image1_clean=[];
    X_image2_clean=[];

    for region=1:num_regions
        theta_region=[];
        Y=[];
        E=[];
        X_image1_region=[];
        X_image2_region=[];

        %extract elements of current region
        if sum(region_f1_points==region)>5
            %region
            theta_region=theta(region_f1_points==region);
            X_image1_region=X_image1(:,region_f1_points==region);
            X_image2_region=X_image2(:,region_f1_points==region);

            % metric 
            %% frequency
            [Y,E] = discretize(theta(region_f1_points==region),10);
            most_frq_bin = mode(Y);
            boolean_indexes_mst_frq_bin=   Y==most_frq_bin ;
            % keep the best
            X_image1_clean=cat(2,X_image1_clean,X_image1_region(:, boolean_indexes_mst_frq_bin));
            X_image2_clean=cat(2,X_image2_clean,X_image2_region(:, boolean_indexes_mst_frq_bin));
            %%
            %% mean 
%             mean_theta_region=mean(theta_region);
%             std_theta_region=std(theta_region);
%             %% the error is here
%             boolean_indexes_within_std=   (theta_region< (mean_theta_region + ratio_std*std_theta_region) ) &  ( theta_region> (mean_theta_region - ratio_std*std_theta_region)  )        ;
%             
%             X_image1_clean=cat(2,X_image1_clean,X_image1_region(:,  boolean_indexes_within_std));
%             X_image2_clean=cat(2,X_image2_clean,X_image2_region(:,  boolean_indexes_within_std));
            %% mean +-
%             mean_theta_region=mean(theta_region);
%             std_theta_region=std(theta_region);
%             boolean_indexes_within_std=   (theta_region< (mean_theta_region + degrees) ) &  ( theta_region> (mean_theta_region - degrees)  )        ;
%             
%             X_image1_clean=cat(2,X_image1_clean,X_image1_region(:,  boolean_indexes_within_std));
%             X_image2_clean=cat(2,X_image2_clean,X_image2_region(:,  boolean_indexes_within_std));
        else 
            continue 
        end
        
    end

    if draw==1

        points_img_one= X_image1_clean;
        points_img_final= X_image2_clean;

        figure ; %clf ;
        imagesc(cat(2, Ia, Ib)) ;

        xa = points_img_one(1,:);
        xb =  points_img_final(1,:) + size(Ia,2) ;
        ya = points_img_one(2,:) ;
        yb = points_img_final(2,:);

        hold on ;
        h = line([xa ; xb], [ya ; yb]) ;
        set(h,'linewidth', 1, 'color', 'g') ;



    end



   