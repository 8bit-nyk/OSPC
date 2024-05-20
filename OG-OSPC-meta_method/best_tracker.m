% read `N images
function [points_img_one,points_img_final]=best_tracker(N_start_img,N_final_img,N_corr,draw,draw_final)

% clc
% clear all
% % var int 
% [N_start_img,N_final_img,N_corr,draw,draw_final] = deal(1,2,1000,1,1);
%%
N_imgs= (N_final_img - N_start_img) + 1  ; %%%% hard assumption N_fin > N_start %%%%%
%%

image1_num=get_num_image(N_start_img) ;
img1=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\images\" + image1_num + ".jpg");
images=zeros([ size(img1,1)   , size(img1,2)   , N_imgs ]);
images(:,:,1)=img1;

%% fill the images in one array 
%% here fix the case where N_imges = 1

for i=2:N_imgs

image_num=get_num_image(N_start_img + i-1) ;

images(:,:,i)=imread("C:\Users\User\Desktop\AUB\spring 2022\dataset\sequence_30\images\" + image_num + ".jpg");


end
%% get matches for consecutive pictures 
for i=1:N_imgs - 1
    
[X_first,X_second] = meta_sift_corr_fn_photo(images(:,:,i),images(:,:,i+1),N_corr,draw) ;
X_img_first(:,:,i)=ceil(X_first)'; % i did transpose for better visualization 
X_img_second(:,:,i)=ceil(X_second)'; %% switch between ceil and round 

end 



% fill 
num_pts_tb_tracked=size(X_img_first(:,:,1),1);
point_tracked_value=ones( N_imgs , 2  , num_pts_tb_tracked   )*-1;
point_tracked_value(1 ,:,:)= X_img_first(:,:,1)';
point_tracked_value(2 ,:,:)= X_img_second(:,:,1)';

num_pairs=N_imgs - 1;
count=0;
real_index_list=[];
for point_f1=1:num_pts_tb_tracked
    
    for pair=1: num_pairs - 1
        
     if pair==1            
       [ tf , index ] = ismember( X_img_first(:,:,pair+1) , X_img_second(point_f1 , : , pair ) , 'rows' ); 
     else 
       [ tf , index ] = ismember( X_img_first(:,:,pair+1) , X_img_second(real_index_list(count) , : , pair ) , 'rows' );   
     end
    
    
    indication=sum(index);
    
    real_index=[];
    if indication > 0
        real_index=find(index==1);
        real_index_list=[real_index_list   real_index(1)    ];
        count=count+1;
       % plus two since pair i and pair i + 1 tell me where pt is in frame
       % i +2
        point_tracked_value (pair + 2 , : , point_f1 ) = X_img_second(real_index(1),:,pair + 1);
        %point_tracked_index(pair ,:,point_f1)=
    else 
        break % assumed break gets me out of one loop only
    end

    
    end
    
end


%% count

for i=1:N_corr
    list_count(i)=sum( point_tracked_value(:,1,i) > -1 );
end

index_full_track= find(list_count==N_imgs);


for i=1:size(index_full_track,2)
points_img_one(:,i)=point_tracked_value(1,:,index_full_track(i))';
points_img_final(:,i)=point_tracked_value(N_imgs,:,index_full_track(i))';
%% note it is smart to use track 3 frames to remove outliers
% remove exposure change to help
end 


if draw_final == 1 
    
Ia=uint8(images(:,:,1));
Ib=uint8(images(:,:,N_imgs ));

figure ; clf ;
imagesc(cat(2, Ia, Ib)) ;

xa = points_img_one(1,:);
xb =  points_img_final(1,:) + size(Ia,2) ;
ya = points_img_one(2,:) ;
yb = points_img_final(2,:);

hold on ;
h = line([xa ; xb], [ya ; yb]) ;
set(h,'linewidth', 1, 'color', 'b') ;
    
    
end 


