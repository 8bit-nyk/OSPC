% sift fn
% input two images
%number of corresponding points
% output corresponding vectors in homo form 
% X_image1 3*N 
% X_image2 3*N 
function [X_image1,X_image2,limit]=meta_sift_corr_fn_photo(I1,I2,draw)
%Ia = vl_impattern('house.000') ;
Ia=single(I1);

%Ia = rgb2gray(Ia) ;
%Ib = vl_impattern('house.001') ;
Ib=single(I2);

%{
The matrix f has a column for each frame. 
A frame is a disk of center f(1:2), 
scale f(3) and orientation f(4)
%}
[fa, da] = vl_sift(Ia);
[fb, db] = vl_sift(Ib);

f_centers_a=round(fa(1:2,:)); % rounded them
f_centers_b=round(fb(1:2,:)); % rounded them
f_centers_a=fa(1:2,:); % without round %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
f_centers_b=fb(1:2,:); % without round %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
[matches, scores] = vl_ubcmatch(da, db) ;

% sorted , index of scores from highest to lowest
%The index of the original match and the closest descriptor 
%is stored in each column of matches
[drop, perm] = sort(scores, 'descend') ;

matches = matches(:, perm) ;    %sorted the matches
scores  = scores(perm) ;        %sorted the scores (same as drop)

limit=size(matches,2);
% xa = fa(1,matches(1,limit-N_corr:limit)) ;
% xb = fb(1,matches(2,limit-N_corr:limit)) ;
% ya = fa(2,matches(1,limit-N_corr:limit));
% yb = fb(2,matches(2,limit-N_corr:limit));

xa = fa(1,matches(1,1:limit)) ;
xb = fb(1,matches(2,1:limit)) ;
ya = fa(2,matches(1,1:limit));
yb = fb(2,matches(2,1:limit));
X_image1=cat(1,xa,ya);
X_image2=cat(1,xb,yb);

Ia=I1;
Ib=I2;

if draw==1
%     figure ; 
%     imagesc(Ia) ;
% 
%     hold on
    figure
    scatter(xa,ya,20)
    hold on
    scatter(xb,yb,20)
%     plot(50,100, 'r+', 'MarkerSize', 30, 'LineWidth', 2);
%     hold on ;
%     h = line(cat(1,xa',xb')', cat(1,ya',yb')') ;
%     set(h,'linewidth', 1, 'color', 'b') ;
    hold on ;
    h = line([xa ; xb], [ya ; yb]) ;
    set(h,'linewidth', 1, 'color', 'b') ;

% figure(1) ; clf ;
% imagesc(cat(2, Ia, Ib)) ;
% axis image off ;
% vl_demo_print('sift_match_1', 1) ;

figure ; clf ;
imagesc(cat(2, Ia, Ib)) ;

xa = fa(1,matches(1,:)) ;
xb = fb(1,matches(2,:)) + size(Ia,2) ;
ya = fa(2,matches(1,:)) ;
yb = fb(2,matches(2,:)) ;

% xa = fa(1,matches(1,limit-N_corr:limit)) ;
% xb = fb(1,matches(2,limit-N_corr:limit)) + size(Ia,2) ;
% ya = fa(2,matches(1,limit-N_corr:limit)) ;
% yb = fb(2,matches(2,limit-N_corr:limit)) ;

hold on ;
h = line([xa ; xb], [ya ; yb]) ;
set(h,'linewidth', 1, 'color', 'b') ;
    


end

