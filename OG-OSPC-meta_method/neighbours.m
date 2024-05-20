% tAKE 8 NEIGHBOURS

function X_image = neighbours(X)


num_pts=size(X,2);
X_image=[];

x_all=X(1,:);
y_all=X(2,:);

for i=1:num_pts
    
    x=x_all(i);
    y=y_all(i);
    
    mid=[x ;  y] ;
    right=[ x+1   ;     y     ];
    left= [x-1   ;   y];
    up=   [ x    ;     y-1   ];
    down= [ x    ;      y+1       ];
    up_right=[x+1 ;     y-1];
    up_left=[ x-1  ;    y-1      ];
    down_right=[x+1 ;   y+1];
    down_left= [x-1  ;    y+1] ;
    
    com=[ mid  right  left  up  down  up_right  up_left  down_right  down_left ];
    
    X_image=[ X_image  com];
    
 
    
end