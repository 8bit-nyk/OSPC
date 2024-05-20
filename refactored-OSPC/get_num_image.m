function [numImage] = get_num_image(nb)

    %nb_digits = max(ceil(log10(abs(nb))),1);
    
    nb_digits = 1;
    temp = nb;
    while temp > 1
      temp = temp / 10;
      if temp >= 1
          nb_digits = nb_digits + 1;
      end
    end

%     if nb_digits == 1
%         numImage = "0000" + num2str(nb);
%     elseif nb_digits == 2
%         numImage = "000" + num2str(nb);
%     elseif nb_digits == 3
%         numImage = "00" + num2str(nb);
%     elseif nb_digits == 4
%         numImage = "0" + num2str(nb);
%     end
    if nb_digits == 1
        numImage = "0000" + num2str(nb);
    elseif nb_digits == 2
        numImage = "000" + num2str(nb);
    elseif nb_digits == 3
        numImage = "00" + num2str(nb);
    elseif nb_digits == 4
        numImage = "0" + num2str(nb);

    end
                  
end