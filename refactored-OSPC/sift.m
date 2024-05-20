function [X_image1, X_image2, limit] = sift(I1, I2, draw)
    % Convert images to single precision
    Ia = single(I1);
    Ib = single(I2);

    % Compute SIFT features for both images
    % fa, fb are the frames (keypoints) - each column corresponds to a keypoint
    % da, db are the descriptors - each column corresponds to a descriptor
    [fa, da] = vl_sift(Ia);
    [fb, db] = vl_sift(Ib);

    % Match features between the two images
    % matches - two rows of indices, where the first row contains indices into fa/da and the second row contains indices into fb/db
    % scores - the score of each match
    [matches, scores] = vl_ubcmatch(da, db);

    % Sort matches by score
    [drop, perm] = sort(scores, 'descend');
    matches = matches(:, perm);
    scores = scores(perm);

    % Number of matches found
    limit = size(matches, 2);

    % Extract coordinates of matched features
    xa = fa(1, matches(1, 1:limit));
    xb = fb(1, matches(2, 1:limit));
    ya = fa(2, matches(1, 1:limit));
    yb = fb(2, matches(2, 1:limit));

    % Create homogenous coordinates of matched features
    X_image1 = [xa; ya; ones(1, limit)];
    X_image2 = [xb; yb; ones(1, limit)];

    % If draw is true, plot the matched features
    if draw == 1
        figure;
        % Display the two images side by side
        imagesc(cat(2, I1, I2));

        % Adjust x coordinates of features in the second image
        xb = xb + size(I1, 2);

        hold on;
        % Draw lines connecting matched features
        line([xa; xb], [ya; yb], 'LineWidth', 1, 'Color', 'b');
    end
end
