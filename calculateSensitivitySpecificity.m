function [sensitivity, specificity, maxYoudenIndex, threshold] = calculateSensitivitySpecificity(x, x_l)
    % Input:
    % x: True signal
    % x_l: Estimated signal

    % Set a range of thresholds
    thresholds = linspace(min(x), max(x), 10);

    % Initialize variables to store results
    maxYoudenIndex = -Inf;
    threshold = NaN;
    sensitivity = NaN;
    specificity = NaN;

    for t = thresholds
        % Convert continuous values to binary using the threshold
        binary_x_l = x_l >= t;

        % Find the positive and negative classes
        positiveClass = x > 0;
        negativeClass = x == 0;

        % True Positive (TP): Estimated positive when it's actually positive
        TP = sum(binary_x_l(positiveClass) == 1);

        % True Negative (TN): Estimated negative when it's actually negative
        TN = sum(binary_x_l(negativeClass) == 0);

        % False Positive (FP): Estimated positive when it's actually negative
        FP = sum(binary_x_l(negativeClass) == 1);

        % False Negative (FN): Estimated negative when it's actually positive
        FN = sum(binary_x_l(positiveClass) == 0);

        % Calculate sensitivity (True Positive Rate)
        currentSensitivity = TP / (TP + FN);

        % Calculate specificity (True Negative Rate)
        currentSpecificity = TN / (TN + FP);

        % Calculate Youden's index (J statistic)
        currentYoudenIndex = currentSensitivity + currentSpecificity - 1;

        % Check if the current threshold gives a higher Youden's index
        if currentYoudenIndex > maxYoudenIndex
            maxYoudenIndex = currentYoudenIndex;
            threshold = t;
            sensitivity = currentSensitivity;
            specificity = currentSpecificity;
        end
    end
end