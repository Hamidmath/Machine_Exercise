function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%



values = [0.005 0.015 0.045 0.135 0.405 1.215 3.645 10.935];
errMin = inf;

for C = values
  for sigma = values
    model = svmTrain( X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma) );
    errMain   = mean( double ( svmPredict(model, Xval) ~= yval ) );
    if( errMain <= errMin )
      CFin = C;  
      sigmaFin = sigma;
      errMin = errMain;
      fprintf('new minimum C, sigma = %f, %f by error = %f', CFin, sigmaFin, errMin)
    end
  end
end
C = CFin;
sigma = sigmaFin;



% =========================================================================

end
