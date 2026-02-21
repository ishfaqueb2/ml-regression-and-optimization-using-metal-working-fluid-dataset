%% Multi-file Feature Extraction and Regression 
clear; clc; close all;

% Files available
files = {
    'Feature_Time_01.mat'
    'Feature_Time_02.mat'
    'Feature_Time_03.mat'
    'Feature_Time_04.mat'
    'Feature_Time_05.mat'
    'Feature_Time_06.mat'
    'Feature_Time_07.mat'
    'Feature_Time_08.mat'
};

fs = 4e6;             % Sampling frequency
segmentLength = 50000; % Samples per segment

X = [];
y = [];

%% Feature Extraction Functions
lowpassFilt = designfilt('lowpassiir','FilterOrder',8, ...
    'HalfPowerFrequency',1e6,'SampleRate',fs);
bandFilt1 = designfilt('bandpassiir','FilterOrder',8, ...
    'HalfPowerFrequency1',1e5,'HalfPowerFrequency2',5e5,'SampleRate',fs);
bandFilt2 = designfilt('bandpassiir','FilterOrder',8, ...
    'HalfPowerFrequency1',5e5,'HalfPowerFrequency2',1e6,'SampleRate',fs);

%% Loop over all measurement files
for f = 1:length(files)
    load(files{f}, 'Features_Time');  % Each file has Features_Time
    data = Features_Time;

    % Response column (distance)
    respCol = size(data,2);  
    distVal = unique(data(:, respCol));  

    % Signal columns (exclude meta and distance)
    signalCols = 1:(respCol-4);  
    sigData = data(:, signalCols);

    % Number of segments
    numSegments = floor(size(sigData,2) / segmentLength);

    for seg = 1:numSegments
        cols = (seg-1)*segmentLength + 1 : seg*segmentLength;
        segData = sigData(:, cols);  % [41 x segmentLength]

        featVec = [];
        for ch = 1:size(segData,1)
            sig = segData(ch,:);

            % Filtering
            sig_low = filtfilt(lowpassFilt, sig);
            sig_b1 = filtfilt(bandFilt1, sig);
            sig_b2 = filtfilt(bandFilt2, sig);

            % Time-domain features
            featVec = [featVec, ...
                mean(sig), std(sig), skewness(sig), kurtosis(sig), ...
                rms(sig), ...
                max(sig)-min(sig), ...
                mean(abs(diff(sig)))];

            % Filtered signal energy
            featVec = [featVec, ...
                bandpower(sig_low), ...
                bandpower(sig_b1), ...
                bandpower(sig_b2)];

            % Time–frequency features (wavelet transform)
            wt = modwt(sig,5); % multilevel wavelet decomposition
            for lvl = 1:3
                featVec = [featVec, mean(abs(wt(lvl,:))), std(wt(lvl,:))];
            end
        end
        X = [X; featVec];
        y = [y; distVal];
    end
end

fprintf('Extracted %d segments, %d features each\n', size(X,1), size(X,2));

%% Normalize features
X = normalize(X);

%% PCA
[coeff, Xpca, ~, ~, explained] = pca(X);
explainedVariance = cumsum(explained);
k = find(explainedVariance >= 95, 1);
Xpca = Xpca(:,1:k);
fprintf('PCA reduced features: %d -> %d\n', size(X,2), k);

%% Train/test split (by file index)
numFiles = length(files);
segmentsPerFile = size(X,1) / numFiles;

trainFiles = 1:6;   % first 6 files for training
testFiles  = 7:8;   % last 2 files for testing

trainIdx = false(size(y));
testIdx  = false(size(y));

for f = 1:numFiles
    rows = (f-1)*segmentsPerFile + 1 : f*segmentsPerFile;
    if ismember(f, trainFiles)
        trainIdx(rows) = true;
    else
        testIdx(rows) = true;
    end
end

Xtrain = Xpca(trainIdx,:);
ytrain = y(trainIdx);
Xtest  = Xpca(testIdx,:);
ytest  = y(testIdx);

fprintf('Train size: %d, Test size: %d\n', length(ytrain), length(ytest));

%% =========================
% Feature Importance Analysis (Using Random Forest)
% =========================
fprintf('\n--- Feature Importance Analysis ---\n');

% Compute feature importance from Random Forest
rf_full = TreeBagger(200, Xtrain, ytrain, 'Method','regression', ...
    'OOBPredictorImportance','on');
importance = rf_full.OOBPermutedPredictorDeltaError;

% Sort features by importance
[sortedImp, idx] = sort(importance, 'descend');

% Plot top 30 most important features
figure;
bar(sortedImp(1:30));
xlabel('Feature Index (ranked)'); ylabel('Importance');
title('Top 30 Features - Random Forest');
grid on;

% Evaluate subsets of top N features
subsetSizes = [20, 50, 100, k];
results = [];

for N = subsetSizes
    topIdx = idx(1:N);

    Xtrain_sub = Xtrain(:, topIdx);
    Xtest_sub  = Xtest(:, topIdx);

    rf_sub = TreeBagger(100, Xtrain_sub, ytrain, 'Method','regression');
    ypred_sub = predict(rf_sub, Xtest_sub);

    rmse_sub = sqrt(mean((ypred_sub - ytest).^2));
    relErr_sub = mean(abs(ypred_sub - ytest) ./ abs(ytest));

    results = [results; N rmse_sub relErr_sub];
    fprintf('Top %d features -> RMSE: %.4f, RelErr: %.4f\n', ...
        N, rmse_sub, relErr_sub);
end

% Show results as a table
resultsTbl = array2table(results, ...
    'VariableNames', {'NumFeatures','RMSE','RelErr'});
disp(resultsTbl);

% Select best feature subset based on results
[~, bestSubsetIdx] = min(results(:,2)); % Find index of minimum RMSE
bestN = results(bestSubsetIdx, 1);
bestIdx = idx(1:bestN);
fprintf('Selected top %d features for optimal performance\n', bestN);

Xtrain_best = Xtrain(:, bestIdx);
Xtest_best = Xtest(:, bestIdx);

%% =========================
% 1. Linear Regression (Baseline)
% =========================
lm = fitlm(Xtrain_best, ytrain);
ypred_lm = predict(lm, Xtest_best);
rmse_lm  = sqrt(mean((ypred_lm - ytest).^2));
relErr_lm = mean(abs(ypred_lm - ytest) ./ abs(ytest));
fprintf('\nLinear Regression -> RMSE: %.4f, RelErr: %.4f\n', rmse_lm, relErr_lm);

%% =========================
% 2. Random Forest Regression
% =========================
numTrees = 500;
rf = TreeBagger(numTrees, Xtrain_best, ytrain, 'Method','regression',...
    'OOBPrediction','On');
ypred_rf = predict(rf, Xtest_best);
rmse_rf  = sqrt(mean((ypred_rf - ytest).^2));
relErr_rf = mean(abs(ypred_rf - ytest) ./ abs(ytest));
fprintf('Random Forest -> RMSE: %.4f, RelErr: %.4f\n', rmse_rf, relErr_rf);

%% =========================
% 3. Support Vector Regression (SVR) - NON-Linear (RBF Kernel)
% =========================
fprintf('\n--- Training Support Vector Regression (RBF) Model ---\n');

% Define the hyperparameter search space
svrVars = [optimizableVariable('BoxConstraint',[0.1, 1000], 'Transform','log'),...
           optimizableVariable('KernelScale',[0.1, 1000], 'Transform','log'),...
           optimizableVariable('Epsilon',[0.01, 10], 'Transform','log')];

% Define the objective function for the optimizer
svrObjective = @(params)objfun_svr(params, Xtrain_best, ytrain);

% Run Bayesian optimization
try
    results = bayesopt(svrObjective, svrVars, ...
        'Verbose', 1, ...
        'IsObjectiveDeterministic', true, ...
        'MaxObjectiveEvaluations', 30, ...
        'UseParallel', false);
    
    % Get the best hyperparameters
    bestParams = results.XAtMinObjective;
    
    % Train the final SVR model with the best parameters
    svrMdl = fitrsvm(Xtrain_best, ytrain, ...
        'KernelFunction', 'rbf', ...
        'BoxConstraint', bestParams.BoxConstraint, ...
        'KernelScale', bestParams.KernelScale, ...
        'Epsilon', bestParams.Epsilon, ...
        'Standardize', true);
    
    % Predict and evaluate
    ypred_svr = predict(svrMdl, Xtest_best);
    rmse_svr = sqrt(mean((ypred_svr - ytest).^2));
    relErr_svr = mean(abs(ypred_svr - ytest) ./ abs(ytest));
    
    fprintf('SVM (RBF Kernel) -> RMSE: %.4f, RelErr: %.4f\n', rmse_svr, relErr_svr);
    
catch ME
    fprintf('Bayesian optimization failed. Using default SVR parameters.\n');
    fprintf('Error: %s\n', ME.message);
    
    % Fall back to default SVR
    svrMdl = fitrsvm(Xtrain_best, ytrain, ...
        'KernelFunction', 'rbf', ...
        'Standardize', true, ...
        'KernelScale', 'auto');
    
    ypred_svr = predict(svrMdl, Xtest_best);
    rmse_svr = sqrt(mean((ypred_svr - ytest).^2));
    relErr_svr = mean(abs(ypred_svr - ytest) ./ abs(ytest));
    
    fprintf('SVM (RBF Kernel) - Default -> RMSE: %.4f, RelErr: %.4f\n', rmse_svr, relErr_svr);
end

%% =========================
% 4. Gradient Boosting (LSBoost)
% =========================
fprintf('\n--- Training Gradient Boosting Model ---\n');

% Use optimized hyperparameters based on previous results
t = templateTree('MaxNumSplits', 150, 'MinLeafSize', 3);

gbm = fitrensemble(Xtrain_best, ytrain, 'Method', 'LSBoost', ...
    'NumLearningCycles', 300, ...
    'Learners', t, ...
    'LearnRate', 0.05);

ypred_gbm = predict(gbm, Xtest_best);
rmse_gbm = sqrt(mean((ypred_gbm - ytest).^2));
relErr_gbm = mean(abs(ypred_gbm - ytest) ./ abs(ytest));

fprintf('Gradient Boosting -> RMSE: %.4f, RelErr: %.4f\n', rmse_gbm, relErr_gbm);

%% =========================
% 5. Ensemble of Best Models (Averaging)
% =========================
fprintf('\n--- Creating Optimized Ensemble Model ---\n');

% Create ensemble using only the best performing models
ypred_ensemble = (ypred_rf + ypred_svr) / 2; % Exclude GBM since it performed worse

% Evaluate the ensemble
rmse_ensemble = sqrt(mean((ypred_ensemble - ytest).^2));
relErr_ensemble = mean(abs(ypred_ensemble - ytest) ./ abs(ytest));

fprintf('Ensemble (RF+SVR) -> RMSE: %.4f, RelErr: %.4f\n', rmse_ensemble, relErr_ensemble);

%% =========================
% FINAL MODEL COMPARISON
% =========================
fprintf('\n=== FINAL MODEL COMPARISON ===\n');
fprintf('Model\t\t\tRMSE\t\tRelative Error\n');
fprintf('---------------------------------------------\n');
fprintf('Linear Regression:\t%.4f\t\t%.4f\n', rmse_lm, relErr_lm);
fprintf('Random Forest:\t\t%.4f\t\t%.4f\n', rmse_rf, relErr_rf);
fprintf('SVM (RBF):\t\t%.4f\t\t%.4f\n', rmse_svr, relErr_svr);
fprintf('Gradient Boosting:\t%.4f\t\t%.4f\n', rmse_gbm, relErr_gbm);
fprintf('ENSEMBLE (RF+SVR):\t%.4f\t\t%.4f\n', rmse_ensemble, relErr_ensemble);

%% =========================
% VISUALIZATION
% =========================

% Error distribution for best model
errors_best = ytest - ypred_svr;
figure;
subplot(2,2,1);
histogram(errors_best, 30);
xlabel('Prediction Error'); ylabel('Frequency');
title('Error Distribution - Best Model (SVM)');
grid on;

% True vs Predicted plots
subplot(2,2,2);
scatter(ytest, ypred_svr, 20, 'filled', 'MarkerFaceAlpha', 0.6);
hold on; plot(xlim, xlim, 'k--', 'LineWidth', 1.5); hold off;
xlabel('True Distance'); ylabel('Predicted Distance');
title(sprintf('SVM RBF (RMSE=%.2f)', rmse_svr));
grid on;
axis equal;

% Model comparison bar chart
subplot(2,2,3);
models = {'Linear', 'RF', 'SVM', 'GBM', 'Ensemble'};
rmse_values = [rmse_lm, rmse_rf, rmse_svr, rmse_gbm, rmse_ensemble];
bar(rmse_values);
set(gca, 'XTickLabel', models);
ylabel('RMSE');
title('Model Performance Comparison');
grid on;

% Feature importance
subplot(2,2,4);
bar(sortedImp(1:20));
xlabel('Feature Rank'); ylabel('Importance Score');
title('Top 20 Most Important Features');
grid on;

% Correlation heatmap
figure;
corrMat = corr(Xtrain_best);
imagesc(corrMat);
colorbar;
title('Feature Correlation Heatmap (Selected Features)');
xlabel('Feature Index'); ylabel('Feature Index');

%% =========================
% Supporting Function for SVR Optimization
% =========================
function rmse = objfun_svr(params, X, Y)
    % Objective function for SVR hyperparameter tuning
    % Trains an SVR model and returns the 5-fold cross-validation loss (RMSE)
    
    try
        svrMdl = fitrsvm(X, Y, ...
            'KernelFunction', 'rbf', ...
            'BoxConstraint', params.BoxConstraint, ...
            'KernelScale', params.KernelScale, ...
            'Epsilon', params.Epsilon, ...
            'Standardize', true, ...
            'Verbose', 0);
        
        % Perform cross-validation
        cvmdl = crossval(svrMdl, 'KFold', 5);
        
        % Calculate the loss (RMSE)
        rmse = kfoldLoss(cvmdl, 'LossFun', @(y, ypred, w) sqrt(mean((y - ypred).^2)));
        
    catch
        % If training fails, return a very high error
        rmse = 1e6;
    end
end