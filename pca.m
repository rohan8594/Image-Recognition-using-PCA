img_files = dir('FaceRecognition_Data/ALL/*.TIF');
M = length(img_files);

for i = 1:M
    img = imread(strcat('FaceRecognition_Data/ALL/', img_files(i).name));
    img = img(:);
    X(:, i) = img;
end

% Calculating avg matrix
avg = mean(X,2);
avg_matrix = reshape(avg, 32, 32);
h1 = figure;
colormap('gray');
subplot(1, 1, 1);
imagesc(avg_matrix)
title('Average Face')

% removing avg from training matrix X
X = double(X);
norm_matrix = [];
for i = 1:M
    norm_matrix(:, i) = X(:, i) - avg;
end

% covariance
cov_matrix = norm_matrix*norm_matrix';
disp(cov_matrix)

% eigen values
[V, D] = eig(cov_matrix);
evalues = diag(D);

[~,indices] = sort(evalues,'descend');

% Selecting top K eigenvalues
K = 16;
W = zeros(32*32, K);
for i=1:K
    W(:, i) = V(:, indices(i));
end

% Testing the training set
K = 16;
testFaces = dir('FaceRecognition_Data/FA/*.TIF');
numOfTestImages = length(testFaces);
DB = zeros(K, numOfTestImages);
for i=1:numOfTestImages
    tempimg = imread(strcat('FaceRecognition_Data/FA/', testFaces(i).name));
    DB(:, i) = W' * (reshape(double(tempimg), [], 1) - avg);
end

imgName = input('Enter filename of your test image: ','s');
fullImgName = strcat('FaceRecognition_Data/FB/', imgName);
testimg = reshape(double(imread(fullImgName)), [], 1);
y = W' * (testimg - avg);
distFromTest = zeros(1, numOfTestImages);
for i=1:numOfTestImages
    distFromTest(i) = norm(y - DB(:, i));
end

[sortedDistValues, sortedDistIndices] = sort(distFromTest, 'ascend');

disp((strcat('Best Match: FaceRecognition_Data/FA/', testFaces(sortedDistIndices(1)).name)));
disp(strcat('Distance from test image: ', int2str(sortedDistValues(1))));
disp(strcat('Second Match: FaceRecognition_Data/FA/', testFaces(sortedDistIndices(2)).name));
disp(strcat('Distance from test image: ', int2str(sortedDistValues(2))));
disp(strcat('Third Match: FaceRecognition_Data/FA/', testFaces(sortedDistIndices(3)).name));
disp(strcat('Distance from test image: ', int2str(sortedDistValues(3))));

h2 = figure;
subplot(2, 2, 1);
imshow(fullImgName)
title('Entered image')
subplot(2, 2, 2);
imshow(strcat('FaceRecognition_Data/FA/', testFaces(sortedDistIndices(1)).name))
title('Best Match')
subplot(2, 2, 3);
imshow(strcat('FaceRecognition_Data/FA/', testFaces(sortedDistIndices(2)).name))
title('Second Match')
subplot(2, 2, 4);
imshow(strcat('FaceRecognition_Data/FA/', testFaces(sortedDistIndices(3)).name))
title('Third Match')