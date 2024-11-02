clear; clc;

% Langkah 1: Memuat data dari file Excel dengan nama variabel yang dipertahankan
data = readtable('Predict Students.xlsx', 'VariableNamingRule', 'preserve');

% Menampilkan nilai unik dalam variabel 'Target'
disp('Nilai unik dalam variabel Target:');
disp(unique(data.Target));

% Mengodekan variabel 'Target' (Dropout = 0, Graduate = 1)
data.Target = categorical(data.Target);

% Menampilkan kategori unik dalam variabel 'Target'
disp('Kategori unik dalam variabel Target:');
disp(categories(data.Target));

% Mengubah kategori menjadi nilai numerik
targetCategories = categories(data.Target);
data.Target = double(data.Target == targetCategories{2}); % Diasumsikan 'Graduate' adalah 1 dan 'Dropout' adalah 0

% Memastikan 'Target' hanya berisi nilai 0 dan 1
assert(all(ismember(data.Target, [0, 1])), 'Variabel Target harus hanya berisi nilai 0 dan 1.');

% Mengidentifikasi kolom kategori dan numerik
categoricalVars = varfun(@iscategorical, data, 'OutputFormat', 'uniform');
numericVars = ~categoricalVars;

% Mengonversi variabel kategori menjadi variabel dummy
categoricalData = data(:, categoricalVars);
numericData = data(:, numericVars);

% Mengonversi data kategori menjadi dummy variables dan menyimpan nama asli kolom
dummyVars = varfun(@(x) double(categorical(x)), categoricalData);
dummyVarNames = dummyVars.Properties.VariableNames;

% Menyimpan nama-nama kolom asli sebelum mengonversi menjadi array
colNames = [dummyVarNames, numericData.Properties.VariableNames];

% Menggabungkan data dummy dan numerik
X = [table2array(dummyVars), table2array(numericData)];

% Menghapus kolom terakhir (Target) dari prediktor
X = X(:, 1:end-1);
y = data.Target; % Variabel respons

% Menetapkan seed acak untuk reproduktibilitas
rng(1);

% Memisahkan data menjadi set pelatihan 70% dan pengujian 30%
cv = cvpartition(size(X, 1), 'HoldOut', 0.3);
XTrain = X(training(cv), :);
yTrain = y(training(cv));
XTest = X(test(cv), :);
yTest = y(test(cv));

% Langkah 2: Mendeteksi dan menghapus outlier dari data pelatihan
outliers = arrayfun(@(col) detectOutliers(XTrain(:, col)), 1:size(XTrain, 2), 'UniformOutput', false);
outliers = any(cat(2, outliers{:}), 2);
XTrain = XTrain(~outliers, :);
yTrain = yTrain(~outliers);

% Langkah 3: Memeriksa korelasi untuk menghindari overfitting
correlationMatrix = corr(XTrain);
disp('Matriks Korelasi:');
disp(correlationMatrix);

% Visualisasi matriks korelasi
figure;
heatmap(correlationMatrix, 'Title', 'Matriks Korelasi', 'Colormap', jet, 'ColorLimits', [-1 1]);

% Langkah 4: Melakukan PCA untuk reduksi dimensi pada data pelatihan
[coeff, score, latent, tsquared, explained] = pca(XTrain);

% Menampilkan varians yang dijelaskan oleh komponen utama
disp('Varians yang Dijelaskan oleh Komponen Utama:');
disp(explained);

% Memilih jumlah komponen untuk mempertahankan (misalnya, 95% dari varians)
varianceThreshold = 95;
cumulativeVariance = cumsum(explained);
numComponents = find(cumulativeVariance >= varianceThreshold, 1);

disp(['Jumlah komponen yang dipertahankan (', num2str(varianceThreshold), '% varians): ', num2str(numComponents)]);

% Mengurangi data pelatihan ke jumlah komponen yang dipilih
XTrainReduced = score(:, 1:numComponents);

% Menerapkan transformasi yang sama pada data uji
XTestReduced = (XTest - mean(XTrain)) * coeff(:, 1:numComponents);

% Model regresi logistik
mdl = fitglm(XTrainReduced, yTrain, 'Distribution', 'binomial', 'Link', 'logit');

% Menampilkan ringkasan model regresi logistik
disp(mdl);

% Memprediksi respons untuk set pengujian
yPred = predict(mdl, XTestReduced);

% Mengonversi probabilitas menjadi hasil biner
yPredBinary = yPred >= 0.5;

% Memastikan kedua yTest dan yPredBinary memiliki tipe yang sama
yTest = double(yTest);
yPredBinary = double(yPredBinary);

% Menghitung akurasi model
accuracy = mean(yPredBinary == yTest);
fprintf('Akurasi: %.2f%%\n', accuracy * 100);

% Menghitung Area Under the Curve (AUC)
[X_ROC, Y_ROC, ~, AUC] = perfcurve(yTest, yPred, 1);
fprintf('AUC: %.2f\n', AUC);

% Memplot confusion matrix
figure;
cm = confusionchart(yTest, yPredBinary);
cm.Title = 'Confusion Matrix untuk Regresi Logistik';
cm.RowSummary = 'row-normalized'; % Normalisasi per baris
cm.ColumnSummary = 'column-normalized'; % Normalisasi per kolom

% Memplot kurva ROC
figure;
plot(X_ROC, Y_ROC)
xlabel('False positive rate') 
ylabel('True positive rate')
title(['Kurva ROC (AUC = ' num2str(AUC) ')'])

% Menentukan faktor-faktor signifikan berdasarkan nilai p
coeffTable = mdl.Coefficients;
significantFactors = coeffTable(coeffTable.pValue < 0.05, :);

% Menampilkan nama faktor-faktor signifikan
disp('Faktor-faktor yang signifikan mempengaruhi kelulusan:');
significantFactorNames = colNames(coeffTable.pValue(2:end) < 0.05);
disp(significantFactorNames);

% Menampilkan faktor signifikan beserta nilai p
disp('Tabel faktor-faktor signifikan:');
disp(significantFactors);

% Mendefinisikan fungsi untuk mendeteksi outlier menggunakan metode IQR
function isOutlier = detectOutliers(data)
    Q1 = quantile(data, 0.25);
    Q3 = quantile(data, 0.75);
    IQR = Q3 - Q1;
    lowerBound = Q1 - 1.5 * IQR;
    upperBound = Q3 + 1.5 * IQR;
    isOutlier = (data < lowerBound) | (data > upperBound);
end
