from scipy import io

mat_file = io.loadmat('specific_person_features.mat')
mat_file2 = io.loadmat('specific_person_features - 복사본.mat')

print(mat_file)
print("-----------------")
print(mat_file2)