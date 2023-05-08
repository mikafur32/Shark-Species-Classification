import os
import cv2
import sklearn.feature_extraction.image.extract_patches_2d as patch


def write_patches(data_path):
    for file1 in os.listdir(data_path):
        file2 = os.path.join(data_path, file1)
        os.makedir(os.path.join(file2,"patches"))
        for file3 in os.listdir(file2):
            if file3 == "unused teeth":
                continue
            else:
                simage = os.path.join(file2, file3)
                #crop_patches(simage)


def crop_patches(image_path):

    # Load the input image
    input_img = cv2.imread(image_path)

    # Set the patch size and stride
    patch_size = 448
    stride = 224

    # Initialize lists to store patches and their positions
    patches = []
    positions = []

    # Iterate over the image in patches
    for i in range(0, input_img.shape[0] - patch_size + 1, stride):
        for j in range(0, input_img.shape[1] - patch_size + 1, stride):
            # Extract the patch and resize it to 224x224 px
            patch = cv2.resize(input_img[i:i + patch_size, j:j + patch_size], (224, 224))

            # Preprocess the patch
            patch = preprocess_patch(patch)

            # Store the patch and its position
            patches.append(patch)
            positions.append((i, j))

            # Write Patch
            cv2.imwrite(os.path.join(image_path, "patches"), patches)

    # Convert the list of patches to a NumPy array
    patches = np.array(patches)



# Function to preprocess a patch
def preprocess_patch(patch):
    # Subtract the mean pixel value and scale the pixel values
    patch = patch.astype(np.float32)
    patch -= np.array([[[103.939, 116.779, 123.68]]])
    patch /= 255.0
    return patch


