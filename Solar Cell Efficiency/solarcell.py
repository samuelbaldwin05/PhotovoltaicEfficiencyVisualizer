'''
Solar Cell Heat Map Analysis
Samuel Baldwin
'''
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans


IMAGE_FOLDER = 'Solar_Cell_Images'
FACTOR = 3

def get_filenames(dirname, ext = ".jpg"):
    ''' 
    Given a directory name, return the full path and name for every
    non directory file in the directory as a list of strings.
    '''
    filenames = []
    files = os.listdir(dirname)
    for file in files:
        if not os.path.isdir(file) and file.endswith(ext):
            filenames.append(dirname + "/" + file)
    return filenames

def read_image(image_path):
    '''
    Given an image path, read in the image, then create a dataframe
    with pixel coordinates and the coordinates given pixel color
    Everything can be factored by FACTOR, which can be changed manually,
    this is to reduce stress on computer and only iterate through some
    pixels.
    Return the dataframe, total coordinates, and width and heigth of image.
    '''
    # Get RGB values
    image = Image.open(image_path)
    width, height = image.size
    pixel_colors = []
    for y in range(int(height/FACTOR)):
        for x in range(int(width/FACTOR)):
            pixel_color = image.getpixel((FACTOR*x, FACTOR*y))
            pixel_colors.append(pixel_color)
    # Create coordinate dataframe    
    width_div = width - (width % FACTOR)  
    height_div = height - (height % FACTOR)
    y_cords = []
    for i in range(0, height_div, FACTOR):
        for j in range(0, width_div, FACTOR):
            y_cords.append(i)
            
    x_cords = []
    for i in range(0, height_div, FACTOR):
        for j in range(0, width_div, FACTOR):
            x_cords.append(j)
        
    coordinate_df = pd.DataFrame({'x': x_cords, 'y': y_cords}) 
    # Turn RGB values into dataframe and combine with coordinate df
    rgb_df = pd.DataFrame({'rgb': pixel_colors})
    merged_df = pd.merge(coordinate_df, rgb_df,
                         left_index=True, right_index=True)
    return merged_df, len(pixel_colors), width, height

'''    
def rgb_to_indexed_color(image, num_colors=256):
    image[['R', 'G', 'B']] = pd.DataFrame(image['rgb'].tolist(), index=image.index)
    rgb_df = image[['R', 'G', 'B']]
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(rgb_df)
    centroids = kmeans.cluster_centers_.astype(int)
    color_palette = [tuple(color) for color in centroids]
    #indexed_img_array = np.array([find_nearest_color(pixel, centroids) for pixel in rgb_df])
    #indexed_img_array = indexed_img_array.reshape(rgb_df.shape[:2])
    #indexed_image = Image.fromarray(indexed_img_array.astype('uint8'))
    print(centroids, color_palette)
    return #indexed_image, color_palette

def find_nearest_color(pixel, centroids):
    distances = np.sqrt(np.sum((centroids - pixel)**2, axis=1))
    nearest_color_index = np.argmin(distances)
    return centroids[nearest_color_index]
'''

def calculate_intensity(df):
    '''
    Given a dataframe with columns R, G, and B (rgb color values) convert 
    to grayscale, and calculate the grayscale intensity as well 
    as the normalized intensity values for each pixel.
    Return the original dataframe, with the grayscale intensity and 
    normalized intensity added.
    '''
    # Convert RGB to grayscale intensity
    df[['R', 'G', 'B']] = pd.DataFrame(df['rgb'].tolist(), index=df.index)
    df.drop('rgb', axis=1, inplace=True)
    grayscale_intensity = 0.299 * df['R'] + 0.587 * df['G'] + 0.114 * df['B']

    
    # Normalize intensity to range [0, 1]
    max_intensity = grayscale_intensity.max()
    min_intensity = grayscale_intensity.min()
    normalized_intensity = ((grayscale_intensity - min_intensity) /
                            (max_intensity - min_intensity))

    # Calculate additional statistics
    mean_intensity = normalized_intensity.mean()
    median_intensity = normalized_intensity.median()
    std_intensity = normalized_intensity.std()
    
    # Name dataframes and merge
    result_df = pd.concat([df, grayscale_intensity,
                           normalized_intensity], axis=1)
    result_df.columns = ['x', 'y', 'R', 'G', 'B', 'Grayscale', 'Normalized']
    return result_df

    
def find_outliers(df, column_name, percentile = 95):
    '''
    Given a dataframe and a column name, find the values in the column
    that are either in the top percentile from a given value (or automatically
    95) or bottom percentile. 
    Return a new dataframe with all of the same data from the original
    dataframe, but only for the points that are considered outliers.
    '''
    percentile = percentile / 100
    bottom_percentile = df[column_name].quantile((1-percentile))
    top_percentile = df[column_name].quantile(percentile)
    outliers_df = df[(df[column_name] < bottom_percentile) | (df[column_name] > top_percentile)]   
    return outliers_df
    
def change_pixel_color(image_path, df, image_name = 'output_image'):
    '''
    Given an image path and a dataframe of outliers and their coordinates,
    color in the outlier points to highlight them.
    Save the image to the project images folder.
    Return nothing.
    '''
    coordinates = list(zip(df['x'], df['y']))
    img = Image.open(image_path)
    # Create a drawing context
    draw = ImageDraw.Draw(img)

    color = (0, 0, 255)
    for coord in coordinates:
        draw.point(coord, fill = color)
    folder_path = "Solar_Cell_Images"
    img.save(folder_path + "/" + image_name)

def create_outlier_image(image_file):
    '''
    Combine functionality of all other functions that when given an image file,
    create a dataframe with coordinates, find intensity of the grayscale
    of the pixels in the image, and highlight the outliers by coordinate
    in a copy of the original image.
    Return the image intensity dataframe and save the new image.
    '''
    image = read_image(image_file)
    image_df = image[0]
    image_intensity = calculate_intensity(image_df)
    image_outliers = find_outliers(image_intensity, 'Grayscale', 97.5)

    image_name = image_file.split("/")[-1]
    new_image_name = "highlighted_" + image_name
    change_pixel_color(image_file, image_outliers, new_image_name)
    return image_intensity
    
   
def column_correlation(df1, df2, column_name):
    '''
    Given two dataframes that share a column, and the name of the column
    Run a correlation between the two columns in the dataframes.
    Return the correlation.
    '''
    column1 = df1[column_name]
    column2 = df2[column_name]
    correlation = column1.corr(column2)
    return correlation
    

def main():
    image_files = get_filenames(IMAGE_FOLDER)
    print(image_files)
    image1_intensity = create_outlier_image(image_files[0])
    image2_intensity = create_outlier_image(image_files[1])
    
    images_corr = column_correlation(image1_intensity, image2_intensity, 'Grayscale')
    print(images_corr)
    

          
if __name__ == "__main__":
    main()
    
    
    
