# Algorithm for Finding a Match of Stars Based on Two Photos
## Itamar Casspi Josef Mamo
### Overview
This algorithm allows you to find a match of stars based on two photos taken with a cell phone or Stellarium. It uses openCV library to open the images, perform grayscale conversion, and convert the image to black and white using a threshold binary function. After that, it goes through all non-black objects and extracts their properties. The algorithm then matches the images using the SIFT/ORB matching algorithm and finds the two matches that are most appropriate according to the algorithm.

### Steps
1. Open the images using the openCV library.
2. Perform grayscale conversion for both images.
3. Convert the image to black and white using a threshold binary function.
4. Go through all non-black objects and extract their properties.
5. Match the images using the SIFT/ORB matching algorithm.
6. Find the two matches that are most appropriate according to the algorithm.
7. According to the two alignments, draw a rib between the two alignments and calculate the distance between them.
8. Divide the distance between the two images and get the ratio between the images.
9. Pick a random vertex from the test image.
10. Stretch two sides to it from the two vertices and check the length of each side.
11. Go through all the vertices in the image being tested and check and stretch two edges for each such vertex and calculate the weight of the edge.
12. Check if the sides maintain the ratio. If so, this vertex exists in both images.

### Usage
1. Clone this repository to your local machine.
2. Navigate to the repository directory in your terminal.
3. Run the following command to match stars in a single image:

`python main.py <image_name>.png`

This will scan the image and output the matched stars and their positions.

4. Alternatively, if you have two images you would like to compare, run the following command:

`python main.py <image_path1>.png <image_path2>.png`

This will compare the two images and output the matched stars and their positions.

### Conclusion
This algorithm provides an easy and effective way to match stars between two images taken with a cell phone or Stellarium. By using openCV and SIFT/ORB matching algorithms, it is able to extract the properties of stars and match them between the two images
