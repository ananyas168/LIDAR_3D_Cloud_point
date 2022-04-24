# LIDAR_23_Cloud_point

## Point Cloud
A point cloud is a set of data points in space. The points may represent a 3D shape or object. Each point position has its set of Cartesian coordinates (X, Y, Z). Point clouds are generally produced by 3D scanners or by photogrammetry software, which measure many points on the external surfaces of objects around them. 
Point clouds are distributed spatial coordinates that can resemble a particular object or a shape when assembled. They feature important applications in domains like autonomous vehicles, terrain modeling, drone navigation, etc. However, unlike their 3D mesh counterpart, point clouds are independently distributed points in space. Thus NN network needs to ignore the permutation changes and the corresponding rigid body transformations.



![image](https://user-images.githubusercontent.com/66965350/164991079-7e1b50a7-4225-427d-8ef6-fe831aa75881.png)



This is example of how cloud points look like for an airplane.


Our goal is to develope better algorithm than the existing architecture which are capable of capturing the toplogy of such 3d space point structure and capable of doing both  better classification and segmentation. 
