# Construct the path to Carla API based on your system
import glob
import os
import sys
carla_api_path = glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    sys.version_info.major,
    sys.version_info.minor,
    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0]

# Add the Carla API path to sys.path
sys.path.append(carla_api_path)

import open3d as o3d
import numpy as np

def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)
vis = o3d.visualization.Visualizer()

# # Create the window
if vis.create_window(
    window_name='Carla Lidar',
    width=960,
    height=540,
    left=480,
    top=270):
    
    # Set the background color
    render_option = vis.get_render_option()
    if render_option is not None:
        render_option.background_color = [0.05, 0.05, 0.05]
        render_option.point_size = 1
        render_option.show_coordinate_frame = True
        add_open3d_axis(vis)
    
    # Continue with your visualization
    
    # Don't forget to call vis.destroy_window() when done to close the window.

else:
    print("Failed to create the Open3D window.")
# Connect the client and set up bp library and spawn point
import carla 
import math 
import random 
import time 
import cv2
from matplotlib import cm
client = carla.Client('localhost', 2000)
world = client.get_world()
bp_lib = world.get_blueprint_library() 
spawn_points = world.get_map().get_spawn_points() 

# Add vehicle
vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020') 
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(world.get_map().get_spawn_points()))

# Move spectator to view ego vehicle
spectator = world.get_spectator() 
transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation) 
spectator.set_transform(transform)

# Add traffic and set in motion with Traffic Manager
for i in range(1): 
    vehicle_bp = random.choice(bp_lib.filter('vehicle')) 
    npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))    
for v in world.get_actors().filter('*vehicle*'): 
    v.set_autopilot(True) 


# Auxilliary code for colormaps and axes
VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

COOL_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
COOL = np.array(cm.get_cmap('winter')(COOL_RANGE))
COOL = COOL[:,:3]


# get me camera
# Spawn camera and all
camera_bp = bp_lib.find('sensor.camera.rgb') 
camera_init_trans = carla.Transform(carla.Location(z=2.5, x=-3), carla.Rotation())
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

# Add navigation sensor
gnss_bp = bp_lib.find('sensor.other.gnss')
gnss_sensor = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=vehicle)

# Add inertial measurement sensor
imu_bp = bp_lib.find('sensor.other.imu')
imu_sensor = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)



# Add obstacle detector
obstacle_bp = bp_lib.find('sensor.other.obstacle')
obstacle_bp.set_attribute('hit_radius','0.5')
obstacle_bp.set_attribute('distance','50')
obstacle_sensor = world.spawn_actor(obstacle_bp, carla.Transform(), attach_to=vehicle)

# All sensor callbacks

# Setting the GPS and IMU datas
def rgb_callback(image, data_dict):
    data_dict['rgb_image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

def gnss_callback(data, data_dict):
    data_dict['gnss'] = [data.latitude, data.longitude]


## IMU Call backs
def imu_callback(data, data_dict):
    data_dict['imu'] = {
        'gyro': data.gyroscope,
        'accel': data.accelerometer,
        'compass': data.compass
    }
    
# Auxilliary geometry functions for transforming to screen coordinates
def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return tuple(map(int, point_img[0:2]))
    
world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

# Get the attributes from the camera
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()
fov = camera_bp.get_attribute("fov").as_float()

# Calculate the camera projection matrix to project from 3D -> 2D
K = build_projection_matrix(image_w, image_h, fov)
def obstacle_callback(event, data_dict, camera, k_mat):
    if 'static' not in event.other_actor.type_id:
        data_dict['obstacle'].append({'transform': event.other_actor.type_id, 'frame': event.frame})
        
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
    image_point = get_image_point(event.other_actor.get_transform().location, k_mat, world_2_camera)
    if  0 < image_point[0] < image_w and 0 < image_point[1] < image_h:
        cv2.circle(data_dict['rgb_image'], tuple(image_point), 10, (0,0,255), 3)


# LIDAR and RADAR callbacks
def lidar_callback(point_cloud, point_list):
    """Prepares a point cloud with intensity
    colors ready to be consumed by Open3D"""
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    # Isolate the intensity and compute a color for it
    intensity = data[:, -1]
    den = np.log(np.exp(-0.004 * 100))
    if(den != 0):
        intensity_col = 1.0 - np.log(intensity) / den
    else:
        intensity_col = 1
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    points = data[:, :-1]

    points[:, :1] = -points[:, :1]

    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)

def radar_callback(data, point_list):
    radar_data = np.zeros((len(data), 4))
    
    for i, detection in enumerate(data):
        x = detection.depth * math.cos(detection.altitude) * math.cos(detection.azimuth)
        y = detection.depth * math.cos(detection.altitude) * math.sin(detection.azimuth)
        z = detection.depth * math.sin(detection.altitude)
        
        radar_data[i, :] = [x, y, z, detection.velocity]
        
    intensity = np.abs(radar_data[:, -1])
    den = np.log(np.exp(-0.004 * 100))
    if(den != 0):
        intensity_col = 1.0 - np.log(intensity) / den
    else:
        intensity_col = 1
    int_color = np.c_[
        np.interp(intensity_col, COOL_RANGE, COOL[:, 0]),
        np.interp(intensity_col, COOL_RANGE, COOL[:, 1]),
        np.interp(intensity_col, COOL_RANGE, COOL[:, 2])]
    
    points = radar_data[:, :-1]
    points[:, :1] = -points[:, :1]
    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)
    
# Camera callback
def camera_callback(image, data_dict):
    data_dict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)) 


# Set up LIDAR and RADAR, parameters are to assisst visualisation

lidar_bp = bp_lib.find('sensor.lidar.ray_cast') 
lidar_bp.set_attribute('range', '100.0')
lidar_bp.set_attribute('noise_stddev', '0.1')
lidar_bp.set_attribute('upper_fov', '15.0')
lidar_bp.set_attribute('lower_fov', '-25.0')
lidar_bp.set_attribute('channels', '64.0')
lidar_bp.set_attribute('rotation_frequency', '20.0')
lidar_bp.set_attribute('points_per_second', '500000')
    
lidar_init_trans = carla.Transform(carla.Location(z=2))
lidar = world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=vehicle)

radar_bp = bp_lib.find('sensor.other.radar') 
radar_bp.set_attribute('horizontal_fov', '30.0')
radar_bp.set_attribute('vertical_fov', '30.0')
radar_bp.set_attribute('points_per_second', '10000')
radar_init_trans = carla.Transform(carla.Location(z=2))
radar = world.spawn_actor(radar_bp, radar_init_trans, attach_to=vehicle)



# Add auxilliary data structures
point_list = o3d.geometry.PointCloud()
radar_list = o3d.geometry.PointCloud()

# Set up dictionary for camera data
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()
camera_data = {'image': np.zeros((image_h, image_w, 4))} 

# Start sensors
lidar.listen(lambda data: lidar_callback(data, point_list))
radar.listen(lambda data: radar_callback(data, radar_list))

# Initialise data
collision_counter = 20
lane_invasion_counter = 20
sensor_data = {'rgb_image': np.zeros((image_h, image_w, 4)),
               'collision': False,
               'lane_invasion': False,
               'gnss': [0,0],
               'obstacle': [],
               'imu': {
                    'gyro': carla.Vector3D(),
                    'accel': carla.Vector3D(),
                    'compass': 0
                }}

# OpenCV window with initial data
cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
cv2.imshow('RGB Camera', sensor_data['rgb_image'])
cv2.waitKey(1)

# Start sensors recording data

camera.listen(lambda image: rgb_callback(image, sensor_data))
gnss_sensor.listen(lambda event: gnss_callback(event, sensor_data))
imu_sensor.listen(lambda event: imu_callback(event, sensor_data))
obstacle_sensor.listen(lambda event: obstacle_callback(event, sensor_data, camera, K))

# Some parameters for text on screen
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,50)
fontScale              = 0.5
fontColor              = (255,255,255)
thickness              = 2
lineType               = 2

frame = 0
while True:
    if frame == 2:
        vis.add_geometry(point_list)
        vis.add_geometry(radar_list)
    vis.update_geometry(point_list)
    vis.update_geometry(radar_list)
    
    vis.poll_events()
    vis.update_renderer()
    # # This can fix Open3D jittering issues:
    time.sleep(0.005)
    frame += 1
    # cv2.imshow('RGB Camera', camera_data['image'])
    # Latitude from GNSS sensor
    cv2.putText(sensor_data['rgb_image'], 'Lat: ' + str(sensor_data['gnss'][0]), 
    (10,30), 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    
    # Longitude from GNSS sensor
    cv2.putText(sensor_data['rgb_image'], 'Long: ' + str(sensor_data['gnss'][1]), 
    (10,50), 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    
    # Calculate acceleration vector minus gravity
    accel = sensor_data['imu']['accel'] - carla.Vector3D(x=0,y=0,z=9.81)
    
    # Display acceleration magnitude
    cv2.putText(sensor_data['rgb_image'], 'Accel: ' + str(accel.length()), 
    (10,70), 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    
    # Gyroscope output
    cv2.putText(sensor_data['rgb_image'], 'Gyro: ' + str(sensor_data['imu']['gyro'].length()), 
    (10,100), 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)    
    if(frame%10):
        cv2.imshow('RGB Camera', sensor_data['rgb_image'])

    # Break if user presses 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Close displayws and stop sensors
cv2.destroyAllWindows()
radar.stop()
radar.destroy()
lidar.stop()
lidar.destroy()
camera.stop()
camera.destroy()
camera.stop()
gnss_sensor.stop()
imu_sensor.stop()
obstacle_sensor.stop()
vis.destroy_window()

for actor in world.get_actors().filter('*vehicle*'):
    actor.destroy()
for actor in world.get_actors().filter('*sensor*'):
    actor.destroy()