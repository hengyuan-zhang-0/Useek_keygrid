import os
import open3d
import open3d as o3d
import seaborn as sns
import numpy as np
import pyvista as pv
import random


def save_kp_and_pc_in_pcd(pc, kp, output_dir, save=True, name=""):
    '''

    Parameters
    ----------
    points      point cloud  [2048, 3]
    kp          estimated key-points  [10, 3]
    both        if plot both or just the point clouds

    Returns     show the key-points/point cloud
    -------

    '''

    palette_PC = sns.color_palette()
    palette = sns.color_palette("bright")
    palette_dark = sns.color_palette("dark")

    pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
    pcd.translate(pc[0])
    pcd.paint_uniform_color(palette_PC[7])

    ''' Add points in the original point cloud'''
    for i in range(len(pc)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.008) ## 0.005
        point.translate(pc[i])
        point.paint_uniform_color(palette_PC[7])
        pcd += point

    ''' Add Keypoitnts '''
    for i in range(0, len(kp)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.085) # ablation: 0.035, figures: 0.050
        point.translate(kp[i])
        if i==7:
            point.paint_uniform_color(palette_dark[7])
        else:
            point.paint_uniform_color(palette[i])
        pcd += point

    if save:
        if not os.path.exists(output_dir+'/ply'):
            os.makedirs(output_dir+'/ply')
        o3d.io.write_triangle_mesh("{}/{}.ply".format(output_dir+'/ply', name), pcd)
        if not os.path.exists(output_dir+'/png'):
            os.makedirs(output_dir+'/png')
        cloud = pv.read("{}/{}.ply".format(output_dir+'/ply', name))
        colors = cloud.point_data['RGB'] / 255.0 
        cloud.point_data['RGB'] = colors
        
        plotter = pv.Plotter(off_screen=True)
        plotter.camera_parallel_projection = True
        plotter.camera.SetParallelProjection(True)
        
        plotter.add_mesh(cloud, rgb=True)     
        plotter.camera_position = [1, -1, -1]
        camera = plotter.camera
        camera.SetViewUp(0,1,0)
        plotter.screenshot(r"{}/{}.png".format(output_dir+'/png', name))
        plotter.view_xy()
    else:
        o3d.visualization.draw_geometries([pcd])
    

def save_pc_in_pcd(pc, output_dir, save=True, name=""):
    '''

    Parameters
    ----------
    points      point cloud  [2048, 3]
    kp          estimated key-points  [10, 3]
    both        if plot both or just the point clouds

    Returns     show the key-points/point cloud
    -------

    '''

    palette_PC = sns.color_palette()
    palette = sns.color_palette("bright")
    palette_dark = sns.color_palette("dark")

    pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
    pcd.translate(pc[0])
    pcd.paint_uniform_color(palette_PC[2])

    ''' Add points in the original point cloud'''
    for i in range(len(pc)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.008) ## 0.005
        point.translate(pc[i])
        point.paint_uniform_color(palette_PC[2])
        pcd += point

    if save:
        if not os.path.exists(output_dir+'/ply'):
            os.makedirs(output_dir+'/ply')
        o3d.io.write_triangle_mesh("{}/{}.ply".format(output_dir+'/ply', name), pcd)
        if not os.path.exists(output_dir+'/png'):
            os.makedirs(output_dir+'/png')
        cloud = pv.read("{}/{}.ply".format(output_dir+'/ply', name))
        colors = cloud.point_data['RGB'] / 255.0 
        cloud.point_data['RGB'] = colors
        
        plotter = pv.Plotter(off_screen=True)
        plotter.camera_parallel_projection = True
        plotter.camera.SetParallelProjection(True)
        
        plotter.add_mesh(cloud, rgb=True)     
        plotter.camera_position = [1, -1, -1]
        camera = plotter.camera
        camera.SetViewUp(0,1,0)
        plotter.screenshot(r"{}/{}.png".format(output_dir+'/png', name))
        plotter.view_xy()
    else:
        o3d.visualization.draw_geometries([pcd])

def save_all_pc_in_pcd(pc1, pc2, output_dir, save=True, name=""):
    '''
    Parameters
    ----------
    pc1         First point cloud  [N1, 3]
    pc2         Second point cloud [N2, 3]
    output_dir  Directory to save the output files
    save        Whether to save the output files
    name        Name of the output files

    Returns     Show the combined point cloud with different colors
    -------
    '''

    palette_PC = sns.color_palette()
    palette = sns.color_palette("bright")
    palette_dark = sns.color_palette("dark")

    # Create point clouds for pc1 and pc2 using spheres
    pcd1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
    pcd1.translate(pc1[0])
    pcd1.paint_uniform_color(palette_PC[7])  # Color for the first point cloud

    for i in range(1, len(pc1)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        point.translate(pc1[i])
        point.paint_uniform_color(palette_PC[7])
        pcd1 += point

    # pcd2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
    # pcd2.translate(pc2[0])
    # pcd2.paint_uniform_color(palette_PC[2])  # Different color for the second point cloud

    for i in range(len(pc2)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        point.translate(pc2[i])
        point.paint_uniform_color(palette_PC[2])
        pcd1 += point

    # Combine the two point clouds into one
    combined_pcd = pcd1 

    if save:
        if not os.path.exists(output_dir+'/ply'):
            os.makedirs(output_dir+'/ply')
        o3d.io.write_triangle_mesh("{}/{}.ply".format(output_dir+'/ply', name), combined_pcd)
        if not os.path.exists(output_dir+'/png'):
            os.makedirs(output_dir+'/png')
        cloud = pv.read("{}/{}.ply".format(output_dir+'/ply', name))
        colors = cloud.point_data['RGB'] / 255.0 
        cloud.point_data['RGB'] = colors
        
        plotter = pv.Plotter(off_screen=True)
        plotter.camera_parallel_projection = True
        plotter.camera.SetParallelProjection(True)
        
        plotter.add_mesh(cloud, rgb=True)     
        plotter.camera_position = [1, -1, -1]
        camera = plotter.camera
        camera.SetViewUp(0,1,0)
        plotter.screenshot(r"{}/{}.png".format(output_dir+'/png', name))
        plotter.view_xy()
    else:
        o3d.visualization.draw_geometries([combined_pcd])


def save_all_pc_kp_in_pcd(pc1, pc2, kp, output_dir, save=True, name=""):
    '''
    Parameters
    ----------
    pc1         First point cloud  [N1, 3]
    pc2         Second point cloud [N2, 3]
    output_dir  Directory to save the output files
    save        Whether to save the output files
    name        Name of the output files

    Returns     Show the combined point cloud with different colors
    -------
    '''

    palette_PC = sns.color_palette()
    palette = sns.color_palette("bright")
    palette_dark = sns.color_palette("dark")

    # Initialize with first point of pc1
    combined_pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
    combined_pcd.translate(pc1[0])
    combined_pcd.paint_uniform_color(palette_PC[7])

    # Add remaining points from pc1
    for i in range(1, len(pc1)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        point.translate(pc1[i])
        point.paint_uniform_color(palette_PC[7])
        combined_pcd += point

    # Add points from pc2
    for i in range(len(pc2)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        point.translate(pc2[i])
        point.paint_uniform_color(palette_PC[2])
        combined_pcd += point

    ''' Add Keypoints '''
    for i in range(len(kp)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.085) # ablation: 0.035, figures: 0.050
        point.translate(kp[i])
        if i==7:
            point.paint_uniform_color(palette_dark[7])
        else:
            point.paint_uniform_color(palette[i])
        combined_pcd += point

    if save:
        if not os.path.exists(output_dir+'/ply'):
            os.makedirs(output_dir+'/ply')
        o3d.io.write_triangle_mesh("{}/{}.ply".format(output_dir+'/ply', name), combined_pcd)
        if not os.path.exists(output_dir+'/png'):
            os.makedirs(output_dir+'/png')
        cloud = pv.read("{}/{}.ply".format(output_dir+'/ply', name))
        colors = cloud.point_data['RGB'] / 255.0 
        cloud.point_data['RGB'] = colors
        
        plotter = pv.Plotter(off_screen=True)
        plotter.camera_parallel_projection = True
        plotter.camera.SetParallelProjection(True)
        
        plotter.add_mesh(cloud, rgb=True)     
        plotter.camera_position = [1, -1, -1]
        camera = plotter.camera
        camera.SetViewUp(0,1,0)
        plotter.screenshot(r"{}/{}.png".format(output_dir+'/png', name))
        plotter.view_xy()
    else:
        o3d.visualization.draw_geometries([combined_pcd])
