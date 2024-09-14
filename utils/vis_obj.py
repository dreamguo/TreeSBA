#!/usr/bin/env python
# coding=utf-8
import argparse
import open3d as o3d


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ind_class', type=str, default=None)
    parser.add_argument('--ind_instance', type=str)
    parser.add_argument('--ply_path', type=str)
    parser.add_argument('--obj_path', type=str)
    parser.add_argument('--off_path', type=str)
    args = parser.parse_args()

    # mesh visualization
    if args.obj_path != None:
        mesh = o3d.io.read_triangle_mesh(args.obj_path)
        print(mesh)
        # for visiualization
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([1,0.7,0])
        o3d.visualization.draw_geometries([mesh])
    if args.off_path != None:
        mesh = o3d.io.read_triangle_mesh(args.off_path)
        print(mesh)
        # for visiualization
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([1,0.7,0])
        o3d.visualization.draw_geometries([mesh])

    # pcd point cloud visualization
    if args.ply_path != None:
        if args.ind_class != None:
            name = 'label' + args.ind_class + '_' + args.ind_instance
            name = name + '/' + name
            pcd = o3d.io.read_point_cloud('./' + name + '.ply')
        else:
            pcd = o3d.io.read_point_cloud(args.ply_path)
        o3d.visualization.draw_geometries([pcd])
