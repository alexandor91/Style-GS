#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
from einops import repeat

import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.gaussian_model import GaussianModel
from scene.gaussian_sh_model import GaussianSHModel


def generate_neural_gaussians(
    viewpoint_camera, pc: GaussianModel, visible_mask=None, is_training=False
):
    ## view frustum filtering for acceleration
    if visible_mask is None:
        visible_mask = torch.ones(
            pc.get_anchor.shape[0], dtype=torch.bool, device=pc.get_anchor.device
        )
    # 只会选择可视的体素，规避了内存爆炸
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)

        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)  # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = (
            feat[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1]
            + feat[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2]
            + feat[:, ::1, :1] * bank_weight[:, :, 2:]
        )
        feat = feat.squeeze(dim=-1)  # [n, c]

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)  # [N, c+3]
    if pc.appearance_dim > 0:
        camera_indicies = (
            torch.ones_like(
                cat_local_view[:, 0], dtype=torch.long, device=ob_dist.device
            )
            * viewpoint_camera.uid
        )
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view)  # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = neural_opacity > 0.0
    mask = mask.view(-1)

    # select opacity
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(
                torch.cat([cat_local_view_wodist, appearance], dim=1)
            )
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])  # [mask]

    # offsets
    offsets = grid_offsets.view([-1, 3])  # [mask]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, "n (c) -> (n k) (c)", k=pc.n_offsets)
    concatenated_all = torch.cat(
        [concatenated_repeated, color, scale_rot, offsets], dim=-1
    )
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split(
        [6, 3, 3, 7, 3], dim=-1
    )

    # post-process cov
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(
        scale_rot[:, :3]
    )  # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:, 3:7])

    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:, :3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot


def generate_neural_gaussians_sh(
    viewpoint_camera,
    pc: GaussianSHModel,
    visible_mask=None,
    is_training=False,
    transformer=None,  # for network
    style_img=None,
):
    # import pdb
    # pdb.set_trace()
    # import numpy as np
    # anchor = pc.get_anchor
    # grid_offsets = pc._offset
    # grid_scaling = pc.get_scaling
    # concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    # concatenated_repeated = repeat(concatenated, "n (c) -> (n k) (c)", k=pc.n_offsets)
    # concatenated_all = torch.cat([concatenated_repeated], dim=-1)
    # masked = concatenated_all
    # scaling_repeat, repeat_anchor = masked.split([6, 3], dim=-1)
    # offsets = grid_offsets.view([-1, 3])
    # offsets = offsets * scaling_repeat[:, :3]
    # xyz = repeat_anchor + offsets
    # sh=pc.get_sh_mlp(pc._anchor_feat).reshape([anchor.shape[0] * pc.n_offsets, 3 * (pc.max_sh_degree + 1) ** 2])
    # sh=sh.reshape([sh.shape[0], (pc.max_sh_degree + 1) ** 2, 3])
    # np.save("sh.npy", sh.detach().cpu().numpy())
    # np.save("xyz.npy", xyz.detach().cpu().numpy())

    ## view frustum filtering for acceleration
    if visible_mask is None:
        visible_mask = torch.ones(
            pc.get_anchor.shape[0], dtype=torch.bool, device=pc.get_anchor.device
        )

    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)

        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)  # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = (
            feat[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1]
            + feat[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2]
            + feat[:, ::1, :1] * bank_weight[:, :, 2:]
        )
        feat = feat.squeeze(dim=-1)  # [n, c]

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)  # [N, c+3]
    if pc.appearance_dim > 0:
        camera_indicies = (
            torch.ones_like(
                cat_local_view[:, 0], dtype=torch.long, device=ob_dist.device
            )
            * viewpoint_camera.uid
        )
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view)  # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # # # get offset's color
    # if pc.appearance_dim > 0:
    #     if pc.add_color_dist:
    #         color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
    #     else:
    #         color = pc.get_color_mlp(
    #             torch.cat([cat_local_view_wodist, appearance], dim=1)
    #         )
    # else:
    #     if pc.add_color_dist:
    #         color = pc.get_color_mlp(cat_local_view)
    #     else:
    #         color = pc.get_color_mlp(cat_local_view_wodist)

    # color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [mask]
    # get offset's color
    if pc.appearance_dim > 0:
        sh = pc.get_sh_mlp(torch.cat([feat, appearance], dim=1))
    else:
        sh = pc.get_sh_mlp(feat)

    # Cross attention here only stylize base color 光照阴影会保留下来（可能）
    # ca block backbone + mlp head 预测 A A*A^T; 惩罚项
    # backbone croco-style (不拼接cls token) ；layer

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)

    if transformer is not None:
        # modify
        assert style_img is not None
        sh, neural_opacity, scale_rot = transformer.predict(
            sh, neural_opacity, scale_rot, style_img
        )

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = neural_opacity > 0.0
    mask = mask.view(-1)

    # select opacity
    opacity = neural_opacity[mask]

    sh = sh.reshape(
        [anchor.shape[0] * pc.n_offsets, 3 * (pc.max_sh_degree + 1) ** 2]
    )  # [mask]

    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])  # [mask]

    # offsets
    offsets = grid_offsets.view([-1, 3])  # [mask]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, "n (c) -> (n k) (c)", k=pc.n_offsets)
    concatenated_all = torch.cat(
        [concatenated_repeated, sh, scale_rot, offsets], dim=-1
    )
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, sh, scale_rot, offsets = masked.split(
        [6, 3, 3 * (pc.max_sh_degree + 1) ** 2, 7, 3], dim=-1
    )

    # post-process cov
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(
        scale_rot[:, :3]
    )  # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:, 3:7])

    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:, :3]
    xyz = repeat_anchor + offsets

    sh = sh.reshape([sh.shape[0], (pc.max_sh_degree + 1) ** 2, 3])
    # modify sh

    if is_training:
        return xyz, sh, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, sh, opacity, scaling, rot


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    visible_mask=None,
    retain_grad=False,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training

    if is_training:
        (
            xyz,
            color,
            opacity,
            scaling,
            rot,
            neural_opacity,
            mask,
        ) = generate_neural_gaussians(
            viewpoint_camera, pc, visible_mask, is_training=is_training
        )
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(
            viewpoint_camera, pc, visible_mask, is_training=is_training
        )

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "selection_mask": mask,
            "neural_opacity": neural_opacity,
            "scaling": scaling,
        }
    else:
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }


def render_sh(
    viewpoint_camera,
    pc: GaussianSHModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    visible_mask=None,
    retain_grad=False,
    transformer=None,  # for stylization
    style_img=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training

    if is_training:
        (
            xyz,
            shs,
            opacity,
            scaling,
            rot,
            neural_opacity,
            mask,
        ) = generate_neural_gaussians_sh(
            viewpoint_camera,
            pc,
            visible_mask,
            is_training=is_training,
            transformer=transformer,
            style_img=style_img,
        )
    else:
        xyz, shs, opacity, scaling, rot = generate_neural_gaussians_sh(
            viewpoint_camera,
            pc,
            visible_mask,
            is_training=is_training,
            transformer=transformer,
            style_img=style_img,
        )

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        # sh_degree=3,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "selection_mask": mask,
            "neural_opacity": neural_opacity,
            "scaling": scaling,
        }
    else:
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }


def render_depth(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    visible_mask=None,
    retain_grad=False,
):
    is_training = pc.get_color_mlp.training

    if is_training:
        (
            xyz,
            color,
            opacity,
            scaling,
            rot,
            neural_opacity,
            mask,
        ) = generate_neural_gaussians(
            viewpoint_camera, pc, visible_mask, is_training=is_training
        )
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(
            viewpoint_camera, pc, visible_mask, is_training=is_training
        )

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # calculate depth for rendering
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)
    xyz_h_c = xyz_h @ viewpoint_camera.world_view_transform  # already transposed
    z_c = xyz_h_c[:, 2:3].repeat(1, 3)

    viewpoint_camera.world_view_transform  # w2c transform just mul
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=z_c,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "selection_mask": mask,
            "neural_opacity": neural_opacity,
            "scaling": scaling,
        }
    else:
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }


def render_depth_sh(
    viewpoint_camera,
    pc: GaussianSHModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    visible_mask=None,
    retain_grad=False,
):
    is_training = pc.get_color_mlp.training

    if is_training:
        (
            xyz,
            shs,
            opacity,
            scaling,
            rot,
            neural_opacity,
            mask,
        ) = generate_neural_gaussians_sh(
            viewpoint_camera, pc, visible_mask, is_training=is_training
        )
    else:
        xyz, shs, opacity, scaling, rot = generate_neural_gaussians_sh(
            viewpoint_camera, pc, visible_mask, is_training=is_training
        )

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # calculate depth for rendering
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)
    xyz_h_c = xyz_h @ viewpoint_camera.world_view_transform  # already transposed
    z_c = xyz_h_c[:, 2:3].repeat(1, 3)
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=z_c,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "selection_mask": mask,
            "neural_opacity": neural_opacity,
            "scaling": scaling,
        }
    else:
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }


def prefilter_voxel(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(
        means3D=means3D,
        scales=scales[:, :3],
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return radii_pure > 0
