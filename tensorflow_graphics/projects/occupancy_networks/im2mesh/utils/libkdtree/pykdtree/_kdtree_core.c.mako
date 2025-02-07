/*
pykdtree, Fast kd-tree implementation with OpenMP-enabled queries

Copyright (C) 2013 - present  Esben S. Nielsen

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free
Software Foundation, either version 3 of the License, or
 (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
details.

You should have received a copy of the GNU Lesser General Public License along
with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
This kd-tree implementation is based on the scipy.spatial.cKDTree by
Anne M. Archibald and libANN by David M. Mount and Sunil Arya.
*/


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>

#define PA(i,d)			(pa[no_dims * pidx[i] + d])
#define PASWAP(a,b) { uint32_t tmp = pidx[a]; pidx[a] = pidx[b]; pidx[b] = tmp; }

#ifdef _MSC_VER
#define restrict __restrict
#endif

% for DTYPE in ['float', 'double']:

typedef struct
{
    ${DTYPE} cut_val;
    int8_t cut_dim;
    uint32_t start_idx;
    uint32_t n;
    ${DTYPE} cut_bounds_lv;
    ${DTYPE} cut_bounds_hv;
    struct Node_${DTYPE} *left_child;
    struct Node_${DTYPE} *right_child;
} Node_${DTYPE};

typedef struct
{
    ${DTYPE} *bbox;
    int8_t no_dims;
    uint32_t *pidx;
    struct Node_${DTYPE} *root;
} Tree_${DTYPE};

% endfor

% for DTYPE in ['float', 'double']:

void insert_point_${DTYPE}(uint32_t *closest_idx, ${DTYPE} *closest_dist, uint32_t pidx, ${DTYPE} cur_dist, uint32_t k);
void get_bounding_box_${DTYPE}(${DTYPE} *pa, uint32_t *pidx, int8_t no_dims, uint32_t n, ${DTYPE} *bbox);
int partition_${DTYPE}(${DTYPE} *pa, uint32_t *pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, ${DTYPE} *bbox, int8_t *cut_dim,
              ${DTYPE} *cut_val, uint32_t *n_lo);
Tree_${DTYPE}* construct_tree_${DTYPE}(${DTYPE} *pa, int8_t no_dims, uint32_t n, uint32_t bsp);
Node_${DTYPE}* construct_subtree_${DTYPE}(${DTYPE} *pa, uint32_t *pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, uint32_t bsp, ${DTYPE} *bbox);
Node_${DTYPE} * create_node_${DTYPE}(uint32_t start_idx, uint32_t n, int is_leaf);
void delete_subtree_${DTYPE}(Node_${DTYPE} *root);
void delete_tree_${DTYPE}(Tree_${DTYPE} *tree);
void print_tree_${DTYPE}(Node_${DTYPE} *root, int level);
${DTYPE} calc_dist_${DTYPE}(${DTYPE} *point1_coord, ${DTYPE} *point2_coord, int8_t no_dims);
${DTYPE} get_cube_offset_${DTYPE}(int8_t dim, ${DTYPE} *point_coord, ${DTYPE} *bbox);
${DTYPE} get_min_dist_${DTYPE}(${DTYPE} *point_coord, int8_t no_dims, ${DTYPE} *bbox);
void search_leaf_${DTYPE}(${DTYPE} *restrict pa, uint32_t *restrict pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, ${DTYPE} *restrict point_coord,
                 uint32_t k, uint32_t *restrict closest_idx, ${DTYPE} *restrict closest_dist);
void search_leaf_${DTYPE}_mask(${DTYPE} *restrict pa, uint32_t *restrict pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, ${DTYPE} *restrict point_coord,
                 uint32_t k, uint8_t *restrict mask, uint32_t *restrict closest_idx, ${DTYPE} *restrict closest_dist);
void search_splitnode_${DTYPE}(Node_${DTYPE} *root, ${DTYPE} *pa, uint32_t *pidx, int8_t no_dims, ${DTYPE} *point_coord,
                      ${DTYPE} min_dist, uint32_t k, ${DTYPE} distance_upper_bound, ${DTYPE} eps_fac, uint8_t *mask, uint32_t *  closest_idx, ${DTYPE} *closest_dist);
void search_tree_${DTYPE}(Tree_${DTYPE} *tree, ${DTYPE} *pa, ${DTYPE} *point_coords,
                 uint32_t num_points, uint32_t k,  ${DTYPE} distance_upper_bound,
                 ${DTYPE} eps, uint8_t *mask, uint32_t *closest_idxs, ${DTYPE} *closest_dists);

% endfor

% for DTYPE in ['float', 'double']:

/************************************************
Insert point into priority queue
Params:
    closest_idx : index queue
    closest_dist : distance queue
    pidx : permutation index of data points
    cur_dist : distance to point inserted
    k : number of neighbours
************************************************/
void insert_point_${DTYPE}(uint32_t *closest_idx, ${DTYPE} *closest_dist, uint32_t pidx, ${DTYPE} cur_dist, uint32_t k)
{
    int i;
    for (i = k - 1; i > 0; i--)
    {
        if (closest_dist[i - 1] > cur_dist)
        {
            closest_dist[i] = closest_dist[i - 1];
            closest_idx[i] = closest_idx[i - 1];
        }
        else
        {
            break;
        }
    }
    closest_idx[i] = pidx;
    closest_dist[i] = cur_dist;
}

/************************************************
Get the bounding box of a set of points
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims: number of dimensions
    n : number of points
    bbox : bounding box (return)
************************************************/
void get_bounding_box_${DTYPE}(${DTYPE} *pa, uint32_t *pidx, int8_t no_dims, uint32_t n, ${DTYPE} *bbox)
{
    ${DTYPE} cur;
    int8_t bbox_idx, i, j;
    uint32_t i2;

    /* Use first data point to initialize */
    for (i = 0; i < no_dims; i++)
    {
        bbox[2 * i] = bbox[2 * i + 1] = PA(0, i);
    }

    /* Update using rest of data points */
    for (i2 = 1; i2 < n; i2++)
    {
        for (j = 0; j < no_dims; j++)
        {
            bbox_idx = 2 * j;
            cur = PA(i2, j);
            if (cur < bbox[bbox_idx])
            {
                bbox[bbox_idx] = cur;
            }
            else if (cur > bbox[bbox_idx + 1])
            {
                bbox[bbox_idx + 1] = cur;
            }
        }
    }
}

/************************************************
Partition a range of data points by manipulation the permutation index.
The sliding midpoint rule is used for the partitioning.
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims: number of dimensions
    start_idx : index of first data point to use
    n :  number of data points
    bbox : bounding box of data points
    cut_dim : dimension used for partition (return)
    cut_val : value of cutting point (return)
    n_lo : number of point below cutting plane (return)
************************************************/
int partition_${DTYPE}(${DTYPE} *pa, uint32_t *pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, ${DTYPE} *bbox, int8_t *cut_dim, ${DTYPE} *cut_val, uint32_t *n_lo)
{
    int8_t dim = 0, i;
    uint32_t p, q, i2;
    ${DTYPE} size = 0, min_val, max_val, split, side_len, cur_val;
    uint32_t end_idx = start_idx + n - 1;

    /* Find largest bounding box side */
    for (i = 0; i < no_dims; i++)
    {
        side_len = bbox[2 * i + 1] - bbox[2 * i];
        if (side_len > size)
        {
            dim = i;
            size = side_len;
        }
    }

    min_val = bbox[2 * dim];
    max_val = bbox[2 * dim + 1];

    /* Check for zero length or inconsistent */
    if (min_val >= max_val)
        return 1;

    /* Use middle for splitting */
    split = (min_val + max_val) / 2;

    /* Partition all data points around middle */
    p = start_idx;
    q = end_idx;
    while (p <= q)
    {
        if (PA(p, dim) < split)
        {
            p++;
        }
        else if (PA(q, dim) >= split)
        {
            /* Guard for underflow */
            if (q > 0)
            {
                q--;
            }
            else
            {
                break;
            }
        }
        else
        {
            PASWAP(p, q);
            p++;
            q--;
        }
    }

    /* Check for empty splits */
    if (p == start_idx)
    {
        /* No points less than split.
           Split at lowest point instead.
           Minimum 1 point will be in lower box.
        */

        uint32_t j = start_idx;
        split = PA(j, dim);
        for (i2 = start_idx + 1; i2 <= end_idx; i2++)
        {
            /* Find lowest point */
            cur_val = PA(i2, dim);
            if (cur_val < split)
            {
                j = i2;
                split = cur_val;
            }
        }
        PASWAP(j, start_idx);
        p = start_idx + 1;
    }
    else if (p == end_idx + 1)
    {
        /* No points greater than split.
           Split at highest point instead.
           Minimum 1 point will be in higher box.
        */

        uint32_t j = end_idx;
        split = PA(j, dim);
        for (i2 = start_idx; i2 < end_idx; i2++)
        {
            /* Find highest point */
            cur_val = PA(i2, dim);
            if (cur_val > split)
            {
                j = i2;
                split = cur_val;
            }
        }
        PASWAP(j, end_idx);
        p = end_idx;
    }

    /* Set return values */
    *cut_dim = dim;
    *cut_val = split;
    *n_lo = p - start_idx;
    return 0;
}

/************************************************
Construct a sub tree over a range of data points.
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims: number of dimensions
    start_idx : index of first data point to use
    n :  number of data points
    bsp : number of points per leaf
    bbox : bounding box of set of data points
************************************************/
Node_${DTYPE}* construct_subtree_${DTYPE}(${DTYPE} *pa, uint32_t *pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, uint32_t bsp, ${DTYPE} *bbox)
{
    /* Create new node */
    int is_leaf = (n <= bsp);
    Node_${DTYPE} *root = create_node_${DTYPE}(start_idx, n, is_leaf);
    int rval;
    int8_t cut_dim;
    uint32_t n_lo;
    ${DTYPE} cut_val, lv, hv;
    if (is_leaf)
    {
        /* Make leaf node */
        root->cut_dim = -1;
    }
    else
    {
        /* Make split node */
        /* Partition data set and set node info */
        rval = partition_${DTYPE}(pa, pidx, no_dims, start_idx, n, bbox, &cut_dim, &cut_val, &n_lo);
        if (rval == 1)
        {
            root->cut_dim = -1;
            return root;
        }
        root->cut_val = cut_val;
        root->cut_dim = cut_dim;

        /* Recurse on both subsets */
        lv = bbox[2 * cut_dim];
        hv = bbox[2 * cut_dim + 1];

        /* Set bounds for cut dimension */
        root->cut_bounds_lv = lv;
        root->cut_bounds_hv = hv;

        /* Update bounding box before call to lower subset and restore after */
        bbox[2 * cut_dim + 1] = cut_val;
        root->left_child = (struct Node_${DTYPE} *)construct_subtree_${DTYPE}(pa, pidx, no_dims, start_idx, n_lo, bsp, bbox);
        bbox[2 * cut_dim + 1] = hv;

        /* Update bounding box before call to higher subset and restore after */
        bbox[2 * cut_dim] = cut_val;
        root->right_child = (struct Node_${DTYPE} *)construct_subtree_${DTYPE}(pa, pidx, no_dims, start_idx + n_lo, n - n_lo, bsp, bbox);
        bbox[2 * cut_dim] = lv;
    }
    return root;
}

/************************************************
Construct a tree over data points.
Params:
    pa : data points
    no_dims: number of dimensions
    n :  number of data points
    bsp : number of points per leaf
************************************************/
Tree_${DTYPE}* construct_tree_${DTYPE}(${DTYPE} *pa, int8_t no_dims, uint32_t n, uint32_t bsp)
{
    Tree_${DTYPE} *tree = (Tree_${DTYPE} *)malloc(sizeof(Tree_${DTYPE}));
    uint32_t i;
    uint32_t *pidx;
    ${DTYPE} *bbox;

    tree->no_dims = no_dims;

    /* Initialize permutation array */
    pidx = (uint32_t *)malloc(sizeof(uint32_t) * n);
    for (i = 0; i < n; i++)
    {
        pidx[i] = i;
    }

    bbox = (${DTYPE} *)malloc(2 * sizeof(${DTYPE}) * no_dims);
    get_bounding_box_${DTYPE}(pa, pidx, no_dims, n, bbox);
    tree->bbox = bbox;

    /* Construct subtree on full dataset */
    tree->root = (struct Node_${DTYPE} *)construct_subtree_${DTYPE}(pa, pidx, no_dims, 0, n, bsp, bbox);

    tree->pidx = pidx;
    return tree;
}

/************************************************
Create a tree node.
Params:
    start_idx : index of first data point to use
    n :  number of data points
************************************************/
Node_${DTYPE}* create_node_${DTYPE}(uint32_t start_idx, uint32_t n, int is_leaf)
{
    Node_${DTYPE} *new_node;
    if (is_leaf)
    {
        /*
            Allocate only the part of the struct that will be used in a leaf node.
            This relies on the C99 specification of struct layout conservation and padding and
            that dereferencing is never attempted for the node pointers in a leaf.
        */
        new_node = (Node_${DTYPE} *)malloc(sizeof(Node_${DTYPE}) - 2 * sizeof(Node_${DTYPE} *));
    }
    else
    {
        new_node = (Node_${DTYPE} *)malloc(sizeof(Node_${DTYPE}));
    }
    new_node->n = n;
    new_node->start_idx = start_idx;
    return new_node;
}

/************************************************
Delete subtree
Params:
    root : root node of subtree to delete
************************************************/
void delete_subtree_${DTYPE}(Node_${DTYPE} *root)
{
    if (root->cut_dim != -1)
    {
        delete_subtree_${DTYPE}((Node_${DTYPE} *)root->left_child);
        delete_subtree_${DTYPE}((Node_${DTYPE} *)root->right_child);
    }
    free(root);
}

/************************************************
Delete tree
Params:
    tree : Tree struct of kd tree
************************************************/
void delete_tree_${DTYPE}(Tree_${DTYPE} *tree)
{
    delete_subtree_${DTYPE}((Node_${DTYPE} *)tree->root);
    free(tree->bbox);
    free(tree->pidx);
    free(tree);
}

/************************************************
Print
************************************************/
void print_tree_${DTYPE}(Node_${DTYPE} *root, int level)
{
    int i;
    for (i = 0; i < level; i++)
    {
        printf(" ");
    }
    printf("(cut_val: %f, cut_dim: %i)\n", root->cut_val, root->cut_dim);
    if (root->cut_dim != -1)
        print_tree_${DTYPE}((Node_${DTYPE} *)root->left_child, level + 1);
    if (root->cut_dim != -1)
        print_tree_${DTYPE}((Node_${DTYPE} *)root->right_child, level + 1);
}

/************************************************
Calculate squared cartesian distance between points
Params:
    point1_coord : point 1
    point2_coord : point 2
************************************************/
${DTYPE} calc_dist_${DTYPE}(${DTYPE} *point1_coord, ${DTYPE} *point2_coord, int8_t no_dims)
{
    /* Calculate squared distance */
    ${DTYPE} dist = 0, dim_dist;
    int8_t i;
    for (i = 0; i < no_dims; i++)
    {
        dim_dist = point2_coord[i] - point1_coord[i];
        dist += dim_dist * dim_dist;
    }
    return dist;
}

/************************************************
Get squared distance from point to cube in specified dimension
Params:
    dim : dimension
    point_coord : cartesian coordinates of point
    bbox : cube
************************************************/
${DTYPE} get_cube_offset_${DTYPE}(int8_t dim, ${DTYPE} *point_coord, ${DTYPE} *bbox)
{
    ${DTYPE} dim_coord = point_coord[dim];

    if (dim_coord < bbox[2 * dim])
    {
        /* Left of cube in dimension */
        return dim_coord - bbox[2 * dim];
    }
    else if (dim_coord > bbox[2 * dim + 1])
    {
        /* Right of cube in dimension */
        return dim_coord - bbox[2 * dim + 1];
    }
    else
    {
        /* Inside cube in dimension */
        return 0.;
    }
}

/************************************************
Get minimum squared distance between point and cube.
Params:
    point_coord : cartesian coordinates of point
    no_dims : number of dimensions
    bbox : cube
************************************************/
${DTYPE} get_min_dist_${DTYPE}(${DTYPE} *point_coord, int8_t no_dims, ${DTYPE} *bbox)
{
    ${DTYPE} cube_offset = 0, cube_offset_dim;
    int8_t i;

    for (i = 0; i < no_dims; i++)
    {
        cube_offset_dim = get_cube_offset_${DTYPE}(i, point_coord, bbox);
        cube_offset += cube_offset_dim * cube_offset_dim;
    }

    return cube_offset;
}

/************************************************
Search a leaf node for closest point
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims : number of dimensions
    start_idx : index of first data point to use
    size :  number of data points
    point_coord : query point
    closest_idx : index of closest data point found (return)
    closest_dist : distance to closest point (return)
************************************************/
void search_leaf_${DTYPE}(${DTYPE} *restrict pa, uint32_t *restrict pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, ${DTYPE} *restrict point_coord,
                 uint32_t k, uint32_t *restrict closest_idx, ${DTYPE} *restrict closest_dist)
{
    ${DTYPE} cur_dist;
    uint32_t i;
    /* Loop through all points in leaf */
    for (i = 0; i < n; i++)
    {
        /* Get distance to query point */
        cur_dist = calc_dist_${DTYPE}(&PA(start_idx + i, 0), point_coord, no_dims);
        /* Update closest info if new point is closest so far*/
        if (cur_dist < closest_dist[k - 1])
        {
            insert_point_${DTYPE}(closest_idx, closest_dist, pidx[start_idx + i], cur_dist, k);
        }
    }
}


/************************************************
Search a leaf node for closest point with data point mask
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims : number of dimensions
    start_idx : index of first data point to use
    size :  number of data points
    point_coord : query point
    mask : boolean array of invalid (True) and valid (False) data points
    closest_idx : index of closest data point found (return)
    closest_dist : distance to closest point (return)
************************************************/
void search_leaf_${DTYPE}_mask(${DTYPE} *restrict pa, uint32_t *restrict pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, ${DTYPE} *restrict point_coord,
                               uint32_t k, uint8_t *mask, uint32_t *restrict closest_idx, ${DTYPE} *restrict closest_dist)
{
    ${DTYPE} cur_dist;
    uint32_t i;
    /* Loop through all points in leaf */
    for (i = 0; i < n; i++)
    {
        /* Is this point masked out? */
        if (mask[pidx[start_idx + i]])
        {
            continue;
        }
        /* Get distance to query point */
        cur_dist = calc_dist_${DTYPE}(&PA(start_idx + i, 0), point_coord, no_dims);
        /* Update closest info if new point is closest so far*/
        if (cur_dist < closest_dist[k - 1])
        {
            insert_point_${DTYPE}(closest_idx, closest_dist, pidx[start_idx + i], cur_dist, k);
        }
    }
}

/************************************************
Search subtree for nearest to query point
Params:
    root : root node of subtree
    pa : data points
    pidx : permutation index of data points
    no_dims : number of dimensions
    point_coord : query point
    min_dist : minumum distance to nearest neighbour
    mask : boolean array of invalid (True) and valid (False) data points
    closest_idx : index of closest data point found (return)
    closest_dist : distance to closest point (return)
************************************************/
void search_splitnode_${DTYPE}(Node_${DTYPE} *root, ${DTYPE} *pa, uint32_t *pidx, int8_t no_dims, ${DTYPE} *point_coord, 
                      ${DTYPE} min_dist, uint32_t k, ${DTYPE} distance_upper_bound, ${DTYPE} eps_fac, uint8_t *mask,
                      uint32_t *closest_idx, ${DTYPE} *closest_dist)
{
    int8_t dim;
    ${DTYPE} dist_left, dist_right;
    ${DTYPE} new_offset;
    ${DTYPE} box_diff;

    /* Skip if distance bound exeeded */
    if (min_dist > distance_upper_bound)
    {
        return;
    }

    dim = root->cut_dim;

    /* Handle leaf node */
    if (dim == -1)
    {
        if (mask)
        {
            search_leaf_${DTYPE}_mask(pa, pidx, no_dims, root->start_idx, root->n, point_coord, k, mask, closest_idx, closest_dist);
        }
        else
        {
            search_leaf_${DTYPE}(pa, pidx, no_dims, root->start_idx, root->n, point_coord, k, closest_idx, closest_dist);
        }
        return;
    }

    /* Get distance to cutting plane */
    new_offset = point_coord[dim] - root->cut_val;

    if (new_offset < 0)
    {
        /* Left of cutting plane */
        dist_left = min_dist;
        if (dist_left < closest_dist[k - 1] * eps_fac)
        {
            /* Search left subtree if minimum distance is below limit */
            search_splitnode_${DTYPE}((Node_${DTYPE} *)root->left_child, pa, pidx, no_dims, point_coord, dist_left, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
        }

        /* Right of cutting plane. Update minimum distance.
           See Algorithms for Fast Vector Quantization
           Sunil Arya and David M. Mount. */
        box_diff = root->cut_bounds_lv - point_coord[dim];
        if (box_diff < 0)
        {
		box_diff = 0;
        }
        dist_right = min_dist - box_diff * box_diff + new_offset * new_offset;
        if (dist_right < closest_dist[k - 1] * eps_fac)
        {
            /* Search right subtree if minimum distance is below limit*/
            search_splitnode_${DTYPE}((Node_${DTYPE} *)root->right_child, pa, pidx, no_dims, point_coord, dist_right, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
        }
    }
    else
    {
        /* Right of cutting plane */
        dist_right = min_dist;
        if (dist_right < closest_dist[k - 1] * eps_fac)
        {
            /* Search right subtree if minimum distance is below limit*/
            search_splitnode_${DTYPE}((Node_${DTYPE} *)root->right_child, pa, pidx, no_dims, point_coord, dist_right, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
        }

        /* Left of cutting plane. Update minimum distance.
           See Algorithms for Fast Vector Quantization
           Sunil Arya and David M. Mount. */
        box_diff = point_coord[dim] - root->cut_bounds_hv;
        if (box_diff < 0)
        {
        	box_diff = 0;
        }
        dist_left = min_dist - box_diff * box_diff + new_offset * new_offset;
	  if (dist_left < closest_dist[k - 1] * eps_fac)
        {
            /* Search left subtree if minimum distance is below limit*/
            search_splitnode_${DTYPE}((Node_${DTYPE} *)root->left_child, pa, pidx, no_dims, point_coord, dist_left, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
        }
    }
}

/************************************************
Search for nearest neighbour for a set of query points
Params:
    tree : Tree struct of kd tree
    pa : data points
    pidx : permutation index of data points
    point_coords : query points
    num_points : number of query points
    mask : boolean array of invalid (True) and valid (False) data points
    closest_idx : index of closest data point found (return)
    closest_dist : distance to closest point (return)
************************************************/
void search_tree_${DTYPE}(Tree_${DTYPE} *tree, ${DTYPE} *pa, ${DTYPE} *point_coords,
                 uint32_t num_points, uint32_t k, ${DTYPE} distance_upper_bound,
                 ${DTYPE} eps, uint8_t *mask, uint32_t *closest_idxs, ${DTYPE} *closest_dists)
{
    ${DTYPE} min_dist;
    ${DTYPE} eps_fac = 1 / ((1 + eps) * (1 + eps));
    int8_t no_dims = tree->no_dims;
    ${DTYPE} *bbox = tree->bbox;
    uint32_t *pidx = tree->pidx;
    uint32_t j = 0;
#if defined(_MSC_VER) && defined(_OPENMP)
    int32_t i = 0;
    int32_t local_num_points = (int32_t) num_points;
#else
    uint32_t i;
    uint32_t local_num_points = num_points;
#endif
    Node_${DTYPE} *root = (Node_${DTYPE} *)tree->root;

    /* Queries are OpenMP enabled */
    #pragma omp parallel
    {
        /* The low chunk size is important to avoid L2 cache trashing
           for spatial coherent query datasets
        */
        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            for (j = 0; j < k; j++)
            {
                closest_idxs[i * k + j] = UINT32_MAX;
                closest_dists[i * k + j] = DBL_MAX;
            }
            min_dist = get_min_dist_${DTYPE}(point_coords + no_dims * i, no_dims, bbox);
            search_splitnode_${DTYPE}(root, pa, pidx, no_dims, point_coords + no_dims * i, min_dist,
                             k, distance_upper_bound, eps_fac, mask, &closest_idxs[i * k], &closest_dists[i * k]);
        }
    }
}
% endfor
