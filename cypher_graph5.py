# graph_builder_final.py
import os
import csv
import math
from collections import defaultdict

# --- Configuration ---
# IMPORTANT: You MUST update this dictionary to match your YOLO model's classes.
CLASS_MAP = {
    0: "Circle",
    1: "Circle_with_square",
    2: "Diamond",
    3: "Other",
    4: "Rectangle_large",
    5: "Rectangle_small",
    # Add all your other class IDs and names here
}

# IMPORTANT: You MUST update these with the dimensions of your P&ID image.
IMAGE_WIDTH = 3300
IMAGE_HEIGHT = 2550

# --- Tuning Parameters ---
# These thresholds may need to be adjusted based on your image resolution and line detection quality.
LINE_STITCHING_THRESHOLD_PX = 8  # Max distance to connect line endpoints together.
SYMBOL_CONNECTION_THRESHOLD_PX = 15 # Max distance from a line endpoint to a symbol's boundary.
T_JUNCTION_THRESHOLD_PX = 15      # Max distance from a line endpoint to the body of another line to form a joint.
PROXIMITY_EXPANSION_PX = 20 

# --- Helper Functions ---

def get_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_closest_point_on_segment(p, a, b):
    """Calculates the closest point on segment (a, b) to point p."""
    px, py = p
    ax, ay = a
    bx, by = b

    ab_x, ab_y = bx - ax, by - ay
    ap_x, ap_y = px - ax, py - ay

    ab_mag_sq = ab_x**2 + ab_y**2
    if ab_mag_sq == 0:
        return a

    # Projection of AP onto AB, clamped to the segment
    t = max(0, min(1, (ap_x * ab_x + ap_y * ab_y) / ab_mag_sq))
    
    # Return the coordinates of the closest point
    closest_point = (ax + t * ab_x, ay + t * ab_y)
    return closest_point

def point_to_bbox_distance(point, node):
    """Calculate the shortest distance from a point to the boundary of a node's bounding box."""
    px, py = point
    x_min = node['x_center'] - node['width'] / 2
    x_max = node['x_center'] + node['width'] / 2
    y_min = node['y_center'] - node['height'] / 2
    y_max = node['y_center'] + node['height'] / 2
    
    closest_x = max(x_min, min(px, x_max))
    closest_y = max(y_min, min(py, y_max))
    
    return get_distance(point, (closest_x, closest_y))

# NEW: Helper function to check for bounding box intersection
def do_bboxes_intersect(box1, box2):
    # box = (x_min, y_min, x_max, y_max)
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])


# --- Data Loading Functions ---

def load_yolo_nodes(yolo_input_path, pid_filename):
    """Parses a YOLO annotation file and returns a list of node data dictionaries."""
    if not os.path.exists(yolo_input_path):
        print(f"Error: YOLO input file not found at {yolo_input_path}")
        return None
    nodes = []
    node_counters = {}
    with open(yolo_input_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5: continue
            try:
                class_id = int(parts[0])
                label = CLASS_MAP.get(class_id)
                if label is None: continue
                
                node_counters[label] = node_counters.get(label, 0) + 1
                unique_tag = f"{pid_filename}_{label}_{node_counters[label]}"
                
                x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:])
                nodes.append({
                    "tag": unique_tag, "label": label,
                    "x_center": int(x_center_norm * IMAGE_WIDTH), "y_center": int(y_center_norm * IMAGE_HEIGHT),
                    "width": int(width_norm * IMAGE_WIDTH), "height": int(height_norm * IMAGE_HEIGHT)
                })
            except (ValueError, IndexError):
                print(f"Warning: Skipping malformed line in YOLO file: {line.strip()}")
    print(f"Loaded {len(nodes)} symbol nodes from YOLO file.")
    return nodes

def load_detected_lines(csv_input_path):
    """Reads a CSV file containing detected line segments."""
    if not os.path.exists(csv_input_path):
        print(f"Error: Lines CSV file not found at {csv_input_path}")
        return None
    lines = []
    with open(csv_input_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 4:
                try:
                    lines.append(tuple(map(int, row)))
                except ValueError:
                    continue
    print(f"Loaded {len(lines)} line segments from CSV file.")
    return lines

# --- Core Logic: Line Stitching and Path Analysis ---

def find_closest_node(point, nodes, max_distance):
    """Finds the closest node to a point by measuring distance to its bounding box."""
    closest_node, min_dist = None, float('inf')
    for node in nodes:
        dist = point_to_bbox_distance(point, node)
        if dist < min_dist:
            min_dist, closest_node = dist, node
    return closest_node if min_dist <= max_distance else None

def build_line_graph(lines, threshold):
    """Groups connected line segments into an adjacency list."""
    adj = defaultdict(list)
    endpoints = []
    for i, (x1, y1, x2, y2) in enumerate(lines):
        endpoints.append(((x1, y1), i))
        endpoints.append(((x2, y2), i))

    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            p1, line_idx1 = endpoints[i]
            p2, line_idx2 = endpoints[j]
            if line_idx1 != line_idx2 and get_distance(p1, p2) < threshold:
                adj[line_idx1].append(line_idx2)
                adj[line_idx2].append(line_idx1)
    return adj

def get_line_paths(lines, adj):
    """Finds all continuous paths of lines using DFS on the line graph."""
    paths, visited = [], set()
    for i in range(len(lines)):
        if i not in visited:
            path, stack = [], [i]
            visited.add(i)
            while stack:
                u = stack.pop()
                path.append(u)
                for v in adj.get(u, []):
                    if v not in visited:
                        visited.add(v)
                        stack.append(v)
            paths.append(path)
    return paths

def get_path_endpoints(path_indices, lines):
    """Finds the two most distant endpoints of a continuous path of lines."""
    points_in_path = {p for idx in path_indices for p in ((lines[idx][0], lines[idx][1]), (lines[idx][2], lines[idx][3]))}
    points_list = list(points_in_path)
    if len(points_list) < 2: return None

    max_dist, end_points = 0, (None, None)
    for i in range(len(points_list)):
        for j in range(i + 1, len(points_list)):
            dist = get_distance(points_list[i], points_list[j])
            if dist > max_dist:
                max_dist, end_points = dist, (points_list[i], points_list[j])
    return end_points

# --- NEW: Multi-Joint Graph Generation Logic ---

def generate_full_cypher_script(yolo_input_path, lines_csv_path, cypher_output_path):
    pid_filename = os.path.splitext(os.path.basename(yolo_input_path))[0]

    # 1. Load data and stitch paths (same as before)
    symbol_nodes = load_yolo_nodes(yolo_input_path, pid_filename)
    lines = load_detected_lines(lines_csv_path)
    if not symbol_nodes or not lines: return

    line_adj = build_line_graph(lines, LINE_STITCHING_THRESHOLD_PX)
    line_paths_indices = get_line_paths(lines, line_adj)
    print(f"Stitched into {len(line_paths_indices)} continuous paths.")
    
    path_details = []
    for i, indices in enumerate(line_paths_indices):
        endpoints = get_path_endpoints(indices, lines)
        if not endpoints: continue
        start_node = find_closest_node(endpoints[0], symbol_nodes, SYMBOL_CONNECTION_THRESHOLD_PX)
        end_node = find_closest_node(endpoints[1], symbol_nodes, SYMBOL_CONNECTION_THRESHOLD_PX)
        path_details.append({
            'id': i, 'indices': indices, 'endpoints': endpoints,
            'start_node': start_node, 'end_node': end_node
        })

    # 2. NEW: Collect all potential joints without making decisions yet
    direct_connections = [p for p in path_details if p['start_node'] and p['end_node'] and p['start_node']['tag'] != p['end_node']['tag']]
    potential_branches = [p for p in path_details if bool(p['start_node']) ^ bool(p['end_node'])]
    
    joints_on_main_paths = defaultdict(list)
    for branch in potential_branches:
        branch_symbol_node = branch['start_node'] or branch['end_node']
        free_endpoint = branch['endpoints'][1] if branch['start_node'] else branch['endpoints'][0]

        for main_path in direct_connections:
            for line_idx in main_path['indices']:
                p1, p2 = (lines[line_idx][0], lines[line_idx][1]), (lines[line_idx][2], lines[line_idx][3])
                joint_coords = get_closest_point_on_segment(free_endpoint, p1, p2)
                dist = get_distance(free_endpoint, joint_coords)

                if dist < T_JUNCTION_THRESHOLD_PX:
                    # Found a potential joint, store its info
                    dist_from_start = get_distance(joint_coords, main_path['endpoints'][0])
                    joints_on_main_paths[main_path['id']].append({
                        'coords': joint_coords,
                        'branch_symbol_tag': branch_symbol_node['tag'],
                        'dist_from_start': dist_from_start
                    })
                    break # A branch only connects at one point to a main line

    # 3. NEW: Build the final graph structure by processing the collected joints
    final_nodes = list(symbol_nodes)
    final_relationships = set()
    joint_counter = 0

    for main_path in direct_connections:
        path_id = main_path['id']
        associated_joints = joints_on_main_paths.get(path_id, [])

        if not associated_joints:
            # Case 1: No joints on this path, create a simple direct connection
            rel = tuple(sorted((main_path['start_node']['tag'], main_path['end_node']['tag'])))
            final_relationships.add(rel)
        else:
            # Case 2: One or more joints found. Create a chain.
            # Sort joints by their distance from the start of the main path
            sorted_joints = sorted(associated_joints, key=lambda j: j['dist_from_start'])
            
            # Create the joint nodes
            joint_nodes_for_path = []
            for joint_info in sorted_joints:
                joint_counter += 1
                joint_tag = f"{pid_filename}_Joint_{joint_counter}"
                joint_node = {'tag': joint_tag, 'label': 'Joint', 'x_center': int(joint_info['coords'][0]), 'y_center': int(joint_info['coords'][1])}
                final_nodes.append(joint_node)
                joint_nodes_for_path.append(joint_node)
                # Connect the branch symbol to this new joint
                final_relationships.add(tuple(sorted((joint_info['branch_symbol_tag'], joint_tag))))

            # Create the main line chain: Symbol -> Joint1 -> Joint2 -> ... -> Symbol
            chain_tags = [main_path['start_node']['tag']] + [j['tag'] for j in joint_nodes_for_path] + [main_path['end_node']['tag']]
            for i in range(len(chain_tags) - 1):
                final_relationships.add(tuple(sorted((chain_tags[i], chain_tags[i+1]))))

    print(f"Detected and created {joint_counter} Joint nodes.")

    # 4. NEW: Add proximity-based connections for specific symbols
    print("Checking for proximity-based connections...")
    proximity_connections_found = 0
    for i in range(len(symbol_nodes)):
        node1 = symbol_nodes[i]
        # Only check from Rectangle_small (class_id 5)
        if node1['label'] != 'Rectangle_small':
            continue

        # Define the expanded bounding box for node1
        n1_x_min = node1['x_center'] - node1['width'] / 2 - PROXIMITY_EXPANSION_PX
        n1_y_min = node1['y_center'] - node1['height'] / 2 - PROXIMITY_EXPANSION_PX
        n1_x_max = node1['x_center'] + node1['width'] / 2 + PROXIMITY_EXPANSION_PX
        n1_y_max = node1['y_center'] + node1['height'] / 2 + PROXIMITY_EXPANSION_PX
        expanded_box1 = (n1_x_min, n1_y_min, n1_x_max, n1_y_max)

        for j in range(i + 1, len(symbol_nodes)):
            node2 = symbol_nodes[j]
            
            # Define the original bounding box for node2
            n2_x_min = node2['x_center'] - node2['width'] / 2
            n2_y_min = node2['y_center'] - node2['height'] / 2
            n2_x_max = node2['x_center'] + node2['width'] / 2
            n2_y_max = node2['y_center'] + node2['height'] / 2
            original_box2 = (n2_x_min, n2_y_min, n2_x_max, n2_y_max)

            if do_bboxes_intersect(expanded_box1, original_box2):
                rel = tuple(sorted((node1['tag'], node2['tag'])))
                if rel not in final_relationships:
                    final_relationships.add(rel)
                    proximity_connections_found += 1
    
    print(f"Found and added {proximity_connections_found} new proximity-based connections.")

    # 5. Generate Cypher script (uses the now-augmented final_relationships set)
    with open(cypher_output_path, 'w') as f:
        f.write(f"// Cypher script for {pid_filename} with multi-joint and proximity logic\n\n")
        f.write("// --- 1. Create all symbol and joint nodes ---\n")
        for node in final_nodes:
            f.write(f"MERGE (n:{node['label']} {{tag: '{node['tag']}'}})\n")
            if 'x_center' in node and 'y_center' in node:
                f.write(f"SET n.x = {node['x_center']}, n.y = {node['y_center']};\n\n")
            else: f.write("\n")

        f.write("// --- 2. Create all relationships ---\n\n")
        for node1_tag, node2_tag in final_relationships:
            f.write(f"MATCH (a {{tag: '{node1_tag}'}}), (b {{tag: '{node2_tag}'}})\n")
            f.write(f"MERGE (a)-[:CONNECTS_TO]-(b);\n\n")
        
        f.write("// --- End of script --- \n")

    print(f"Successfully generated final Cypher file at: {cypher_output_path}")

# --- Main Execution Block ---
if __name__ == "__main__":
    yolo_file = "yoloh9_1.txt"
    lines_file = "detected_lines.xls"
    cypher_file = "pid_graph_complete.cypher"
    generate_full_cypher_script(yolo_file, lines_file, cypher_file)
