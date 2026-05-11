import xml.etree.ElementTree as ET
import argparse
import sys
import os

def generate_minimal_urdf(input_path, output_path, pairs_list):
    if not os.path.exists(input_path):
        print(f"Error: Input URDF {input_path} not found.")
        sys.exit(1)

    tree = ET.parse(input_path)
    root = tree.getroot()

    joints = []
    link_to_parent = {}
    
    for joint in root.findall('joint'):
        parent = joint.find('parent')
        child = joint.find('child')
        if parent is not None and child is not None:
            p_link = parent.get('link')
            c_link = child.get('link')
            link_to_parent[c_link] = (p_link, joint)
            joints.append(joint)
            
    link_elements = {link.get('name'): link for link in root.findall('link')}

    all_lcas = set()
    required_links = set()
    required_joints = set()

    # Process each pair
    for pair in pairs_list:
        targets = pair.split(',')
        ancestors_sets = []
        for target in targets:
            if target not in link_elements:
                print(f"Warning: Target link {target} not found in the URDF.")
                
            curr = target
            path = [curr]
            while curr in link_to_parent:
                parent_link, _ = link_to_parent[curr]
                path.append(parent_link)
                curr = parent_link
            ancestors_sets.append(path)
            
        if not ancestors_sets:
            continue

        # Find LCA for this specific pair
        common_ancestors = set(ancestors_sets[0])
        for path in ancestors_sets[1:]:
            common_ancestors.intersection_update(path)
            
        lca = None
        for node in ancestors_sets[0]:
            if node in common_ancestors:
                lca = node
                break
                
        if lca is None:
            print(f"Error: No common ancestor found for pair: {pair}")
            sys.exit(1)

        print(f"Found LCA for {pair}: {lca}")
        all_lcas.add(lca)

        # Collect required links and joints for this pair, up to the LCA
        for target in targets:
            curr = target
            while True:
                required_links.add(curr)
                if curr == lca:
                    break
                if curr not in link_to_parent:
                    break
                parent_link, joint_elem = link_to_parent[curr]
                required_joints.add(joint_elem)
                curr = parent_link

    if not required_links:
        print("Error: No valid target links were processed.")
        sys.exit(1)

    # Initialize new URDF
    new_root = ET.Element('robot', name=root.get('name', 'minimal_robot'))
    
    # Copy materials
    for mat in root.findall('material'):
        new_root.append(mat)

    # If there are multiple LCAs, we must stitch them together to a common root to satisfy parsers
    # If there is only one LCA, it serves as the natural root of the whole tree.
    if len(all_lcas) > 1:
        dummy_root_name = "minimal_base_link"
        dummy_link = ET.Element('link', name=dummy_root_name)
        new_root.append(dummy_link)
        
        # Connect each LCA to the dummy root
        for i, lca in enumerate(sorted(list(all_lcas))):
            joint_elem = ET.Element('joint', name=f"joint_dummy_{i}", type="fixed")
            parent = ET.Element('parent', link=dummy_root_name)
            child = ET.Element('child', link=lca)
            # Origin is inherently identity if omitted, but let's add an explicit zero origin just in case
            origin = ET.Element('origin', rpy="0 0 0", xyz="0 0 0")
            joint_elem.append(parent)
            joint_elem.append(child)
            joint_elem.append(origin)
            new_root.append(joint_elem)
            
        print(f"Stitched {len(all_lcas)} separate subtrees to dummy root '{dummy_root_name}'")

    print(f"Included {len(required_links)} original links and {len(required_joints)} original joints.")
        
    for link_name in sorted(required_links):
        if link_name in link_elements:
            new_root.append(link_elements[link_name])
            
    for joint_elem in required_joints:
        new_root.append(joint_elem)
        
    new_tree = ET.ElementTree(new_root)
    ET.indent(new_tree, space="  ", level=0)
    new_tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"Successfully generated minimal URDF: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract a minimal URDF tree containing the required pairs of links.')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the full input URDF file')
    parser.add_argument('--output', '-o', type=str, required=True, help='Path to save the minimal output URDF file')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--links', '-l', type=str, nargs='+', help='List of target links to include in the minimal URDF (finds a single LCA for all).')
    group.add_argument('--pairs', '-p', type=str, nargs='+', help='Comma-separated pairs of links (e.g. linkA,linkB). Calculates LCA per pair and stitches them to a dummy root.')
    
    args = parser.parse_args()
    
    if args.links:
        # Wrap everything into one single "pair" group to find one global LCA
        pairs_list = [",".join(args.links)]
    else:
        pairs_list = args.pairs
        
    generate_minimal_urdf(args.input, args.output, pairs_list)

if __name__ == '__main__':
    main()
