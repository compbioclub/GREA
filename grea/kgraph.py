import re
import time
import json
from bioservices import KEGG


class KGNode:
    def __init__(self, name, id=None, level=None, source='KEGG'):
        self.name = name
        self.id = id
        self.source = source
        self.level = level
        self.children = []
        self.parent = None

    def add_child(self, node):
        self.children.append(node)
        node.parent = self

    def __str__(self, level=0):
        ret = "-" * level + f"{self.name}"
        if self.id:
            ret += f" ({self.id})"
        ret += "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret
    
    def to_dict(self):
        return {
            "name": self.name,
            "source": self.source,
            "id": self.id,
            "children": [child.to_dict() for child in self.children]
        }
    
    def save_json(self, out_fn):
        tree_json = self.to_dict()
        with open(out_fn, "w") as f:
            json.dump(tree_json, f, indent=2)

##### KEGG parser ########

def get_kegg_b2c_library():
    kegg = KEGG()
    brite = kegg.get("br:br08901")  # KEGG pathway maps
    root, id2node, level2node = parse_kegg_brite_tree(brite, kegg)

    # Get all B-level nodes and their C-level children
    b_to_c = {
        node.name: [child.name for child in node.children]
        for node in level2node['B']
    }
    return b_to_c

def parse_kegg_brite_tree(
        brite_text, kegg, 
        get_kos=False, organism='hsa', cache={}):
    level_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4}

    root = KGNode("KEGG", level='Root')
    stack = [root]

    id2node = {}
    level2node = {'A': [], 'B': [], 'C': [], 'D': []}

    lines = brite_text.strip().split("\n")
    for line in lines:
        m = re.match(r'^([A-D])\s*(.+)', line)
        if not m:
            continue
        level_chr  = m.group(1)
        content = m.group(2).strip()
        level_num = level_map[level_chr ]

        # Try to extract KEGG ID and name (e.g., "01100  Metabolic pathways")
        if level_chr  in {'C', 'D'}:
            # Format: "01100  Metabolic pathways [PATH:hsa01100]"
            parts = re.match(r'^(\S+)\s+(.*)', content)
            if parts:
                kegg_id = parts.group(1)
                name = re.sub(r'\s*\[.*?\]', '', parts.group(2)).strip()
            else:
                kegg_id = None
                name = content
        else:
            kegg_id = None
            name = content.strip()

        # Synthetic ID for A/B
        kegg_id = kegg_id if kegg_id else f"{level_chr}:{name}"
        
        node = KGNode(name, id=kegg_id, level=level_chr)
        level2node[level_chr].append(node)
        id2node[kegg_id] = node

        # Adjust stack depth
        if len(stack) < level_num:
            for _ in range(level_num - len(stack)):
                dummy = KGNode("Unnamed", level=None)
                stack[-1].add_child(dummy)
                stack.append(dummy)
        else:
            stack = stack[:level_num]

        stack[-1].add_child(node)
        stack = stack[:level_num] + [node]

        # If C-level node is a pathway, add modules and KO terms
        if get_kos and level_chr == 'C' and kegg_id:
            kegg_id = organism + kegg_id
            try:
                get_kegg_modules_kos(kegg, kegg_id, node, cache)
            except Exception as e:
                print(f"Error for {kegg_id}: {e}")
                continue

    return root, id2node, level2node


def get_kegg_modules_kos(kegg, kegg_id, node, cache):

    if kegg_id in cache:
        module_data = cache[kegg_id]
    else:
        module_links = kegg.link("module", kegg_id)
        module_ids = re.findall(r'module:(M\d+)', module_links)
        module_data = []
        for mid in module_ids:
            module_entry = kegg.get(mid)
            module_name_match = re.search(r'NAME\s+(.+)', module_entry)
            module_name = module_name_match.group(1).strip() if module_name_match else mid
                        
            # KO terms in module
            ko_matches = re.findall(r'(K\d{5})', module_entry)
            module_data.append({
                "mid": mid,
                "name": module_name,
                "kos": list(sorted(set(ko_matches)))
            })
            time.sleep(0.2)
            cache[kegg_id] = module_data

    for module in module_data:
        module_node = KGNode(module["name"], module["mid"])
        node.add_child(module_node)
    for ko in module["kos"]:
        module_node.add_child(KGNode(ko, ko))