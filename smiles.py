from rdkit import Chem
from rdkit.Chem import AllChem

# 这段代码使用了 RDKit 库来处理化学分子的 SMILES 字符串，将其转换为分子对象，添加氢原子，生成三维坐标，并将分子保存为 PDB 文件

# SMILES string
smiles_string = "CCOC1=CC2=C(C=C1)N=C(S2)S(=O)(=O)N"  # 定义了一个 SMILES（简化分子线性输入规范）字符串，它是一种用 ASCII 字符串明确描述分子结构的规范。this is for aspirin

# Convert SMILES to mol
# 使用 Chem.MolFromSmiles 函数将 SMILES 字符串转换为 RDKit 的分子对象 mol。如果 SMILES 字符串有效，该函数会返回对应的分子对象；若无效则返回 None。
mol = Chem.MolFromSmiles(smiles_string)

# 使用 Chem.AddHs 函数为分子对象 mol 添加氢原子。在很多情况下，SMILES 字符串中不明确表示氢原子，所以在进行后续处理（如三维结构生成）前，通常需要添加氢原子以确保分子结构完整。
# Add Hydrogens
mol = Chem.AddHs(mol)

# 使用 AllChem.EmbedMolecule 函数为分子对象 mol 生成三维坐标。该函数会尝试为分子构建一个合理的三维结构，其内部使用了一些算法来确定原子的空间位置。
# Generate 3D coordinates
AllChem.EmbedMolecule(mol)

# 使用 Chem.MolToPDBFile 函数将分子对象 mol 保存为 PDB（Protein Data Bank）文件。PDB 文件是一种常见的用于存储分子三维结构信息的文件格式。这里将分子保存为名为 output.pdb 的文件，该文件会包含分子中每个原子的坐标等信息。
# Write to PDB file
Chem.MolToPDBFile(mol, "ethoxzolamide.pdb")