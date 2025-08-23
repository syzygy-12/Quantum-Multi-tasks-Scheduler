import os
import numpy as np
from qiskit import QuantumCircuit
from qiskit import qpy
import random
import shutil

def generate_mixed_circuits():
    """生成混合尺寸的量子电路"""
    circuits = []
    
    # 生成小尺寸电路 (2-4量子比特)
    for i in range(10):
        n_qubits = random.randint(2, 4)
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # 添加随机门操作
        depth = random.randint(5, 15)
        add_random_gates(qc, n_qubits, depth)
        
        qc.measure(range(n_qubits), range(n_qubits))
        circuits.append((f"mixed_small_{i+1}_{n_qubits}q", qc))
    
    # 生成中等尺寸电路 (6-9量子比特)
    for i in range(10):
        n_qubits = random.randint(6, 9)
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # 添加随机门操作
        depth = random.randint(10, 25)
        add_random_gates(qc, n_qubits, depth)
        
        qc.measure(range(n_qubits), range(n_qubits))
        circuits.append((f"mixed_medium_{i+1}_{n_qubits}q", qc))
    
    # 生成大尺寸电路 (11-14量子比特)
    for i in range(10):
        n_qubits = random.randint(11, 14)
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # 添加随机门操作
        depth = random.randint(15, 40)
        add_random_gates(qc, n_qubits, depth)
        
        qc.measure(range(n_qubits), range(n_qubits))
        circuits.append((f"mixed_large_{i+1}_{n_qubits}q", qc))
    
    return circuits

def add_random_gates(qc, n_qubits, depth):
    """向量子电路添加随机门操作"""
    single_qubit_gates = ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg']
    two_qubit_gates = ['cx', 'cz']
    rotation_gates = ['rx', 'ry', 'rz']
    
    for _ in range(depth):
        if n_qubits > 1 and random.random() < 0.4:  # 40%概率添加双量子比特门
            gate = random.choice(two_qubit_gates)
            control = random.randint(0, n_qubits-1)
            target = random.randint(0, n_qubits-1)
            while target == control:
                target = random.randint(0, n_qubits-1)
            
            if gate == 'cx':
                qc.cx(control, target)
            elif gate == 'cz':
                qc.cz(control, target)
        elif random.random() < 0.7:  # 70%概率添加单量子比特门
            gate = random.choice(single_qubit_gates)
            qubit = random.randint(0, n_qubits-1)
            
            if gate == 'h':
                qc.h(qubit)
            elif gate == 'x':
                qc.x(qubit)
            elif gate == 'y':
                qc.y(qubit)
            elif gate == 'z':
                qc.z(qubit)
            elif gate == 's':
                qc.s(qubit)
            elif gate == 't':
                qc.t(qubit)
            elif gate == 'sdg':
                qc.sdg(qubit)
            elif gate == 'tdg':
                qc.tdg(qubit)
        else:  # 剩余概率添加旋转门
            gate = random.choice(rotation_gates)
            qubit = random.randint(0, n_qubits-1)
            angle = random.uniform(0, 2*np.pi)
            
            if gate == 'rx':
                qc.rx(angle, qubit)
            elif gate == 'ry':
                qc.ry(angle, qubit)
            elif gate == 'rz':
                qc.rz(angle, qubit)

def save_circuits_to_qpy(circuits, output_dir):
    """将电路保存为QPY文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    saved_files = []
    for name, circuit in circuits:
        filename = f"{name}.qpy"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'wb') as f:
            qpy.dump(circuit, f)
        
        saved_files.append(filename)
        print(f"已保存: {filename} (量子比特数: {circuit.num_qubits}, 门数: {circuit.size()})")
    
    return saved_files

def create_mixed_collection():
    """创建混合量子比特集合（从已存在的文件中选取）"""
    mixed_dir = "mixed_qubit_circuits"
    if not os.path.exists(mixed_dir):
        os.makedirs(mixed_dir)
    
    # 从各个目录中选择10个文件
    source_dirs = ["small_qubit_circuits", "medium_qubit_circuits", "large_qubit_circuits"]
    
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"警告: 目录 {source_dir} 不存在，跳过")
            continue
        
        files = os.listdir(source_dir)
        qpy_files = [f for f in files if f.endswith('.qpy')]
        
        # 选择前10个文件（或所有文件如果不足10个）
        selected_files = qpy_files[:min(10, len(qpy_files))]
        
        for file in selected_files:
            src_path = os.path.join(source_dir, file)
            dst_path = os.path.join(mixed_dir, file)
            shutil.copy2(src_path, dst_path)
            print(f"已复制: {file} 从 {source_dir} 到 {mixed_dir}")
    
    print(f"\n混合集合创建完成! 文件保存在目录: {mixed_dir}")

def main_mixed_circuits():
    """主函数：生成混合尺寸的量子电路"""
    print("生成混合尺寸的量子电路...")
    
    # 生成电路
    circuits = generate_mixed_circuits()
    
    # 保存为QPY文件
    output_dir = "mixed_qubit_circuits_new"
    saved_files = save_circuits_to_qpy(circuits, output_dir)
    
    # 输出摘要
    print(f"\n生成完成! 共创建了 {len(saved_files)} 个混合尺寸的量子电路")
    print(f"文件保存在目录: {output_dir}")
    
    # 统计各尺寸电路数量
    small_count = sum(1 for name, _ in circuits if "small" in name)
    medium_count = sum(1 for name, _ in circuits if "medium" in name)
    large_count = sum(1 for name, _ in circuits if "large" in name)
    
    print(f"小尺寸电路 (2-4量子比特): {small_count}个")
    print(f"中等尺寸电路 (6-9量子比特): {medium_count}个")
    print(f"大尺寸电路 (11-14量子比特): {large_count}个")

if __name__ == "__main__":
    # 生成新的混合电路
    main_mixed_circuits()
    
    # 从已存在的文件中创建混合集合
    print("\n" + "="*50)
    print("从已存在的文件中创建混合集合...")
    create_mixed_collection()