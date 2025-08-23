import os
import numpy as np
from qiskit import QuantumCircuit
from qiskit import qpy
import random

def generate_medium_circuits(num_circuits=30, min_qubits=6, max_qubits=9):
    """生成大于5但小于10量子比特的量子电路"""
    circuits = []
    
    for i in range(num_circuits):
        # 随机选择量子比特数量 (6-9)
        n_qubits = random.randint(min_qubits, max_qubits)
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # 随机决定电路深度 (10-25层)
        depth = random.randint(10, 25)
        
        # 标准门集合
        single_qubit_gates = ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg']
        two_qubit_gates = ['cx', 'cz']
        rotation_gates = ['rx', 'ry', 'rz']
        
        # 添加随机门操作
        for _ in range(depth):
            # 随机选择门类型
            if random.random() < 0.4:  # 40%概率添加双量子比特门
                gate = random.choice(two_qubit_gates)
                control = random.randint(0, n_qubits-1)
                target = random.randint(0, n_qubits-1)
                while target == control:  # 确保控制位和目标位不同
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
        
        # 添加测量操作
        qc.measure(range(n_qubits), range(n_qubits))
        
        # 添加到电路列表
        circuits.append((f"medium_circuit_{i+1}_{n_qubits}q", qc))
    
    return circuits

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

def main_medium_circuits():
    """主函数：生成大于5但小于10量子比特的电路"""
    print("生成大于5但小于10量子比特的量子电路...")
    
    # 生成电路
    circuits = generate_medium_circuits(30, 6, 9)
    
    # 保存为QPY文件
    output_dir = "medium_qubit_circuits"
    saved_files = save_circuits_to_qpy(circuits, output_dir)
    
    # 输出摘要
    print(f"\n生成完成! 共创建了 {len(saved_files)} 个大于5但小于10量子比特的量子电路")
    print(f"文件保存在目录: {output_dir}")

if __name__ == "__main__":
    main_medium_circuits()