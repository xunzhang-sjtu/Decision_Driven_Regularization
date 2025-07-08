import numpy as np
np.random.seed(1)

class VectorBinaryTreeNode:
    def __init__(self, depth, max_depth, output_dim, current_depth=0, feature_names=None):
        """
        带向量输出的二叉树节点类
        
        参数:
        - depth: 树的深度
        - max_depth: 最大深度
        - output_dim: 输出向量维度
        - current_depth: 当前深度
        - feature_names: 特征名称列表
        """
        self.depth = current_depth
        self.max_depth = max_depth
        self.output_dim = output_dim
        self.feature_names = feature_names or [f'feature_{i}' for i in range(max_depth)]
        self.is_leaf = current_depth >= max_depth
        
        if not self.is_leaf:
            # 内部节点属性
            self.split_feature = self.feature_names[current_depth]
            self.split_threshold = np.random.uniform(-1, 1)  # 随机分裂阈值
            self.left = VectorBinaryTreeNode(depth, max_depth, output_dim, current_depth+1, feature_names)
            self.right = VectorBinaryTreeNode(depth, max_depth, output_dim, current_depth+1, feature_names)
        else:
            # 叶子节点属性 - 现在是向量
            self.value = np.random.uniform(0, 10, size=output_dim)  # 随机向量估计值
    
    def evaluate(self, x):
        """评估样本x，返回向量"""
        if self.is_leaf:
            return self.value
        
        if x[self.split_feature] <= self.split_threshold:
            return self.left.evaluate(x)
        else:
            return self.right.evaluate(x)
    
    def get_rules(self, parent_rules=None):
        """获取从根节点到当前节点的规则和向量值"""
        if parent_rules is None:
            parent_rules = []
            
        if self.is_leaf:
            return [{'rules': parent_rules, 'value': self.value}]
        
        left_rules = parent_rules + [f"{self.split_feature} <= {self.split_threshold:.2f}"]
        right_rules = parent_rules + [f"{self.split_feature} > {self.split_threshold:.2f}"]
        
        return self.left.get_rules(left_rules) + self.right.get_rules(right_rules)
    
    # def visualize(self, graph=None, node_id=0):
    #     """可视化二叉树（简化版，显示向量维度）"""
    #     if graph is None:
    #         graph = Digraph()
    #         graph.attr('node', shape='box')
        
    #     if self.is_leaf:
    #         graph.node(str(node_id), label=f"Value: vector({self.output_dim})")
    #     else:
    #         graph.node(str(node_id), 
    #                   label=f"{self.split_feature}\n<= {self.split_threshold:.2f}?")
            
    #         left_id = 2 * node_id + 1
    #         right_id = 2 * node_id + 2
            
    #         self.left.visualize(graph, left_id)
    #         self.right.visualize(graph, right_id)
            
    #         graph.edge(str(node_id), str(left_id), label="True")
    #         graph.edge(str(node_id), str(right_id), label="False")
        
    #     if node_id == 0:
    #         return graph

def generate_vector_binary_tree(depth=3, output_dim=2, feature_names=None):
    """
    生成带向量输出的二叉树结构
    
    参数:
    - depth: 树的深度
    - output_dim: 输出向量维度
    - feature_names: 可选的特征名称列表
    
    返回:
    - VectorBinaryTreeNode: 树的根节点
    """
    return VectorBinaryTreeNode(depth, depth, output_dim, feature_names=feature_names)

# 示例使用
if __name__ == "__main__":
    # 1. 生成深度为3，输出维度为3的二叉树
    feature_names = ['age', 'income']
    tree = generate_vector_binary_tree(depth=2, output_dim=2, feature_names=feature_names[:3])
    
    # 2. 打印所有叶子节点的规则和向量值
    print("二叉树所有路径规则和叶子节点向量值:")
    for i, rule_info in enumerate(tree.get_rules()):
        print(f"路径 {i+1}: {' AND '.join(rule_info['rules'])}")
        print(f"向量值: {rule_info['value']}")
        print("-" * 50)
    
    # # 3. 可视化树结构
    # dot = tree.visualize()
    # dot.render('vector_binary_tree', format='png', cleanup=True)
    # print("二叉树可视化已保存为 vector_binary_tree.png")
    
    # 4. 演示评估样本
    sample = {'age': 0.5, 'income': -0.8, 'education': 1.2}
    prediction = tree.evaluate(sample)
    print(f"\n样本评估: {sample} -> 预测向量: {prediction}")