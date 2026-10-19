#include <iostream>
#include <string>
#include <memory>

class VarNode;
class AddNode;
class ExprNode;

// 用别名简化 shared_ptr 的书写
using Expr = std::shared_ptr<ExprNode>;

// ----- 架构层 -----
// 改写接口 (IRMutator)
class IRMutator {
public:
    virtual ~IRMutator() = default;

    // 核心变动：visit 必须返回 Expr，允许在遍历过程中替换节点
    virtual Expr visit(VarNode* node) = 0;
    virtual Expr visit(AddNode* node) = 0;
};

// 数据结构接口 (Node)
class ExprNode : public std::enable_shared_from_this<ExprNode> {
public:
    virtual ~ExprNode() = default;
    virtual Expr accept(IRMutator* mutator) = 0;
};


// ----- 实现层 ----- 
// 具体数据结构
class VarNode : public ExprNode {
public:
    std::string name;
    explicit VarNode(std::string n) : name(std::move(n)) {}

    Expr accept(IRMutator* mutator) override {
        return mutator->visit(this);
    }
};

class AddNode : public ExprNode {
public:
    Expr lhs;
    Expr rhs;
    AddNode(Expr l, Expr r) : lhs(std::move(l)), rhs(std::move(r)) {}

    Expr accept(IRMutator* mutator) override {
        return mutator->visit(this);
    }
};


// 常量折叠优化 Pass (Constant Folding)
class ConstFoldingMutator : public IRMutator {
public:
    Expr visit(VarNode* node) override {
        // 叶子节点，一般不需要改写，直接返回它自己。
        return node->shared_from_this();
    }

    Expr visit(AddNode* node) override {
        // 递归改写左子树和右子树
        // 核心思想：自底向上。必须先把底层的子节点优化干净，再来处理当前层。
        Expr optimized_lhs = node->lhs->accept(this);
        Expr optimized_rhs = node->rhs->accept(this);

        // 核心优化逻辑 (X + 0 -> X)
        // 使用 dynamic_pointer_cast 在运行时进行“模式匹配”
        if (auto rhs_var = std::dynamic_pointer_cast<VarNode>(optimized_rhs)) {
            // 如果有右子树，并且右子树是一个 VarNode，名字为 "0"
            if (rhs_var->name == "0") {
                // 触发常数折叠：直接丢弃 AddNode 和 0，把左子树“提拔”上来
                return optimized_lhs;
            }
        }

        // 如果没有触发优化规则（比如是 x + y，或者 x + 1）
        // 则用优化过的左右子树，重新组装出一个新的 AddNode 返回
        return std::make_shared<AddNode>(optimized_lhs, optimized_rhs);
    }
};

int main() {
    // 构造 DAG：计算图 (x + 0)
    Expr x = std::make_shared<VarNode>("x");
    Expr zero = std::make_shared<VarNode>("0");
    Expr graph = std::make_shared<AddNode>(x, zero);

    ConstFoldingMutator optimizer;
    
    // 执行优化 Pass
    Expr optimized_graph = graph->accept(&optimizer);

    // 验证结果：如果优化成功，optimized_graph 应该直接指向 VarNode("x")
    if (auto var = std::dynamic_pointer_cast<VarNode>(optimized_graph)) {
        std::cout << "Optimization Success! Remaining Node: " << var->name << std::endl;
    } else {
        std::cout << "Optimization Failed! Not folded." << std::endl;
    }

    return 0;
}