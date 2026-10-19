#include <iostream>
#include <string>
#include <memory>
#include <unordered_map>

// 前置声明，告诉编译器有这几个数据结构
class VarNode;
class AddNode;

// ----- 架构层 -----

// 操作接口 (Visitor) ：所有 Compiler Pass 的基类
class IRVisitor {
public:
    // 针对不同类型的节点，提供精确的重载函数
    // 这里利用了 C++ 的函数重载（静态绑定），使得 Visitor 明确知道自己正在处理哪种具体节点
    // = 0 表示这是一个纯虚函数，派生类必须实现它
    virtual void visit(VarNode* node) = 0;
    virtual void visit(AddNode* node) = 0;

    // 虚析构函数，确保派生类对象被正确析构
    // = default 表示使用编译器生成的默认实现
    virtual ~IRVisitor() = default;
};

// 数据结构接口 (Node)：所有 AST 节点的基类
class ExprNode {
public:
    // 唯一向外暴露的操作入口。
    // 接受一个基类指针 visitor，并将其“引入”到自己的作用域
    virtual void accept(IRVisitor* visitor) = 0;

    virtual ~ExprNode() = default;
};

// ----- 实现层 ----- 
class VarNode : public ExprNode {
public:
    std::string name;
    
    // 这里使用 std::move 避免堆上字符串的拷贝开销，提高性能
    explicit VarNode(std::string n) : name(std::move(n)) {}

    void accept(IRVisitor* visitor) override {
        // 实现“踢皮球”机制：调用 visitor 的某个方法，并把当前对象的指针传给它
        visitor->visit(this);
        // 思考：为什么不能把这行代码写在基类 ExprNode 里的 accept 中，而必须在每一个子类（如 VarNode, AddNode）里“死板地”重复写一遍？
        // 因为 C++ 是静态类型语言，函数重载是在编译期决定的，如果直接写在基类中，编译器无法知道当前对象的实际类型，从而无法调用正确的 visit 方法。每个子类必须重写 accept 方法，以确保调用的是对应类型的 visit 方法，从而实现多态行为。
    }
};

class AddNode : public ExprNode {
public:
    // 采用 RAII 思想：使用 unique_ptr 独占子节点的所有权，防止多 Pass 遍历时的内存泄漏
    std::unique_ptr<ExprNode> lhs;
    std::unique_ptr<ExprNode> rhs;

    AddNode(std::unique_ptr<ExprNode> l, std::unique_ptr<ExprNode> r) 
        : lhs(std::move(l)), rhs(std::move(r)) {}

    void accept(IRVisitor* visitor) override {
        // 同上
        visitor->visit(this);
    }
};


// ----- 实现层：具体操作 (Concrete Visitor / Pass) -----
// 这是一个具体的 Compiler Pass，用于打印 IR 树
class PrintVisitor : public IRVisitor {
public:
    void visit(VarNode* node) override {
        // 叶子节点，到底了，直接触发动作
        std::cout << node->name;
    }

    void visit(AddNode* node) override {
        std::cout << "(";
        
        // 实现对左子树的递归访问
        // 思考：如何让左子树继续执行当前的 Print 逻辑？
        // 不要尝试去 cast 子节点的类型，而是把“当前正在执行的 Pass（即 this）”继续喂给子节点。
        node->lhs->accept(this);

        std::cout << " + ";
        
        // 实现对右子树的递归访问
        node->rhs->accept(this);

        std::cout << ")";
    }
};

// ----- 求值 Pass (Evaluate Visitor) -----
class EvaluateVisitor : public IRVisitor {
private:
    // 结果寄存器：每次 visit 执行完后，它的“返回值”就会暂存在这里
    int current_result_ = 0; 
    
    // 模拟运行时的变量环境 (Runtime Context)，用来充当“字典”或“内存”
    // 比如你的表达式是 x + y，AST 里只有变量的名字 "x" 和 "y"。当你要计算具体数值时，你必须查字典：env_ 里可能存着：{"x": 10, "y": 20}
    std::unordered_map<std::string, int> env_;

public:
    explicit EvaluateVisitor(std::unordered_map<std::string, int> env) 
        : env_(std::move(env)) {}

    // 对外暴露获取最终结果的接口
    int get_result() const { 
        return current_result_; 
    }

    void visit(VarNode* node) override {
        // 实现叶子节点（变量）的求值：去 env_ 里查变量的值，然后把它放进“寄存器” current_result_ 中
        // at 的作用是：用键（Key）去查找对应的值（Value）
        current_result_ = env_.at(node->name); // （为防崩溃，可以简单假设变量一定存在）
    }

    void visit(AddNode* node) override {
        // 实现加法节点的求值：需要左子树的值，和右子树的值，然后相加
        node->lhs->accept(this); // 递归访问左子树
        int left_value = current_result_; // 备份左子树的结果

        node->rhs->accept(this); // 递归访问右子树
        int right_value = current_result_; // 备份右子树的结果

        // 相加并写回寄存器
        current_result_ = left_value + right_value; 
    }
};

int main() {
    // 构造一棵 IR 树：(x + y)
    // 使用 std::make_unique 避免裸指针 new，确保这棵树在离开 main 作用域时，能沿着枝干从上到下安全、级联地析构，保证内存不会泄露
    auto x = std::unique_ptr<VarNode>(new VarNode("x"));
    auto y = std::unique_ptr<VarNode>(new VarNode("y"));
    auto tree = std::unique_ptr<AddNode>(new AddNode(std::move(x), std::move(y)));

    // 实例化我们的 Compiler Pass
    PrintVisitor printer;
    
    // 启动：把 Pass 喂给 AST 的根节点
    std::cout << "IR Tree: ";
    tree->accept(&printer);
    std::cout << std::endl;

    std::unordered_map<std::string, int> env = {{"x", 10}, {"y", 20}};
    EvaluateVisitor evaluator(env);
    tree->accept(&evaluator);
    std::cout << "Evaluate Result: " << evaluator.get_result() << std::endl;

    return 0;
}