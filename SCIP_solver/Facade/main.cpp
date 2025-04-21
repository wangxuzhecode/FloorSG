#include "base.h"
#include "../basic/Config.h"
#include <glog/logging.h>
#include "method_common.h"
#include "../math/linear_program.h"
#include "../math/linear_program_solver.h"
#include <malloc.h>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_conformer_2.h>
#include <CGAL/Partition_traits_2.h>
#include <CGAL/partition_2.h>



typedef CGAL::Exact_predicates_tag                               Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, CGAL::Default, Itag> CDT;
typedef CGAL::Partition_traits_2<K>                         Traits;
typedef Traits::Point_2                                     Pointp2;
typedef Traits::Polygon_2                                   Polygon2;
typedef std::list<Polygon2>     Polygon_list;

int judge_linear(std::vector<std::vector<float>> lines, int i, int j)
{
    std::vector<float> v1 = lines[i];
    std::vector<float> v2 = lines[j];
    float dx1 = v1[2] - v1[0];
    float dy1 = v1[3] - v1[1];
    float dx2 = v2[2] - v2[0];
    float dy2 = v2[3] - v2[1];

    float d1 = dy1 * dx2;
    float d2 = dy2 * dx1;

    if (fabs(d1 - d2) < 0.05)
    {
        if ((v1[0] == v2[0] && v1[1] == v2[1]) || (v1[0] == v2[2] && v1[1] == v2[3]) || (v1[2] == v2[0] && v1[3] == v2[1]) || (v1[2] == v2[2] && v1[3] == v2[3]))
        {
            return 1;
        }
    }
    return 0;
}

void read_input(std::string filepath, std::vector<int>& support_num, std::vector<float>& uncovered_length, std::vector<std::vector<int>>& interpoint_list, std::vector<std::vector<float>>& lines, int& num_points, float& bbox_length)
{
    std::string support_num_path = filepath + std::string("support_num.txt");
    std::string uncovered_length_path = filepath + "uncovered_length.txt";
    std::string interpoint_list_path = filepath + "interpoint_list.txt";
    std::string lines_path = filepath + "line.txt";
    std::string num_points_path = filepath + "num_points.txt";
    std::string bbox_length_path = filepath + "bbox_length.txt";

    std::ifstream support_num_file(support_num_path);
    std::ifstream uncovered_length_file(uncovered_length_path);
    std::ifstream interpoint_list_file(interpoint_list_path);
    std::ifstream lines_file(lines_path);
    std::ifstream num_points_file(num_points_path);
    std::ifstream bbox_length_file(bbox_length_path);
    int tmp;
    float tmp1;
    while (support_num_file >> tmp)
        support_num.push_back(tmp);

    while (num_points_file >> tmp)
        num_points = tmp;
    while (bbox_length_file >> tmp1)
        bbox_length = tmp1;

    while (uncovered_length_file >> tmp1)
        uncovered_length.push_back(tmp1);
    std::string line;
    while (std::getline(interpoint_list_file, line))
    {
        std::vector<int> row;
        std::stringstream ss(line);
        int value;
        while (ss >> value)
            row.push_back(value);
        interpoint_list.push_back(row);
    }
    while (std::getline(lines_file, line))
    {
        std::vector<float> row;
        std::stringstream ss(line);
        float value;
        while (ss >> value)
            row.push_back(value);
        lines.push_back(row);
    }

    for (const auto& row : lines) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

}

void save_result(std::vector<std::vector<float>> lines, std::vector<int> selected_idx, std::string output_path)
{
    std::ofstream outFile(output_path);

    if (!outFile) {
        std::cerr << "无法打开文件!" << std::endl;
    }

    for (int j = 0; j < selected_idx.size(); j++)
    {
        int idx = selected_idx[j];
        for (int i = 0; i < lines[idx].size(); ++i)
        {
            outFile << lines[idx][i];
            if (i < lines[idx].size() - 1)
                outFile << " ";
        }
        outFile << std::endl;
    }

    outFile.close();
}

std::vector<int> process_roomwise(std::string output_path, std::vector<int>& support_num, std::vector<float>& uncovered_length, std::vector<std::vector<int>>& interpoint_list, std::vector<std::vector<float>>& lines, int& total_points, float& bbox_length)
{
    double lambda_data_fitting = 0.8;
    double lambda_model_coverage = 0.1;
    double lambda_model_complexity = 0.1;

    LinearProgramSolver::SolverName solver_name = LinearProgramSolver::SCIP; // scip 
    LinearProgram program_;

    double coeff_data_fitting = lambda_data_fitting;
    double coeff_coverage = total_points * lambda_model_coverage / bbox_length;
    double coeff_complexity = total_points * lambda_model_complexity / double(interpoint_list.size());

    int num_segments = lines.size();
    int num_points = interpoint_list.size();
    int num_sharp_points = interpoint_list.size();
    int total_variables = num_segments + num_points + num_sharp_points;
    std::cout << total_variables << std::endl;
    program_.clear();
    LinearObjective* objective = program_.create_objective(LinearObjective::MINIMIZE);

    const vector<Variable*>& variables = program_.create_n_variables(total_variables);
    for (size_t i = 0; i < total_variables; ++i)
    {
        Variable* v = variables[i];
        v->set_variable_type(Variable::BINARY);
    }

    for (int i = 0; i < num_segments; i++)
    {
        if (i > total_variables)
        {
            LOG(INFO) << "Error: variables undefine.";
            std::cout << "error" << std::endl;
        }
        objective->add_coefficient(i, -coeff_data_fitting * support_num[i]); // accumulate data fitting term
        objective->add_coefficient(i, coeff_coverage * uncovered_length[i]); // accumulate model coverage term
        //std::cout << -coeff_data_fitting * support_num[i] << std::endl;
        //std::cout << coeff_coverage * uncovered_length[i] << std::endl;
    }
    //std::cout << "complexity" << std::endl;
    for (int i = num_segments + num_points; i < total_variables; i++)
    {
        if (i > total_variables)
        {
            LOG(INFO) << "Error: variables undefine.";
            std::cout << "error" << std::endl;

        }
        objective->add_coefficient(i, coeff_complexity); // accumulate model complexity term
        //std::cout << coeff_complexity << std::endl;
    }

    //for (int i = 0; i < interpoint_list.size(); i++)
    //{
    //    LinearConstraint* c1 = program_.create_constraint();
    //    LinearConstraint* c2 = program_.create_constraint();
    //    for (int j = 0; j < interpoint_list[i].size(); j++) {
    //        if (interpoint_list[i][j] > total_variables) {
    //            LOG(INFO) << "Error: variables undefine.";
    //            std::cout << "error" << std::endl;
    //        }
    //        c1->add_coefficient(interpoint_list[i][j], 1.0);
    //        c2->add_coefficient(interpoint_list[i][j], -1.0);
    //    }
    //    if (num_segments + i > total_variables) {
    //        LOG(INFO) << "Error: variables undefine.";
    //        std::cout << "error" << std::endl;
    //    }
    //    c1->add_coefficient(num_segments + i, -2.0);
    //    c1->set_bound(LinearConstraint::LOWER, 0.0);
    //    c2->add_coefficient(num_segments + i, 2.0);
    //    c2->set_bound(LinearConstraint::LOWER, 0.0);
    //}

    //for (int i = 0; i < interpoint_list.size(); i++) {  // 遍历每一个点
    //    // 创建一个新的约束，用于表示当前点的线段数为 0 或 2
    //    LinearConstraint* c1 = program_.create_constraint();  // 约束 c1：度数下界
    //    LinearConstraint* c2 = program_.create_constraint();  // 约束 c2：度数上界
    //    LinearConstraint* c3 = program_.create_constraint();  // 约束 c3：关联线段数是 0 或 2

    //    double segment_count = 0;  // 记录当前点的关联线段数

    //    for (int j = 0; j < interpoint_list[i].size(); j++) {  // 遍历当前点关联的所有线段

    //        c1->add_coefficient(interpoint_list[i][j], 1.0);  // 线段的ID添加到约束 c1 中，系数为 1.0
    //        c2->add_coefficient(interpoint_list[i][j], 1.0);  // 线段的ID添加到约束 c2 中，系数为 1.0
    //    }

    //    // 确保每个点的度数不超过 2
    //    c1->add_coefficient(num_segments + i, -2.0);  // 在约束 c1 中加入 num_segments + i 变量，系数为 -2.0
    //    c1->set_bound(LinearConstraint::LOWER, 0.0);  // c1 的下界为 0.0，表示度数不能小于 0

    //    // 确保每个点的度数不超过 2
    //    c2->add_coefficient(num_segments + i, 2.0);  // 在约束 c2 中加入 num_segments + i 变量，系数为 4.0
    //    c2->set_bound(LinearConstraint::LOWER, 0.0);  // c2 的下界为 0.0，表示度数不能小于 0

    //    // 约束 c3 负责确保每个点的关联线段数是 0 或 2
    //    c3->add_coefficient(num_segments + i, 2.0);  // 将变量 num_segments + i 添加到约束 c3 中，系数为 2
    //    c3->set_bound(LinearConstraint::LOWER, 0.0);  // c3 的下界为 0.0，表示度数不能小于 0

    //    // 这里还需要添加额外的约束来确保该点的关联线段数是 0 或 2
    //    // 比如使用一些技术来确保精确匹配
    //    // 使用线性规划模型中的整数约束或通过增加额外的二进制变量来表示线段的数量限制
    //}

    for (int i = 0; i < interpoint_list.size(); i++)
    {
        LinearConstraint* c1 = program_.create_constraint(LinearConstraint::FIXED, 0.0, 0.0);
        for (int j = 0; j < interpoint_list[i].size(); j++) {

            c1->add_coefficient(interpoint_list[i][j], 1.0);
        }
        if (num_segments + num_points + i > total_variables) {
            LOG(INFO) << "Error: variables undefine.";
        }
        c1->add_coefficient(num_segments + i, -2.0);

    }


    //double m = 1.0;
    //for (int i = 0; i < interpoint_list.size(); i++)
    //{
    //    // if a point is sharp, the point must be selected first:
    //    // x[var_point_usage_idx] >= x[var_point_sharp_idx]
    //    LinearConstraint* c = program_.create_constraint();
    //    int var_point_usage_idx = num_segments + i;
    //    int var_point_sharp_idx = num_segments + num_points + i;
    //    if (var_point_usage_idx > total_variables || var_point_sharp_idx > total_variables) {
    //        std::cout << "error" << std::endl;
    //    }

    //    c->add_coefficient(var_point_usage_idx, 1.0);
    //    c->add_coefficient(var_point_sharp_idx, -1.0);
    //    c->set_bound(LinearConstraint::LOWER, 0.0);

    //    for (int j = 0; j < interpoint_list[i].size(); j++) {
    //        int s1 = interpoint_list[i][j];
    //        for (int k = j + 1; k < interpoint_list[i].size(); k++) {
    //            int s2 = interpoint_list[i][k];
    //            if (judge_linear(lines, interpoint_list[i][j], interpoint_list[i][k])==0) { // non-colinear
    //                // the constraint is:
    //                //x[var_point_sharp_idx] + m * (3 - (x[s1] + x[s2] + x[var_point_usage_idx])) >= 1
    //                c = program_.create_constraint();
    //                if (s1 > total_variables || s2 > total_variables) {
    //                    std::cout << "error" << std::endl;

    //                }
    //                c->add_coefficient(var_point_sharp_idx, 1.0);
    //                c->add_coefficient(s1, -m);
    //                c->add_coefficient(s2, -m);
    //                c->add_coefficient(var_point_usage_idx, -m);
    //                c->set_bound(LinearConstraint::LOWER, 1.0 - 3.0 * m);
    //            }
    //        }
    //    }
    //}

    LOG(INFO) << "#Total constraints: " << program_.constraints().size();

    // optimize model
    LOG(INFO) << "Solving the binary program. Please wait...";

    LinearProgramSolver solver;
    std::vector<int> selected_idx;
    if (solver.solve(&program_, solver_name))
    {
        LOG(INFO) << "Solving the binary program done. ";
        //std::cout << "Solving the binary program done." << std::end;
        vector<int> index;
        vector<Point_2> l;
        vector<Point_2> r;
        vector<Segment> segs;
        // mark result
        const vector<double>& X = solver.solution();
        for (int i = 0; i < lines.size(); i++)
        {
            //std::cout << X[i] << std::endl;
            if (static_cast<int>(std::round(X[i])) == 0)
                continue;
            else
            {
                std::cout << i << std::endl;
                selected_idx.push_back(i);
                //Point_2 s, t;
                //s = segments[i].segment2.source();
                //t = segments[i].segment2.target();
                //if (s.x() > t.x() || s.x() == t.x() && s.y() > t.y())
                //    segs.push_back(Segment(Segment_2(t, s), segments[i].ID));
                //else
                //    segs.push_back(Segment(segments[i].segment2, segments[i].ID));
            }
        }
        LOG(INFO) << " segments are selected.";
    }
    return selected_idx;
}

int main(int argc, char* argv[])
{
	
    std::string inputdir = "/mnt/hdd1/wxz/data/SCIP_data/S3DIS/area2/";
    std::string outputdir = "/mnt/hdd1/wxz/data/SCIP_data/output/S3DIS_area2/";
    for (int i = 42; i <=42; i++)
    {
        std::stringstream ss;
        ss << i;
        std::string str = ss.str();
        std::cout << str;
        std::string inputpath = inputdir + str + "/";

        std::vector<int> vec;
        std::vector<int> support_num;
        std::vector<float> uncovered_length;
        std::vector<std::vector<int>> interpoint_list;
        std::vector<std::vector<float>> lines;
        int total_points;
        float bbox_length;
        read_input(inputpath, support_num, uncovered_length, interpoint_list, lines, total_points, bbox_length);
        std::string output_path = outputdir + str + ".txt";
        std::vector<int> selected_idx = process_roomwise(output_path, support_num, uncovered_length, interpoint_list, lines, total_points, bbox_length);
        std::cout << output_path << std::endl;
        save_result(lines, selected_idx, output_path);
    }


    return 0;
}
