#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <algorithm>
#include<iostream>
#include<fstream>
#include<vector>

#define MAX_STRING 100

using namespace std;

static int (*info)(const char *fmt,...) = &printf;

char predict_result_file[MAX_STRING], origin_data_file[MAX_STRING];
int pf = -1, lf = -1, olf = -1;

vector<string> sepstr(const string &sStr, const string &sSep, bool withEmpty=false)
{
    vector<string> vt;

    string::size_type pos = 0;
    string::size_type pos1 = 0;

    while(true)
    {
        string s;
        pos1 = sStr.find_first_of(sSep, pos);
        if(pos1 == string::npos)
        {
            if(pos + 1 <= sStr.length())
            {
                s = sStr.substr(pos);
            }
        }
        else if(pos1 == pos)
        {
            s = "";
        }
        else
        {
            s = sStr.substr(pos, pos1 - pos);
            pos = pos1;
        }

        if(withEmpty)
        {
            vt.push_back(s);
        }
        else
        {
            if(!s.empty())
            {
                vt.push_back(s);
            }
        }

        if(pos1 == string::npos)
        {
            break;
        }

        pos++;
    }

    return vt;
}

struct Pred
{
    int real;       //real label, 0:negative sample, 1:positive sample
    double p;        //probability of prediction

    Pred():real(0),p(0) {}
};

typedef std::vector<Pred> PredArray;

bool pred_cmp(const Pred& a, const Pred& b)
{
    return a.p < b.p;
}

double fast_auc(PredArray& pred_arr)
{
    if (pred_arr.empty()) return 0.0;
    std::stable_sort(pred_arr.begin(), pred_arr.end(), pred_cmp);
    int nfalse = 0;
    double auc = 0.0;
    for (size_t i = 0; i < pred_arr.size(); i++)
    {
        int y = pred_arr[i].real;
        nfalse += (1 - y);
        auc += y * nfalse;
    }
    auc /= (nfalse * double((pred_arr.size() - nfalse)));
    return auc;
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int computeAuc()
{
    int correct = 0;
    int tp = 0, fp = 0, tn = 0, fn = 0;
    int total = 0;
    double error = 0;
    double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

    PredArray predArr;

    std::vector<int> realLabels;
    //use label field in prediction result file by first
    if (origin_data_file[0] != 0 && olf >= 0 && lf < 0)
    {
        ifstream ifile;
        ifile.open(origin_data_file);
        if (!ifile.is_open())
        {
            cerr << "can not open file[" << origin_data_file << "]" << endl;
            return -1;
        }

        int64_t lineNum = 0;
        string buffer;
        while (ifile.good() && getline(ifile, buffer))
        {
            vector<string> segs = sepstr(buffer, " \t");
            if ((int)segs.size() > olf)
            {
                realLabels.push_back(atoi(segs[olf].c_str()));
            }
            ++lineNum;
        }
        cout<<"Origin data file lines:"<<lineNum
            <<",real labels size:"<<realLabels.size()<<endl;
        ifile.close();
    }

    ifstream ifile;
    ifile.open(predict_result_file);
    if (!ifile.is_open())
    {
        cerr << "can not open file[" << predict_result_file << "]" << endl;
        return -1;
    }

    std::vector<double> predicts;
    int64_t lineNum = 0;
    string buffer;
    while (ifile.good() && getline(ifile, buffer))
    {
        vector<string> segs = sepstr(buffer, " \t");
        //if there is no prediction result for some samples
        if ((int)segs.size() <= pf)
        {
            cerr<<"probility field cannot be found, line:"<<lineNum<<endl;
            return -1;
        }
        predicts.push_back(atof(segs[pf].c_str()));
        if (lf >= 0 && (int)segs.size() > lf)
        {
            realLabels.push_back(atoi(segs[lf].c_str()));
        }
        ++lineNum;
    }

    if (realLabels.size() != predicts.size())
    {
        cerr<<"label size is not equal to predict size:"<<realLabels.size()
            <<"!="<<predicts.size()<<endl;
        return -1;
    }

    for (size_t i = 0; i < predicts.size(); i++)
    {
        double p = predicts[i];
        int target_label = (realLabels[i] == 1 ? 1 : 0);
        int predict_label = (p > 0.5 ? 1 : 0);

        Pred pred;
        pred.real = (target_label==1 ? 1 : 0);
        pred.p = p;
        predArr.push_back(pred);

        if (predict_label == target_label)
        {
            ++correct;
            if (target_label == 1)
                tp++;
            else
                tn++;
        }
        else
        {
            if (target_label == 1)
                fn++;
            else
                fp++;
        }
        error += (predict_label-target_label)*(predict_label-target_label);
        sump += predict_label;
        sumt += target_label;
        sumpp += predict_label*predict_label;
        sumtt += target_label*target_label;
        sumpt += predict_label*target_label;
        ++total;
    }

    info("Accuracy = %g%% (%d/%d)\n",(double) correct/total*100,correct,total);
    info("TP = %d, TN = %d, FP = %d, FN = %d\n",tp, tn, fp, fn);
    double precision = (double)100*tp/(tp+fp);
    double recall = (double)100*tp/(tp+fn);
    double f1score = 2 * precision * recall / (precision + recall);
    info("Precision = %g%%\n", precision);
    info("Recall = %g%%\n", recall);
    info("F1-score = %g%%\n", f1score);
    if (predArr.size())
    {
        info("AUC = %g\n",fast_auc(predArr));
    }

    return 0;
}

int main(int argc, char **argv)
{
    int i;
    if (argc == 1)
    {
        printf("AUC fast compute toolkit v1.0\n\n");
        printf("Options:\n");
        printf("\t-result <file>\n");
        printf("\t\tPredict result file\n");
        printf("\t-pf <int>\n");
        printf("\t\tThe predict probability field index of line in the result file\n");
        printf("\t-lf <int>\n");
        printf("\t\tThe real label field index of line in the result file; if not has, do not set and keep default -1\n");
        printf("\t-origin_data <file>\n");
        printf("\t\tThe origin data file before prediction\n");
        printf("\t-olf <int>\n");
        printf("\t\tThe real label field index of line in the origin data file\n");
        printf("\nExamples:\n");
        printf("./compute_auc -result predict_result.txt -pf 0 -lf 1\n");
        printf("./compute_auc -result predict_result.txt -pf 0 -origin_data test_data.txt -olf 0\n");
        return 0;
    }

    predict_result_file[0] = 0;
    origin_data_file[0] = 0;
    if ((i = ArgPos((char *)"-result", argc, argv)) > 0) strcpy(predict_result_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-pf", argc, argv)) > 0) pf = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-lf", argc, argv)) > 0) lf = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-origin_data", argc, argv)) > 0) strcpy(origin_data_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-olf", argc, argv)) > 0) olf = atoi(argv[i + 1]);

    computeAuc();

    return 0;
}
