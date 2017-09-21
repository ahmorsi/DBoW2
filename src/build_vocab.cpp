#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
// DBoW2
#include "DBoW2.h" // defines Surf64Vocabulary and Surf64Database

#include <DUtils/DUtils.h>
#include <DVision/DVision.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <matio.h>

using namespace DBoW2;
using namespace DUtils;
using namespace std;

void generateVocab(vector<std::string> features_dir,std::string vocfilename);
void loadFeatures(std::string basedir,std::vector<std::vector<std::vector<double> > > &features);
void loadFeaturesFromMat(string filename,vector<vector<double> > &features);
int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printf("Usage: ./build_vocab <vocfilename> <features_dir1> <features_dir2> <features_dir3> ..... <features_dirN>\n");
        return -1;
    }
    std::string vocfilename = argv[1];
    char** features_dir_arr=argv + 2;
    std::vector<string> features_dir(features_dir_arr,features_dir_arr + ((argc -2)/sizeof(char)));
    generateVocab(features_dir,vocfilename);
    return 0;
}
void generateVocab(std::vector<std::string> features_dir,std::string vocfilename)
{
    std::vector<std::vector<std::vector<double> > > temp_features,features;
    for(int i=0;i<features_dir.size();++i)
    {
        loadFeatures(features_dir[i],temp_features);
        features.reserve(features.size()+temp_features.size());
        features.insert(features.end(),temp_features.begin(),temp_features.end());
    }
    // branching factor and depth levels
    const int k = 10;
    const int L = 6;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;

    CnnVocabulary voc(k, L, weight, score);

    cout << "Creating a " << k << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;

    cout << "Vocabulary information: " << endl
         << voc << endl << endl;
    // save the vocabulary to disk
    cout << endl << "Saving vocabulary in " << vocfilename<< endl;
    voc.saveToTextFile(vocfilename);
    cout << "Done" << endl;
}
// ----------------------------------------------------------------------------
void loadFeatures(string basedir,vector<vector<vector<double> > > &features)
{
    features.clear();
    cout << "Extracting CNN features from ... "<<basedir << endl;
    std::vector<std::string> feat_files = DUtils::FileFunctions::Dir(basedir.c_str(),".fv.mat",true);
    string path = "";
    features.reserve(feat_files.size());
    for(int i = 0; i < feat_files.size(); ++i)
    {
        path = feat_files[i];
        cout<<path<<endl;
        vector<vector<double> > fv;
        loadFeaturesFromMat(path.c_str(),fv);
        if (fv.size() > 0)
            features.push_back(fv);
    }
}
// ----------------------------------------------------------------------------
void loadFeaturesFromMat(string filename,vector<vector<double> > &features)
{
    mat_t *mat = Mat_Open(filename.c_str(),MAT_ACC_RDONLY);
    if(mat){
         matvar_t *matVar=0 ;
        matVar = Mat_VarRead(mat,(char*)"fv");

        if(matVar)
        {
            unsigned xSize = matVar->nbytes/matVar->data_size ;
//            cout<<xSize<<endl;
            const double *xData = static_cast<const double*>(matVar->data) ;
            int height = matVar->dims[0];
            int width = matVar->dims[1];
            int idx = 0;
            cout<<height << " " << width<<endl;
//            for(int i=0;i< height; ++i){
//                    idx = i*width;
//                    const double* arr = xData + idx;
//                    //cout<<"Vector of row no. "<<idx<<endl;
//                    //cout<<idx<<' '<<idx + width<< ' '<<xSize<<endl;
//                    vector<double> fv;
//                    for(int j=idx;j<idx+width;++j){
//                        //cout<<xData[j]<<endl;
//                        fv.push_back((double)xData[j]);
//                    }
//                    features.push_back(fv);
//                }
            vector<double> fv;
            for(int i=0; i<xSize; ++i)
            {
                //std::cout<<"\tx["<<i<<"] = "<<xData[i]<<"\n" ;
                fv.push_back(xData[i]);
                if(fv.size() == width){
                    features.push_back(fv);
                    fv.clear();
                }
            }

//            for(int i=0; i<matVar->rank; ++i)
//            {
//                std::cout<<"\tdim["<<i<<"] == "<<matVar->dims[i]<<"\n" ;
//            }
        }
    }
    Mat_Close(mat);
}
