/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
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

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadFeatures(string basedir,vector<vector<vector<double> > > &features,int NIMAGES);
void changeStructure(const vector<double> &plain, vector<vector<double> > &out,
                     int L);
CnnVocabulary testVocCreation(const vector<vector<vector<double> > > &features,const string vocfilename);
void testDatabase(const vector<vector<vector<double> > > &features,CnnVocabulary voc);
void loadFeaturesFromMat(string filename,vector<vector<double> > &features);
void buildDatabase(const vector<vector<vector<double> > > &features,CnnDatabase &db);
void queryDatabase(const vector<vector<vector<double> > > &features,CnnDatabase& db,ofstream &out);
void buildVoc(const string vocfilename);
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
//const int NIMAGES = 4022;

// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
    cout << endl << "Press enter to continue" << endl;
    getchar();
}

// ----------------------------------------------------------------------------

int main()
{
    const string vocFile = "//home//develop//Work//Source_Code//DBoW2//vprice_cnn_voc_K10L4.txt";
    //buildVoc(vocFile);
        string ref_basedir = "/home/develop/Work/Datasets/GardensPointWalking/day_left/Vgg_LCF_conv5_1";//"/home/develop/Work/Datasets/GardensPointWalking/day_left/Vgg_LCF_conv5_1";
    string query_basedir = "/home/develop/Work/Datasets/GardensPointWalking/night_right/Vgg_LCF_conv5_1";

    int n_ref_imgs = 200,n_query_imgs = 200;

    vector<vector<vector<double> > > ref_features,query_features,features;
    loadFeatures(ref_basedir,ref_features,n_ref_imgs);
    loadFeatures(query_basedir,query_features,n_query_imgs);

    CnnVocabulary voc;
    cout<<"Loading Vocabulary ...\n";
    voc.loadFromTextFile(vocFile);
    cout<<"Voc Info:\n";
    cout<<voc<<endl;
    wait();
    CnnDatabase db(voc, false, 0);
    buildDatabase(ref_features,db);
    ofstream query_results_ostream;
    queryDatabase(query_features,db,query_results_ostream);
    //query_results_ostream.close();
    return 0;
}

// ----------------------------------------------------------------------------
void loadFeatures(string basedir,vector<vector<vector<double> > > &features,int NIMAGES)
{
    features.clear();
    features.reserve(NIMAGES);

    //cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
    char img_name[100];
    cout << "Extracting CNN features..." << endl;
    for(int i = 0; i < NIMAGES; ++i)
    {
        stringstream ss;
        sprintf( img_name, "Image%03d", i );
        //sprintf( img_name, "image-%05d", i );
        //ss << basedir<< "/" << img_name << ".conv3.grp.fv.mat";
        ss << basedir<< "/" << img_name << ".conv5.fv.mat";
        cout<<ss.str()<<endl;
        vector<vector<double> > fv;
        loadFeaturesFromMat(ss.str(),fv);
        if (fv.size() > 0)
            features.push_back(fv);
    }
}

// ----------------------------------------------------------------------------

void changeStructure(const vector<double> &plain, vector<vector<double> > &out,
                     int L)
{
    out.resize(plain.size() / L);

    unsigned int j = 0;
    for(unsigned int i = 0; i < plain.size(); i += L, ++j)
    {
        out[j].resize(L);
        std::copy(plain.begin() + i, plain.begin() + i + L, out[j].begin());
    }
}

// ----------------------------------------------------------------------------
CnnVocabulary testVocCreation(const vector<vector<vector<double> > > &features,const string vocfilename)
{
    // branching factor and depth levels
    const int k = 10;
    const int L = 4;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L2_NORM;

    CnnVocabulary voc(k, L, weight, score);

    cout << "Creating a " << k << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;

    cout << "Vocabulary information: " << endl
         << voc << endl << endl;

    // lets do something with this vocabulary
//    cout << "Matching images against themselves (0 low, 1 high): " << endl;
//    BowVector v1, v2;
//    for(int i = 0; i < NIMAGES; i++)
//    {
//        voc.transform(features[i], v1);
//        for(int j = 0; j < NIMAGES; j++)
//        {
//            voc.transform(features[j], v2);

//            double score = voc.score(v1, v2);
//            cout << "Image " << i << " vs Image " << j << ": " << score << endl;
//        }
//    }

    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    voc.saveToTextFile(vocfilename);
    cout << "Done" << endl;
    return voc;
}
// ----------------------------------------------------------------------------

void testDatabase(const vector<vector<vector<double> > > &features,CnnVocabulary voc)
{
    cout << "Creating a small database..." << endl;

    // load the vocabulary from disk
    //Surf64Vocabulary voc("small_voc.yml.gz");
    //CnnVocabulary voc("small_cnn_voc.yml.gz");
    //Surf64Database db(voc, false, 0); // false = do not use direct index
    CnnDatabase db(voc, false, 0); // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that
    // belong to some vocabulary node.
    // db creates a copy of the vocabulary, we may get rid of "voc" now

    // add images to the database
    int nfeatures = features.size();
    for(int i = 0; i < nfeatures; i++)
    {
        db.add(features[i]);
    }

    cout << "... done!" << endl;

    cout << "Database information: " << endl << db << endl;

    // and query the database
    cout << "Querying the database: " << endl;

    QueryResults ret;
    for(int i = 0; i < nfeatures; i++)
    {
        db.query(features[i], ret, 4);

        // ret[0] is always the same image in this case, because we added it to the
        // database. ret[1] is the second best match.

        cout << "Searching for Image " << i << ". " << ret << endl;
    }

    cout << endl;

    // we can save the database. The created file includes the vocabulary
    // and the entries added
    cout << "Saving database..." << endl;
    //db.save("small_db.yml.gz");
    cout << "... done!" << endl;

    // once saved, we can load it again
    //cout << "Retrieving database once again..." << endl;
    //Surf64Database db2("small_db.yml.gz");
    //CnnDatabase db2("small_db.yml.gz");
    //cout << "... done! This is: " << endl << db2 << endl;
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
//-------------------------------------------------------------------------------
void buildDatabase(const vector<vector<vector<double> > > &features,CnnDatabase &db)
{
    cout << "Creating a database..." << endl;

    //CnnDatabase db(voc, false, 0); // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that
    // belong to some vocabulary node.
    // db creates a copy of the vocabulary, we may get rid of "voc" now

    // add images to the database
    int nfeatures = features.size();
    for(int i = 0; i < nfeatures; i++)
    {
        db.add(features[i]);
    }

    cout << "... done!" << endl;

    cout << "Database information: " << endl << db << endl;
}
//-------------------------------------------------------------------------------
void queryDatabase(const vector<vector<vector<double> > > &features,CnnDatabase &db,ofstream &out)
{
    cout << "Querying the database: " << endl;
    int rangeSz = 5;
    QueryResults ret;
    int tp = 0,fp=0,fn = 0;
    int nfeatures = features.size();
    for(int i = 0; i < nfeatures; i++)
    {
        db.query(features[i], ret, 5);

        // ret[0] is always the same image in this case, because we added it to the
        // database. ret[1] is the second best match.

        cout << "Searching for Image " << i << ". " << ret << endl;
        //out << "Searching for Image " << i << ". " << ret << endl;
        bool found = false;
        for(int n=0;n<ret.size();++n)
        {
            int entID = ret[n].Id;
            if(i -rangeSz <= entID && entID <= i+rangeSz){
                 ++ tp;
                found = true;
                break;
            }
        }
        if(!found){
            ++ fp;
            ++ fn;
        }
    }

    cout << endl;
    cout<<"============================\n";
    cout<<"Percision: "<<tp*100.0 / (fp + tp)<<endl;
}

//------------------------------------------------------------------------------
void buildVoc(const string vocFile)
{
    string ref_basedir = "/home/develop/Work/Datasets/vprice_live_vggfeatures";//"/home/develop/Work/Datasets/GardensPointWalking/day_left/Vgg_LCF_conv5_1";
    string query_basedir = "/home/develop/Work/Datasets/vprice_memory_vggfeatures";

    int n_ref_imgs = 4022,n_query_imgs = 3756;

    vector<vector<vector<double> > > ref_features,query_features,features;
    loadFeatures(ref_basedir,ref_features,n_ref_imgs);
    //loadFeatures(query_basedir,query_features,n_query_imgs);
    //features = ref_features;
    //features.insert(features.end(),query_features.begin(),query_features.end());
    //ref_features.clear();
    //query_features.clear();
    //CnnVocabulary voc = testVocCreation(ref_features);
    //cout<<ref_features.size()<<' ' << ref_features[0].size()<<endl;
    //cout<<query_features.size()<<' ' << query_features[0].size()<<endl;
    //cout<<features.size()<<' ' << features[0].size()<<endl;
    CnnVocabulary voc = testVocCreation(ref_features,vocFile);
}
