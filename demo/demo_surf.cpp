/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>

// DBoW2
#include "DBoW2.h" // defines Surf64Vocabulary and Surf64Database

#include <DUtils/DUtils.h>
#include <DVision/DVision.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>


using namespace DBoW2;
using namespace DUtils;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadFeatures(string basedir,vector<vector<vector<float> > > &features);
void changeStructure(const vector<float> &plain, vector<vector<float> > &out,
  int L);
Surf64Vocabulary createVocab(const vector<vector<vector<float> > > &features,const string vocfilename,bool save_file=true);
void testVocCreation(const vector<vector<vector<float> > > &features,bool save_file=true);
void testDatabase(const vector<vector<vector<float> > > &features);
//void buildDatabase(const vector<vector<vector<float> > > &features);
void buildDatabase(const vector<vector<vector<float> > > &features,Surf64Database &db);
void queryDatabase(const vector<vector<vector<float> > > &features);
void queryDatabase(const vector<vector<vector<float> > > &features,Surf64Database &db,map<int,vector<int> >& correspondances);
void read_correspondances(string frames_correspondances_file,map<int,vector<int> >& correspondances);

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 400;

// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

// ----------------------------------------------------------------------------

int main(int argc, char* argv[])
{
  vector<vector<vector<float> > > ref_features,query_features;
  string ref_basedir = "/home/develop/Work/Datasets/GardensPointWalking/day_right";
  string query_basedir = "/home/develop/Work/Datasets/GardensPointWalking/night_right";

  if (argc < 3)
  {
      printf("Usage: ./demo_surf <ref_folder> <query_folder> [file_correspondances] \n");
      return -1;
  }
  const string vocFile = "//home//develop//Work//Source_Code//DBoW2//surf_voc_K10L6.txt";
  ref_basedir = argv[1];
  query_basedir = argv[2];
  loadFeatures(ref_basedir,ref_features);
  loadFeatures(query_basedir,query_features);

  //testVocCreation(ref_features);
  Surf64Vocabulary voc = createVocab(ref_features,vocFile,false);
  //wait();
  Surf64Database db(voc, false, 0);
  buildDatabase(ref_features,db);
  ref_features.clear();
  //buildDatabase(ref_features);
  if(argc > 3)
  {
     //map<int,int> correspondances;
     map<int,vector<int> > correspondances;
     string frames_correspondances_file = argv[3];
     read_correspondances(frames_correspondances_file,correspondances);
     queryDatabase(query_features,db,correspondances);
  }
  else
    queryDatabase(query_features);
  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(string basedir,vector<vector<vector<float> > > &features)
{
  cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
  features.clear();
  cout << "Extracting SURF features..." << endl;
  std::vector<std::string> feat_files = DUtils::FileFunctions::Dir(basedir.c_str(),".png",true);
  string path = "";
  features.reserve(feat_files.size());
  for(int i = 0; i < feat_files.size(); ++i)
  {
      path = feat_files[i];
      cout<<path<<endl;
      cv::Mat image = cv::imread(path, 0);
      cv::Mat mask;
      vector<cv::KeyPoint> keypoints;
      vector<float> descriptors;

      surf->detectAndCompute(image, mask, keypoints, descriptors);

      features.push_back(vector<vector<float> >());
      changeStructure(descriptors, features.back(), surf->descriptorSize());
  }
}

// ----------------------------------------------------------------------------

void changeStructure(const vector<float> &plain, vector<vector<float> > &out,
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

void testVocCreation(const vector<vector<vector<float> > > &features,bool save_file)
{
  // branching factor and depth levels 
  const int k = 10;
  const int L = 6;
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;

  Surf64Vocabulary voc(k, L, weight, score);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;
//  BowVector v1;
//  for(int i=0;i<features.size();++i)
//  {
//      voc.transform(features[i], v1);
//      v1.saveM("/home/develop/Work/Datasets/BOW/test_surf.txt",voc.size());
//      break;
//  }
//  // lets do something with this vocabulary
//  cout << "Matching images against themselves (0 low, 1 high): " << endl;
//  BowVector v1, v2;
//  for(int i = 0; i < NIMAGES; i++)
//  {
//    voc.transform(features[i], v1);
//    for(int j = 0; j < NIMAGES; j++)
//    {
//      voc.transform(features[j], v2);
      
//      double score = voc.score(v1, v2);
//      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
//    }
//  }

  if(save_file){
  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save("small_voc.yml.gz");
  }
  cout << "Done" << endl;
}
//-----------------------------------------------------------------------------
Surf64Vocabulary createVocab(const vector<vector<vector<float> > > &features,const string vocfilename,bool save_file)
{
    // branching factor and depth levels
    const int k = 10;
    const int L = 6;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;

    Surf64Vocabulary voc(k, L, weight, score);

    cout << "Creating a " << k << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;

    cout << "Vocabulary information: " << endl
         << voc << endl << endl;

    if(save_file){
    // save the vocabulary to disk
    cout << endl << "Saving vocabulary in " << vocfilename<< endl;
    voc.saveToTextFile(vocfilename);
    }
    cout << "Done" << endl;
    return voc;

}

// ----------------------------------------------------------------------------

void testDatabase(const vector<vector<vector<float> > > &features)
{
  cout << "Creating a small database..." << endl;

  // load the vocabulary from disk
  Surf64Vocabulary voc("small_voc.yml.gz");
  
  Surf64Database db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < NIMAGES; i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for(int i = 0; i < NIMAGES; i++)
  {
    db.query(features[i], ret, 5);

    // ret[0] is always the same image in this case, because we added it to the 
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save("small_db.yml.gz");
  cout << "... done!" << endl;
  
  // once saved, we can load it again  
  cout << "Retrieving database once again..." << endl;
  Surf64Database db2("small_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;
}

// ----------------------------------------------------------------------------

//void buildDatabase(const vector<vector<vector<float> > > &features)
//{
//    cout << "Creating a small database..." << endl;

//    // load the vocabulary from disk
//    Surf64Vocabulary voc("small_voc.yml.gz");

//    Surf64Database db(voc, false, 0); // false = do not use direct index
//    // (so ignore the last param)
//    // The direct index is useful if we want to retrieve the features that
//    // belong to some vocabulary node.
//    // db creates a copy of the vocabulary, we may get rid of "voc" now

//    // add images to the database
//    for(int i = 0; i < NIMAGES; i++)
//    {
//      db.add(features[i]);
//    }

//    cout << "... done!" << endl;

//    cout << "Database information: " << endl << db << endl;

//    // we can save the database. The created file includes the vocabulary
//    // and the entries added
//    cout << "Saving database..." << endl;
//    db.save("small_db.yml.gz");
//    cout << "... done!" << endl;
//}
//-------------------------------------------------------------------------
void buildDatabase(const vector<vector<vector<float> > > &features,Surf64Database &db)
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
//--------------------------------------------------------------------------
void queryDatabase(const vector<vector<vector<float> > > &features)
{
    // once saved, we can load it again
    cout << "Retrieving database once again..." << endl;
    Surf64Database db("small_db.yml.gz");
    cout << "Querying the database: " << endl;
    int rangeSz = 5;
    QueryResults ret;
    int tp = 0,fp=0,fn = 0;
    int tp_best = 0,fp_best=0,fn_best = 0;
    bool found,best_match_found;
    for(int i = 0; i < NIMAGES; i++)
    {
        db.query(features[i], ret, 5);

        // ret[0] is always the same image in this case, because we added it to the
        // database. ret[1] is the second best match.

        cout << "Searching for Image " << i << ". " << ret << endl;
        found = false;
        best_match_found = false;
        for(int n=0;n<ret.size();++n)
        {
            int entID = ret[n].Id;
            if(i -rangeSz <= entID && entID <= i+rangeSz){
                 ++ tp;
                found = true;
                if(n==0){
                    best_match_found = true;
                     ++ tp_best;
                }
                break;
            }
        }
        if(!found){
            ++ fp;
            ++ fn;
        }
        if(!best_match_found){
            ++ fp_best;
            ++ fn_best;
        }
    }

    float per_top_k = tp*100.0 / (fp + tp);
    float per_top_1 = tp_best*100.0 / (fp_best + tp_best);
    cout << endl;
    cout<<"============================\n";
    cout<<"Top K-Percision: "<<per_top_k<< " Top 1-Percision: "<<per_top_1<<endl;

}
void read_correspondances(string frames_correspondances_file,map<int,vector<int> >& correspondances)
{
//    mat_t *mat = Mat_Open(frames_correspondances_file.c_str(),MAT_ACC_RDONLY);
//    if(mat){
//         matvar_t *matVar=0 ;
//        matVar = Mat_VarRead(mat,(char*)"fm");

//        if(matVar)
//        {
//             unsigned xSize = matVar->nbytes/matVar->data_size ;
//             const double *xData = static_cast<const double*>(matVar->data) ;
//             for(int i=0; i<xSize; i+=2)
//             {
//                 correspondances.insert( {xData[i]-1,xData[i+1]-1});
//             }
//        }
//    }
//    Mat_Close(mat);
    std::ifstream infile(frames_correspondances_file);
    std::string line;
    int quId,numOfRefMatches,refId;
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        iss >> quId >> numOfRefMatches;
        vector<int> ref_ids;
        while(numOfRefMatches--){
            iss >> refId;
            ref_ids.push_back(refId);
        }
        correspondances[quId] = ref_ids;
    }
}
void queryDatabase(const vector<vector<vector<float> > > &features,Surf64Database &db,map<int,vector<int> >& correspondances)
{
    // once saved, we can load it again
    //cout << "Retrieving database once again..." << endl;
    //Surf64Database db("small_db.yml.gz");
    cout << "Querying the database: " << endl;
    int rangeSz = 3;
    QueryResults ret;
    int tp = 0,fp=0,fn = 0;
    int tp_best = 0,fp_best=0,fn_best = 0;
    bool found,best_match_found;
    vector<int> ground_truth;
    for(int i = 0; i < features.size(); i++)
    {
        db.query(features[i], ret, 5);

        // ret[0] is always the same image in this case, because we added it to the
        // database. ret[1] is the second best match.

        cout << "Searching for Image " << i << ". " << ret << endl;
        if(correspondances.find(i) == correspondances.end())
            continue;
        ground_truth = correspondances[i];
        found = false;
        best_match_found = false;
        for(int n=0;n<ret.size();++n)
        {
            int entID = ret[n].Id;
            //if(i -rangeSz <= entID && entID <= i+rangeSz){
            for(int idx=0;idx<ground_truth.size();++idx){
                if(ground_truth[idx] -rangeSz <= entID && entID <= ground_truth[idx]+rangeSz){
                ++ tp;
                found = true;
                if(n==0){
                    best_match_found = true;
                     ++ tp_best;
                }
                break;
            }
            }
            if(found)
                break;
        }
        if(!found){
            ++ fp;
            ++ fn;
        }
        if(!best_match_found){
            ++ fp_best;
            ++ fn_best;
        }
    }

    float per_top_k = tp*100.0 / (fp + tp);
    float per_top_1 = tp_best*100.0 / (fp_best + tp_best);
    cout << endl;
    cout<<"============================\n";
    cout<<"Top K-Percision: "<<per_top_k<< " Top 1-Percision: "<<per_top_1<<endl;
}
