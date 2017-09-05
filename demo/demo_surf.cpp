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
void testVocCreation(const vector<vector<vector<float> > > &features,bool save_file=true);
void testDatabase(const vector<vector<vector<float> > > &features);
void buildDatabase(const vector<vector<vector<float> > > &features);
void queryDatabase(const vector<vector<vector<float> > > &features);

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

  if (argc != 3)
  {
      printf("Usage: ./demo_orb <ref_folder> <query_folder>\n");
      return -1;
  }
  ref_basedir = argv[1];
  query_basedir = argv[2];
  loadFeatures(ref_basedir,ref_features);
  loadFeatures(query_basedir,query_features);

  testVocCreation(ref_features);

  //wait();

  buildDatabase(ref_features);
  queryDatabase(query_features);
  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(string basedir,vector<vector<vector<float> > > &features)
{
  features.clear();
  features.reserve(NIMAGES);

  cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
  char img_name[100];
  cout << "Extracting SURF features..." << endl;
  for(int i = 1; i <= NIMAGES; ++i)
  {
    stringstream ss;
    sprintf( img_name, "Image%03d.jpg", i );
    //ss << "images/image" << i << ".png";
    ss << basedir<< "/" << img_name;
    cout << ss.str()<<endl;
    cv::Mat image = cv::imread(ss.str(), 0);
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
  BowVector v1;
  for(int i=0;i<features.size();++i)
  {
      voc.transform(features[i], v1);
      v1.saveM("/home/develop/Work/Datasets/BOW/test_surf.txt",voc.size());
      break;
  }
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

void buildDatabase(const vector<vector<vector<float> > > &features)
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

    // we can save the database. The created file includes the vocabulary
    // and the entries added
    cout << "Saving database..." << endl;
    db.save("small_db.yml.gz");
    cout << "... done!" << endl;
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