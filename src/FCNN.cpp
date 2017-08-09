#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include<iostream>
#include "FClass.h"
#include "FCNN.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------

void FCNN::meanValue(const std::vector<FCNN::pDescriptor> &descriptors,
  FCNN::TDescriptor &mean)
{
  mean.resize(0);
  mean.resize(FCNN::L, 0);

  double s = descriptors.size();

  vector<FCNN::pDescriptor>::const_iterator it;
  for(it = descriptors.begin(); it != descriptors.end(); ++it)
  {
    const FCNN::TDescriptor &desc = **it;
    for(int i = 0; i < FCNN::L; ++i)
    {
      mean[i] += desc[i] / s;
    }
  }

}

// --------------------------------------------------------------------------

double FCNN::distance(const FCNN::TDescriptor &a, const FCNN::TDescriptor &b)
{
//  double cosine_dist = 0.;
//  double dot_prod = 0.;
//  double l2_norm_a = 0.,l2_norm_b = 0.;
//  for(int i = 0; i < FCNN::L; ++i)
//  {
//    dot_prod += a[i]*b[i];
//    l2_norm_a += (a[i]*a[i]);
//    l2_norm_b += (b[i]*b[i]);
//  }
//  double norm_dot_prod = dot_prod / ((sqrt(l2_norm_a)*sqrt(l2_norm_b)));
//  cosine_dist = 1 - norm_dot_prod;
//  return cosine_dist;
    double sqd = 0;
    for(int i = 0; i < FCNN::L; ++i)
    {
        sqd += ((a[i]-b[i])*(a[i]-b[i]));
    }
    return sqd;
}

// --------------------------------------------------------------------------

std::string FCNN::toString(const FCNN::TDescriptor &a)
{
  stringstream ss;
  for(int i = 0; i < FCNN::L; ++i)
  {
    ss << a[i] << " ";
  }
  return ss.str();
}

// --------------------------------------------------------------------------

void FCNN::fromString(FCNN::TDescriptor &a, const std::string &s)
{
  a.resize(FCNN::L);

  stringstream ss(s);
  for(int i = 0; i < FCNN::L; ++i)
  {
    ss >> a[i];
  }
}

// --------------------------------------------------------------------------

void FCNN::toMat32F(const std::vector<TDescriptor> &descriptors,
    cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }

  const int N = descriptors.size();
  const int L = FCNN::L;

  mat.create(N, L, CV_64F);

  for(int i = 0; i < N; ++i)
  {
    const TDescriptor& desc = descriptors[i];
    double *p = mat.ptr<double>(i);
    for(int j = 0; j < L; ++j, ++p)
    {
      *p = desc[j];
    }
  }
}

// --------------------------------------------------------------------------

} // namespace DBoW2

