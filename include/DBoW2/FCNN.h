#ifndef FCNN_H
#define FCNN_H

#include "FClass.h"


namespace DBoW2 {
class FCNN : protected FClass
{
public:
    /// Descriptor type
    typedef std::vector<float> TDescriptor;
    /// Pointer to a single descriptor
    typedef const TDescriptor *pDescriptor;
    /// Descriptor length
    static const int L = 4096;

    /**
     * Returns the number of dimensions of the descriptor space
     * @return dimensions
     */
    inline static int dimensions()
    {
      return L;
    }

    /**
     * Calculates the mean value of a set of descriptors
     * @param descriptors vector of pointers to descriptors
     * @param mean mean descriptor
     */
    static void meanValue(const std::vector<pDescriptor> &descriptors,
      TDescriptor &mean);

    /**
     * Calculates the (cosine) distance between two descriptors
     * @param a
     * @param b
     * @return (cosine) distance
     */
    static double distance(const TDescriptor &a, const TDescriptor &b);

    /**
     * Returns a string version of the descriptor
     * @param a descriptor
     * @return string version
     */
    static std::string toString(const TDescriptor &a);

    /**
     * Returns a descriptor from a string
     * @param a descriptor
     * @param s string version
     */
    static void fromString(TDescriptor &a, const std::string &s);

    /**
     * Returns a mat with the descriptors in float format
     * @param descriptors
     * @param mat (out) NxL 32F matrix
     */
    static void toMat32F(const std::vector<TDescriptor> &descriptors,
      cv::Mat &mat);
};
}
#endif // FCNN_H
