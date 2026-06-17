/*****************************************************************************
*
* pyobjcryst
*
* boost::python bindings to ObjCryst::PowderPatternBackgroundHist.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/args.hpp>
#include <boost/python/copy_const_reference.hpp>

#undef B0

#include <ObjCryst/ObjCryst/PowderPattern.h>

#include "helpers.hpp"

namespace bp = boost::python;
using namespace boost::python;
using namespace ObjCryst;

namespace {

void _SetHistogram(PowderPatternBackgroundHist &b, bp::object hist)
{
    CrystVector_REAL cv;
    assignCrystVector(cv, hist);
    b.SetHistogram(cv);
}

}  // namespace


void wrap_powderpatternbackgroundhist()
{
    class_<PowderPatternBackgroundHist, bases<PowderPatternComponent>, boost::noncopyable>(
            "PowderPatternBackgroundHist", no_init)
        .def("GetPowderPatternCalc",
                &PowderPatternBackgroundHist::GetPowderPatternCalc,
                return_value_policy<copy_const_reference>())
        .def("SetHistogram",
                &_SetHistogram,
                bp::arg("histogram"))
        .def("GetHistogram",
                &PowderPatternBackgroundHist::GetHistogram,
                return_value_policy<copy_const_reference>())
        ;
}
