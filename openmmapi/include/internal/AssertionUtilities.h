#include "openmm/OpenMMException.h"
#include "openmm/Vec3.h"
#include <math.h>
#include <vector>
#include <iostream>

#define assertEqualTo(expected, found, tol) {\
    double _scale_ = std::abs(expected) > 1.0 ? std::abs(expected) : 1.0; \
    if (!(std::abs((expected)-(found))/_scale_ <= (tol))) {\
        std::stringstream details; \
        details<<__FILE__<<":"<< __LINE__<<": Expected "<<(expected)<<", found "<<(found); \
        throw OpenMMException(details.str()); \
    } \
};

#define assertEqualVec(expected, found, tol) { \
    double _norm_ = std::sqrt((expected).dot(expected)); \
    double _scale_ = _norm_ > 1.0 ? _norm_ : 1.0; \
    if ((std::abs(((expected)[0])-((found)[0]))/_scale_ > (tol)) || \
        (std::abs(((expected)[1])-((found)[1]))/_scale_ > (tol)) || \
        (std::abs(((expected)[2])-((found)[2]))/_scale_ > (tol))) { \
        std::stringstream details; \
        details<<__FILE__<<":"<< __LINE__<<": Expected "<<(expected)<<", found "<<(found); \
        throw OpenMMException(details.str()); \
    } \
};

#define assertEnergy(state0, state1, tol) { \
    assertEqualTo(state0.getPotentialEnergy(), state1.getPotentialEnergy(), tol); \
}

#define assertForces(state0, state1, tol) { \
    const vector<Vec3>& forces0 = state0.getForces(); \
    const vector<Vec3>& forces1 = state1.getForces(); \
    for (int i = 0; i < forces0.size(); i++) \
        assertEqualVec(forces0[i], forces1[i], tol); \
}

#define assertForcesAndEnergy(context, tol) { \
    State state0 = context.getState(State::Forces | State::Energy, false, 1<<0); \
    State state1 = context.getState(State::Forces | State::Energy, false, 1<<1); \
    assertEnergy(state0, state1, tol); \
    assertForces(state0, state1, tol); \
}

#define assertEqualTo(expected, found, tol) {\
    double _scale_ = std::abs(expected) > 1.0 ? std::abs(expected) : 1.0; \
    if (!(std::abs((expected)-(found))/_scale_ <= (tol))) {\
        std::stringstream details; \
        details<<__FILE__<<":"<< __LINE__<<": Expected "<<(expected)<<", found "<<(found); \
        throw OpenMMException(details.str()); \
    } \
};
