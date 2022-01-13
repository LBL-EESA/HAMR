#ifndef hamr_gil_state_h
#define hamr_gil_state_h

#include <Python.h>

namespace hamr
{

/// A RAII helper for managing the Python GIL.
/** The GIL is aquired and held while the object exists. The GIL must be held
 * by C++ code invoking any Python C-API calls.
 */
class HAMR_EXPORT gil_state
{
public:
    gil_state()
    { m_state = PyGILState_Ensure(); }

    ~gil_state()
    { PyGILState_Release(m_state); }

    gil_state(const gil_state&) = delete;
    void operator=(const gil_state&) = delete;

private:
    PyGILState_STATE m_state;
};

}

#endif
