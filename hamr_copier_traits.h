#ifndef hamr_copier_traits_h
#define hamr_copier_traits_h

#include "hamr_config.h"
#include <type_traits>

namespace hamr
{
/// @name type trait that enables object copy
///@{
template <typename T, typename U, bool val = (!std::is_arithmetic<T>::value || !std::is_arithmetic<U>::value)> struct use_object_copier : std::false_type {};
template <typename T, typename U> struct use_object_copier<T, U, true> : std::true_type {};
template <typename T, typename U> using use_object_copier_t = typename std::enable_if<use_object_copier<T,U>::value>::type;
///@}


/// @name type trait that enables POD copy from different types
///@{
#if defined(HAMR_ENABLE_OBJECTS)
template <typename T, typename U, bool val = (!std::is_same<T,U>::value)> struct use_cons_copier : std::false_type {};
template <typename T, typename U> struct use_cons_copier<T, U, true> : std::true_type {};
template <typename T, typename U> using use_cons_copier_t = typename std::enable_if<use_cons_copier<T,U>::value>::type;
#else
template <typename T, typename U, bool val = (!std::is_same<T,U>::value && std::is_arithmetic<T>::value)> struct use_cons_copier : std::false_type {};
template <typename T, typename U> struct use_cons_copier<T, U, true> : std::true_type {};
template <typename T, typename U> using use_cons_copier_t = typename std::enable_if<use_cons_copier<T,U>::value>::type;
#endif
///@}

/// @name type trait that enables POD copy from the same types
///@{
template <typename T, typename U, bool obj = (std::is_same<T,U>::value && std::is_arithmetic<T>::value)> struct use_bytes_copier : std::false_type {};
template <typename T, typename U> struct use_bytes_copier<T, U, true> : std::true_type {};
template <typename T, typename U> using use_bytes_copier_t = typename std::enable_if<use_bytes_copier<T,U>::value>::type;
///@}

}

#endif
