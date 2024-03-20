//
// span for C++98 and later.
// Based on http://wg21.link/p0122r7
// For more information see https://github.com/martinmoene/span-lite
//
// Copyright 2018-2020 Martin Moene
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef NONSTD_SPAN_HPP_INCLUDED
#define NONSTD_SPAN_HPP_INCLUDED

#define span_lite_MAJOR  0
#define span_lite_MINOR  9
#define span_lite_PATCH  0

#define span_lite_VERSION  span_STRINGIFY(span_lite_MAJOR) "." span_STRINGIFY(span_lite_MINOR) "." span_STRINGIFY(span_lite_PATCH)

#define span_STRINGIFY(  x )  span_STRINGIFY_( x )
#define span_STRINGIFY_( x )  #x

// span configuration:

#define span_SPAN_DEFAULT  0
#define span_SPAN_NONSTD   1
#define span_SPAN_STD      2

// tweak header support:

#ifdef __has_include
# if __has_include(<nonstd/span.tweak.hpp>)
#  include <nonstd/span.tweak.hpp>
# endif
#define span_HAVE_TWEAK_HEADER  1
#else
#define span_HAVE_TWEAK_HEADER  0
//# pragma message("span.hpp: Note: Tweak header not supported.")
#endif

// span selection and configuration:

#define span_HAVE( feature )  ( span_HAVE_##feature )

#ifndef  span_CONFIG_SELECT_SPAN
# define span_CONFIG_SELECT_SPAN  ( span_HAVE_STD_SPAN ? span_SPAN_STD : span_SPAN_NONSTD )
#endif

#ifndef  span_CONFIG_EXTENT_TYPE
# define span_CONFIG_EXTENT_TYPE  std::size_t
#endif

#ifndef  span_CONFIG_SIZE_TYPE
# define span_CONFIG_SIZE_TYPE  std::size_t
#endif

#ifdef span_CONFIG_INDEX_TYPE
# error `span_CONFIG_INDEX_TYPE` is deprecated since v0.7.0; it is replaced by `span_CONFIG_SIZE_TYPE`.
#endif

// span configuration (features):

#ifndef  span_FEATURE_WITH_CONTAINER
#ifdef   span_FEATURE_WITH_CONTAINER_TO_STD
# define span_FEATURE_WITH_CONTAINER  span_IN_STD( span_FEATURE_WITH_CONTAINER_TO_STD )
#else
# define span_FEATURE_WITH_CONTAINER  0
#endif
#endif

#ifndef  span_FEATURE_CONSTRUCTION_FROM_STDARRAY_ELEMENT_TYPE
# define span_FEATURE_CONSTRUCTION_FROM_STDARRAY_ELEMENT_TYPE  0
#endif

#ifndef  span_FEATURE_MEMBER_AT
# define span_FEATURE_MEMBER_AT  0
#endif

#ifndef  span_FEATURE_MEMBER_BACK_FRONT
# define span_FEATURE_MEMBER_BACK_FRONT  1
#endif

#ifndef  span_FEATURE_MEMBER_CALL_OPERATOR
# define span_FEATURE_MEMBER_CALL_OPERATOR  0
#endif

#ifndef  span_FEATURE_MEMBER_SWAP
# define span_FEATURE_MEMBER_SWAP  0
#endif

#ifndef  span_FEATURE_NON_MEMBER_FIRST_LAST_SUB
# define span_FEATURE_NON_MEMBER_FIRST_LAST_SUB  0
#endif

#ifndef  span_FEATURE_COMPARISON
# define span_FEATURE_COMPARISON  0  // Note: C++20 does not provide comparison
#endif

#ifndef  span_FEATURE_SAME
# define span_FEATURE_SAME  0
#endif

#if span_FEATURE_SAME && !span_FEATURE_COMPARISON
# error `span_FEATURE_SAME` requires `span_FEATURE_COMPARISON`
#endif

#ifndef  span_FEATURE_MAKE_SPAN
#ifdef   span_FEATURE_MAKE_SPAN_TO_STD
# define span_FEATURE_MAKE_SPAN  span_IN_STD( span_FEATURE_MAKE_SPAN_TO_STD )
#else
# define span_FEATURE_MAKE_SPAN  0
#endif
#endif

#ifndef  span_FEATURE_BYTE_SPAN
# define span_FEATURE_BYTE_SPAN  0
#endif

// Control presence of exception handling (try and auto discover):

#ifndef span_CONFIG_NO_EXCEPTIONS
# if _MSC_VER
#  include <cstddef>    // for _HAS_EXCEPTIONS
# endif
# if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || (_HAS_EXCEPTIONS)
#  define span_CONFIG_NO_EXCEPTIONS  0
# else
#  define span_CONFIG_NO_EXCEPTIONS  1
#  undef  span_CONFIG_CONTRACT_VIOLATION_THROWS
#  undef  span_CONFIG_CONTRACT_VIOLATION_TERMINATES
#  define span_CONFIG_CONTRACT_VIOLATION_THROWS  0
#  define span_CONFIG_CONTRACT_VIOLATION_TERMINATES  1
# endif
#endif

// Control pre- and postcondition violation behaviour:

#if    defined( span_CONFIG_CONTRACT_LEVEL_ON )
# define        span_CONFIG_CONTRACT_LEVEL_MASK  0x11
#elif  defined( span_CONFIG_CONTRACT_LEVEL_OFF )
# define        span_CONFIG_CONTRACT_LEVEL_MASK  0x00
#elif  defined( span_CONFIG_CONTRACT_LEVEL_EXPECTS_ONLY )
# define        span_CONFIG_CONTRACT_LEVEL_MASK  0x01
#elif  defined( span_CONFIG_CONTRACT_LEVEL_ENSURES_ONLY )
# define        span_CONFIG_CONTRACT_LEVEL_MASK  0x10
#else
# define        span_CONFIG_CONTRACT_LEVEL_MASK  0x11
#endif

#if    defined( span_CONFIG_CONTRACT_VIOLATION_THROWS )
# define        span_CONFIG_CONTRACT_VIOLATION_THROWS_V  span_CONFIG_CONTRACT_VIOLATION_THROWS
#else
# define        span_CONFIG_CONTRACT_VIOLATION_THROWS_V  0
#endif

#if    defined( span_CONFIG_CONTRACT_VIOLATION_THROWS     ) && span_CONFIG_CONTRACT_VIOLATION_THROWS && \
       defined( span_CONFIG_CONTRACT_VIOLATION_TERMINATES ) && span_CONFIG_CONTRACT_VIOLATION_TERMINATES
# error Please define none or one of span_CONFIG_CONTRACT_VIOLATION_THROWS and span_CONFIG_CONTRACT_VIOLATION_TERMINATES to 1, but not both.
#endif

// C++ language version detection (C++20 is speculative):
// Note: VC14.0/1900 (VS2015) lacks too much from C++14.

#ifndef   span_CPLUSPLUS
# if defined(_MSVC_LANG ) && !defined(__clang__)
#  define span_CPLUSPLUS  (_MSC_VER == 1900 ? 201103L : _MSVC_LANG )
# else
#  define span_CPLUSPLUS  __cplusplus
# endif
#endif

#define span_CPP98_OR_GREATER  ( span_CPLUSPLUS >= 199711L )
#define span_CPP11_OR_GREATER  ( span_CPLUSPLUS >= 201103L )
#define span_CPP14_OR_GREATER  ( span_CPLUSPLUS >= 201402L )
#define span_CPP17_OR_GREATER  ( span_CPLUSPLUS >= 201703L )
#define span_CPP20_OR_GREATER  ( span_CPLUSPLUS >= 202000L )

// C++ language version (represent 98 as 3):

#define span_CPLUSPLUS_V  ( span_CPLUSPLUS / 100 - (span_CPLUSPLUS > 200000 ? 2000 : 1994) )

#define span_IN_STD( v )  ( ((v) == 98 ? 3 : (v)) >= span_CPLUSPLUS_V )

#define span_CONFIG(         feature )  ( span_CONFIG_##feature )
#define span_FEATURE(        feature )  ( span_FEATURE_##feature )
#define span_FEATURE_TO_STD( feature )  ( span_IN_STD( span_FEATURE( feature##_TO_STD ) ) )

// Use C++20 std::span if available and requested:

#if span_CPP20_OR_GREATER && defined(__has_include )
# if __has_include( <span> )
#  define span_HAVE_STD_SPAN  1
# else
#  define span_HAVE_STD_SPAN  0
# endif
#else
# define  span_HAVE_STD_SPAN  0
#endif

#define  span_USES_STD_SPAN  ( (span_CONFIG_SELECT_SPAN == span_SPAN_STD) || ((span_CONFIG_SELECT_SPAN == span_SPAN_DEFAULT) && span_HAVE_STD_SPAN) )

//
// Use C++20 std::span:
//

#if span_USES_STD_SPAN

#include <span>

namespace nonstd {

using std::span;

// Note: C++20 does not provide comparison
// using std::operator==;
// using std::operator!=;
// using std::operator<;
// using std::operator<=;
// using std::operator>;
// using std::operator>=;
}  // namespace nonstd

#else  // span_USES_STD_SPAN

#include <algorithm>

// Compiler versions:
//
// MSVC++  6.0  _MSC_VER == 1200  span_COMPILER_MSVC_VERSION ==  60  (Visual Studio 6.0)
// MSVC++  7.0  _MSC_VER == 1300  span_COMPILER_MSVC_VERSION ==  70  (Visual Studio .NET 2002)
// MSVC++  7.1  _MSC_VER == 1310  span_COMPILER_MSVC_VERSION ==  71  (Visual Studio .NET 2003)
// MSVC++  8.0  _MSC_VER == 1400  span_COMPILER_MSVC_VERSION ==  80  (Visual Studio 2005)
// MSVC++  9.0  _MSC_VER == 1500  span_COMPILER_MSVC_VERSION ==  90  (Visual Studio 2008)
// MSVC++ 10.0  _MSC_VER == 1600  span_COMPILER_MSVC_VERSION == 100  (Visual Studio 2010)
// MSVC++ 11.0  _MSC_VER == 1700  span_COMPILER_MSVC_VERSION == 110  (Visual Studio 2012)
// MSVC++ 12.0  _MSC_VER == 1800  span_COMPILER_MSVC_VERSION == 120  (Visual Studio 2013)
// MSVC++ 14.0  _MSC_VER == 1900  span_COMPILER_MSVC_VERSION == 140  (Visual Studio 2015)
// MSVC++ 14.1  _MSC_VER >= 1910  span_COMPILER_MSVC_VERSION == 141  (Visual Studio 2017)
// MSVC++ 14.2  _MSC_VER >= 1920  span_COMPILER_MSVC_VERSION == 142  (Visual Studio 2019)

#if defined(_MSC_VER ) && !defined(__clang__)
# define span_COMPILER_MSVC_VER      (_MSC_VER )
# define span_COMPILER_MSVC_VERSION  (_MSC_VER / 10 - 10 * ( 5 + (_MSC_VER < 1900 ) ) )
#else
# define span_COMPILER_MSVC_VER      0
# define span_COMPILER_MSVC_VERSION  0
#endif

#define span_COMPILER_VERSION( major, minor, patch )  ( 10 * ( 10 * (major) + (minor) ) + (patch) )

#if defined(__clang__)
# define span_COMPILER_CLANG_VERSION  span_COMPILER_VERSION(__clang_major__, __clang_minor__, __clang_patchlevel__)
#else
# define span_COMPILER_CLANG_VERSION  0
#endif

#if defined(__GNUC__) && !defined(__clang__)
# define span_COMPILER_GNUC_VERSION  span_COMPILER_VERSION(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#else
# define span_COMPILER_GNUC_VERSION  0
#endif

// half-open range [lo..hi):
#define span_BETWEEN( v, lo, hi )  ( (lo) <= (v) && (v) < (hi) )

// Compiler warning suppression:

#if defined(__clang__)
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wundef"
# pragma clang diagnostic ignored "-Wmismatched-tags"
# define span_RESTORE_WARNINGS()   _Pragma( "clang diagnostic pop" )

#elif defined __GNUC__
# pragma GCC   diagnostic push
# pragma GCC   diagnostic ignored "-Wundef"
# define span_RESTORE_WARNINGS()   _Pragma( "GCC diagnostic pop" )

#elif span_COMPILER_MSVC_VER >= 1900
# define span_DISABLE_MSVC_WARNINGS(codes)  __pragma(warning(push))  __pragma(warning(disable: codes))
# define span_RESTORE_WARNINGS()            __pragma(warning(pop ))

// Suppress the following MSVC GSL warnings:
// - C26439, gsl::f.6 : special function 'function' can be declared 'noexcept'
// - C26440, gsl::f.6 : function 'function' can be declared 'noexcept'
// - C26472, gsl::t.1 : don't use a static_cast for arithmetic conversions;
//                      use brace initialization, gsl::narrow_cast or gsl::narrow
// - C26473: gsl::t.1 : don't cast between pointer types where the source type and the target type are the same
// - C26481: gsl::b.1 : don't use pointer arithmetic. Use span instead
// - C26490: gsl::t.1 : don't use reinterpret_cast

span_DISABLE_MSVC_WARNINGS( 26439 26440 26472 26473 26481 26490 )

#else
# define span_RESTORE_WARNINGS()  /*empty*/
#endif

// Presence of language and library features:

#ifdef _HAS_CPP0X
# define span_HAS_CPP0X  _HAS_CPP0X
#else
# define span_HAS_CPP0X  0
#endif

#define span_CPP11_80   (span_CPP11_OR_GREATER || span_COMPILER_MSVC_VER >= 1400)
#define span_CPP11_90   (span_CPP11_OR_GREATER || span_COMPILER_MSVC_VER >= 1500)
#define span_CPP11_100  (span_CPP11_OR_GREATER || span_COMPILER_MSVC_VER >= 1600)
#define span_CPP11_110  (span_CPP11_OR_GREATER || span_COMPILER_MSVC_VER >= 1700)
#define span_CPP11_120  (span_CPP11_OR_GREATER || span_COMPILER_MSVC_VER >= 1800)
#define span_CPP11_140  (span_CPP11_OR_GREATER || span_COMPILER_MSVC_VER >= 1900)

#define span_CPP14_000  (span_CPP14_OR_GREATER)
#define span_CPP14_120  (span_CPP14_OR_GREATER || span_COMPILER_MSVC_VER >= 1800)
#define span_CPP14_140  (span_CPP14_OR_GREATER || span_COMPILER_MSVC_VER >= 1900)

#define span_CPP17_000  (span_CPP17_OR_GREATER)

// Presence of C++11 language features:

#define span_HAVE_ALIAS_TEMPLATE            span_CPP11_140
#define span_HAVE_AUTO                      span_CPP11_100
#define span_HAVE_CONSTEXPR_11              span_CPP11_140
#define span_HAVE_DEFAULT_FUNCTION_TEMPLATE_ARG  span_CPP11_120
#define span_HAVE_EXPLICIT_CONVERSION       span_CPP11_140
#define span_HAVE_INITIALIZER_LIST          span_CPP11_120
#define span_HAVE_IS_DEFAULT                span_CPP11_140
#define span_HAVE_IS_DELETE                 span_CPP11_140
#define span_HAVE_NOEXCEPT                  span_CPP11_140
#define span_HAVE_NULLPTR                   span_CPP11_100
#define span_HAVE_STATIC_ASSERT             span_CPP11_100

// Presence of C++14 language features:

#define span_HAVE_CONSTEXPR_14              span_CPP14_000

// Presence of C++17 language features:

#define span_HAVE_DEPRECATED                span_CPP17_000
#define span_HAVE_NODISCARD                 span_CPP17_000
#define span_HAVE_NORETURN                  span_CPP17_000

// MSVC: template parameter deduction guides since Visual Studio 2017 v15.7

#if defined(__cpp_deduction_guides)
# define span_HAVE_DEDUCTION_GUIDES         1
#else
# define span_HAVE_DEDUCTION_GUIDES         (span_CPP17_OR_GREATER && ! span_BETWEEN( span_COMPILER_MSVC_VER, 1, 1913 ))
#endif

// Presence of C++ library features:

#define span_HAVE_ADDRESSOF                 span_CPP17_000
#define span_HAVE_ARRAY                     span_CPP11_110
#define span_HAVE_BYTE                      span_CPP17_000
#define span_HAVE_CONDITIONAL               span_CPP11_120
#define span_HAVE_CONTAINER_DATA_METHOD    (span_CPP11_140 || ( span_COMPILER_MSVC_VER >= 1500 && span_HAS_CPP0X ))
#define span_HAVE_DATA                      span_CPP17_000
#define span_HAVE_LONGLONG                  span_CPP11_80
#define span_HAVE_REMOVE_CONST              span_CPP11_110
#define span_HAVE_SNPRINTF                  span_CPP11_140
#define span_HAVE_STRUCT_BINDING            span_CPP11_120
#define span_HAVE_TYPE_TRAITS               span_CPP11_90

// Presence of byte-lite:

#ifdef NONSTD_BYTE_LITE_HPP
# define span_HAVE_NONSTD_BYTE  1
#else
# define span_HAVE_NONSTD_BYTE  0
#endif

// C++ feature usage:

#if span_HAVE_ADDRESSOF
# define span_ADDRESSOF(x)  std::addressof(x)
#else
# define span_ADDRESSOF(x)  (&x)
#endif

#if span_HAVE_CONSTEXPR_11
# define span_constexpr constexpr
#else
# define span_constexpr /*span_constexpr*/
#endif

#if span_HAVE_CONSTEXPR_14
# define span_constexpr14 constexpr
#else
# define span_constexpr14 /*span_constexpr*/
#endif

#if span_HAVE_EXPLICIT_CONVERSION
# define span_explicit explicit
#else
# define span_explicit /*explicit*/
#endif

#if span_HAVE_IS_DELETE
# define span_is_delete = delete
#else
# define span_is_delete
#endif

#if span_HAVE_IS_DELETE
# define span_is_delete_access public
#else
# define span_is_delete_access private
#endif

#if span_HAVE_NOEXCEPT && ! span_CONFIG_CONTRACT_VIOLATION_THROWS_V
# define span_noexcept noexcept
#else
# define span_noexcept /*noexcept*/
#endif

#if span_HAVE_NULLPTR
# define span_nullptr nullptr
#else
# define span_nullptr NULL
#endif

#if span_HAVE_DEPRECATED
# define span_deprecated(msg) [[deprecated(msg)]]
#else
# define span_deprecated(msg) /*[[deprecated]]*/
#endif

#if span_HAVE_NODISCARD
# define span_nodiscard [[nodiscard]]
#else
# define span_nodiscard /*[[nodiscard]]*/
#endif

#if span_HAVE_NORETURN
# define span_noreturn [[noreturn]]
#else
# define span_noreturn /*[[noreturn]]*/
#endif

// Other features:

#define span_HAVE_CONSTRAINED_SPAN_CONTAINER_CTOR  span_HAVE_DEFAULT_FUNCTION_TEMPLATE_ARG
#define span_HAVE_ITERATOR_CTOR                    span_HAVE_DEFAULT_FUNCTION_TEMPLATE_ARG

// Additional includes:

#if span_HAVE( ADDRESSOF )
# include <memory>
#endif

#if span_HAVE( ARRAY )
# include <array>
#endif

#if span_HAVE( BYTE )
# include <cstddef>
#endif

#if span_HAVE( DATA )
# include <iterator> // for std::data(), std::size()
#endif

#if span_HAVE( TYPE_TRAITS )
# include <type_traits>
#endif

#if ! span_HAVE( CONSTRAINED_SPAN_CONTAINER_CTOR )
# include <vector>
#endif

#if span_FEATURE( MEMBER_AT ) > 1
# include <cstdio>
#endif

#if ! span_CONFIG( NO_EXCEPTIONS )
# include <stdexcept>
#endif

// Contract violation

#define span_ELIDE_CONTRACT_EXPECTS  ( 0 == ( span_CONFIG_CONTRACT_LEVEL_MASK & 0x01 ) )
#define span_ELIDE_CONTRACT_ENSURES  ( 0 == ( span_CONFIG_CONTRACT_LEVEL_MASK & 0x10 ) )

#if span_ELIDE_CONTRACT_EXPECTS
# define span_constexpr_exp    span_constexpr
# define span_EXPECTS( cond )  /* Expect elided */
#else
# define span_constexpr_exp    span_constexpr14
# define span_EXPECTS( cond )  span_CONTRACT_CHECK( "Precondition", cond )
#endif

#if span_ELIDE_CONTRACT_ENSURES
# define span_constexpr_ens    span_constexpr
# define span_ENSURES( cond )  /* Ensures elided */
#else
# define span_constexpr_ens    span_constexpr14
# define span_ENSURES( cond )  span_CONTRACT_CHECK( "Postcondition", cond )
#endif

#define span_CONTRACT_CHECK( type, cond ) \
    cond ? static_cast< void >( 0 ) \
         : nonstd::span_lite::detail::report_contract_violation( span_LOCATION( __FILE__, __LINE__ ) ": " type " violation." )

#ifdef __GNUG__
# define span_LOCATION( file, line )  file ":" span_STRINGIFY( line )
#else
# define span_LOCATION( file, line )  file "(" span_STRINGIFY( line ) ")"
#endif

// Method enabling

#if span_HAVE( DEFAULT_FUNCTION_TEMPLATE_ARG )

#define span_REQUIRES_0(VA) \
    template< bool B = (VA), typename std::enable_if<B, int>::type = 0 >

# if span_BETWEEN( span_COMPILER_MSVC_VERSION, 1, 140 )
// VS 2013 and earlier seem to have trouble with SFINAE for default non-type arguments
# define span_REQUIRES_T(VA) \
    , typename = typename std::enable_if< ( VA ), nonstd::span_lite::detail::enabler >::type
# else
# define span_REQUIRES_T(VA) \
    , typename std::enable_if< (VA), int >::type = 0
# endif

#define span_REQUIRES_R(R, VA) \
    typename std::enable_if< (VA), R>::type

#define span_REQUIRES_A(VA) \
    , typename std::enable_if< (VA), void*>::type = nullptr

#else

# define span_REQUIRES_0(VA)    /*empty*/
# define span_REQUIRES_T(VA)    /*empty*/
# define span_REQUIRES_R(R, VA) R
# define span_REQUIRES_A(VA)    /*empty*/

#endif

namespace nonstd {
namespace span_lite {

// [views.constants], constants

typedef span_CONFIG_EXTENT_TYPE extent_t;
typedef span_CONFIG_SIZE_TYPE   size_t;

span_constexpr const extent_t dynamic_extent = static_cast<extent_t>( -1 );

template< class T, extent_t Extent = dynamic_extent >
class span;

// Tag to select span constructor taking a container (prevent ms-gsl warning C26426):

struct with_container_t { span_constexpr with_container_t() span_noexcept {} };
const  span_constexpr   with_container_t with_container;

// C++11 emulation:

namespace std11 {

#if span_HAVE( REMOVE_CONST )

using std::remove_cv;
using std::remove_const;
using std::remove_volatile;

#else

template< class T > struct remove_const            { typedef T type; };
template< class T > struct remove_const< T const > { typedef T type; };

template< class T > struct remove_volatile               { typedef T type; };
template< class T > struct remove_volatile< T volatile > { typedef T type; };

template< class T >
struct remove_cv
{
    typedef typename std11::remove_volatile< typename std11::remove_const< T >::type >::type type;
};

#endif  // span_HAVE( REMOVE_CONST )

#if span_HAVE( TYPE_TRAITS )

using std::is_same;
using std::is_signed;
using std::integral_constant;
using std::true_type;
using std::false_type;
using std::remove_reference;

#else

template< class T, T v > struct integral_constant { enum { value = v }; };
typedef integral_constant< bool, true  > true_type;
typedef integral_constant< bool, false > false_type;

template< class T, class U > struct is_same : false_type{};
template< class T          > struct is_same<T, T> : true_type{};

template< typename T >  struct is_signed : false_type {};
template<> struct is_signed<signed char> : true_type {};
template<> struct is_signed<signed int > : true_type {};
template<> struct is_signed<signed long> : true_type {};

#endif

} // namespace std11

// C++17 emulation:

namespace std17 {

template< bool v > struct bool_constant : std11::integral_constant<bool, v>{};

#if span_CPP11_120

template< class...>
using void_t = void;

#endif

#if span_HAVE( DATA )

using std::data;
using std::size;

#elif span_HAVE( CONSTRAINED_SPAN_CONTAINER_CTOR )

template< typename T, std::size_t N >
inline span_constexpr auto size( const T(&)[N] ) span_noexcept -> size_t
{
    return N;
}

template< typename C >
inline span_constexpr auto size( C const & cont ) -> decltype( cont.size() )
{
    return cont.size();
}

template< typename T, std::size_t N >
inline span_constexpr auto data( T(&arr)[N] ) span_noexcept -> T*
{
    return &arr[0];
}

template< typename C >
inline span_constexpr auto data( C & cont ) -> decltype( cont.data() )
{
    return cont.data();
}

template< typename C >
inline span_constexpr auto data( C const & cont ) -> decltype( cont.data() )
{
    return cont.data();
}

template< typename E >
inline span_constexpr auto data( std::initializer_list<E> il ) span_noexcept -> E const *
{
    return il.begin();
}

#endif // span_HAVE( DATA )

#if span_HAVE( BYTE )
using std::byte;
#elif span_HAVE( NONSTD_BYTE )
using nonstd::byte;
#endif

} // namespace std17

// C++20 emulation:

namespace std20 {

#if span_HAVE( DEDUCTION_GUIDES )
template< class T >
using iter_reference_t = decltype( *std::declval<T&>() );
#endif

} // namespace std20

// Implementation details:

namespace detail {

/*enum*/ struct enabler{};

template< typename T >
bool is_positive( T x )
{
    return std11::is_signed<T>::value ? x >= 0 : true;
}

#if span_HAVE( TYPE_TRAITS )

template< class Q >
struct is_span_oracle : std::false_type{};

template< class T, span_CONFIG_EXTENT_TYPE Extent >
struct is_span_oracle< span<T, Extent> > : std::true_type{};

template< class Q >
struct is_span : is_span_oracle< typename std::remove_cv<Q>::type >{};

template< class Q >
struct is_std_array_oracle : std::false_type{};

#if span_HAVE( ARRAY )

template< class T, std::size_t Extent >
struct is_std_array_oracle< std::array<T, Extent> > : std::true_type{};

#endif

template< class Q >
struct is_std_array : is_std_array_oracle< typename std::remove_cv<Q>::type >{};

template< class Q >
struct is_array : std::false_type {};

template< class T >
struct is_array<T[]> : std::true_type {};

template< class T, std::size_t N >
struct is_array<T[N]> : std::true_type {};

#if span_CPP11_140 && ! span_BETWEEN( span_COMPILER_GNUC_VERSION, 1, 500 )

template< class, class = void >
struct has_size_and_data : std::false_type{};

template< class C >
struct has_size_and_data
<
    C, std17::void_t<
        decltype( std17::size(std::declval<C>()) ),
        decltype( std17::data(std::declval<C>()) ) >
> : std::true_type{};

template< class, class, class = void >
struct is_compatible_element : std::false_type {};

template< class C, class E >
struct is_compatible_element
<
    C, E, std17::void_t<
        decltype( std17::data(std::declval<C>()) ) >
> : std::is_convertible< typename std::remove_pointer<decltype( std17::data( std::declval<C&>() ) )>::type(*)[], E(*)[] >{};

template< class C >
struct is_container : std17::bool_constant
<
    ! is_span< C >::value
    && ! is_array< C >::value
    && ! is_std_array< C >::value
    &&   has_size_and_data< C >::value
>{};

template< class C, class E >
struct is_compatible_container : std17::bool_constant
<
    is_container<C>::value
    && is_compatible_element<C,E>::value
>{};

#else // span_CPP11_140

template<
    class C, class E
        span_REQUIRES_T((
            ! is_span< C >::value
            && ! is_array< C >::value
            && ! is_std_array< C >::value
            && ( std::is_convertible< typename std::remove_pointer<decltype( std17::data( std::declval<C&>() ) )>::type(*)[], E(*)[] >::value)
        //  &&   has_size_and_data< C >::value
        ))
        , class = decltype( std17::size(std::declval<C>()) )
        , class = decltype( std17::data(std::declval<C>()) )
>
struct is_compatible_container : std::true_type{};

#endif // span_CPP11_140

#endif // span_HAVE( TYPE_TRAITS )

#if ! span_CONFIG( NO_EXCEPTIONS )
#if   span_FEATURE( MEMBER_AT ) > 1

// format index and size:

#if defined(__clang__)
# pragma clang diagnostic ignored "-Wlong-long"
#elif defined __GNUC__
# pragma GCC   diagnostic ignored "-Wformat=ll"
# pragma GCC   diagnostic ignored "-Wlong-long"
#endif

inline void throw_out_of_range( size_t idx, size_t size )
{
    const char fmt[] = "span::at(): index '%lli' is out of range [0..%lli)";
    char buffer[ 2 * 20 + sizeof fmt ];
    sprintf( buffer, fmt, static_cast<long long>(idx), static_cast<long long>(size) );

    throw std::out_of_range( buffer );
}

#else // MEMBER_AT

inline void throw_out_of_range( size_t /*idx*/, size_t /*size*/ )
{
    throw std::out_of_range( "span::at(): index outside span" );
}
#endif  // MEMBER_AT
#endif  // NO_EXCEPTIONS

#if span_CONFIG( CONTRACT_VIOLATION_THROWS_V )

struct contract_violation : std::logic_error
{
    explicit contract_violation( char const * const message )
        : std::logic_error( message )
    {}
};

inline void report_contract_violation( char const * msg )
{
    throw contract_violation( msg );
}

#else // span_CONFIG( CONTRACT_VIOLATION_THROWS_V )

span_noreturn inline void report_contract_violation( char const * /*msg*/ ) span_noexcept
{
    std::terminate();
}

#endif // span_CONFIG( CONTRACT_VIOLATION_THROWS_V )

}  // namespace detail

// Prevent signed-unsigned mismatch:

#define span_sizeof(T)  static_cast<extent_t>( sizeof(T) )

template< class T >
inline span_constexpr size_t to_size( T size )
{
    return static_cast<size_t>( size );
}

//
// [views.span] - A view over a contiguous, single-dimension sequence of objects
//
template< class T, extent_t Extent /*= dynamic_extent*/ >
class span
{
public:
    // constants and types

    typedef T element_type;
    typedef typename std11::remove_cv< T >::type value_type;

    typedef T &       reference;
    typedef T *       pointer;
    typedef T const * const_pointer;
    typedef T const & const_reference;

    typedef size_t    size_type;
    typedef extent_t  extent_type;

    typedef pointer        iterator;
    typedef const_pointer  const_iterator;

    typedef std::ptrdiff_t difference_type;

    typedef std::reverse_iterator< iterator >       reverse_iterator;
    typedef std::reverse_iterator< const_iterator > const_reverse_iterator;

//    static constexpr extent_type extent = Extent;
    enum { extent = Extent };

    // 26.7.3.2 Constructors, copy, and assignment [span.cons]

    span_REQUIRES_0(
        ( Extent == 0 ) ||
        ( Extent == dynamic_extent )
    )
    span_constexpr span() span_noexcept
        : data_( span_nullptr )
        , size_( 0 )
    {
        // span_EXPECTS( data() == span_nullptr );
        // span_EXPECTS( size() == 0 );
    }

#if span_HAVE( ITERATOR_CTOR )
    template< typename It >
    span_constexpr_exp span( It first, size_type count )
        : data_( to_address( first ) )
        , size_( count )
    {
        span_EXPECTS(
            ( data_ == span_nullptr && count == 0 ) ||
            ( data_ != span_nullptr && detail::is_positive( count ) )
        );
    }
#else
    span_constexpr_exp span( pointer ptr, size_type count )
        : data_( ptr )
        , size_( count )
    {
        span_EXPECTS(
            ( ptr == span_nullptr && count == 0 ) ||
            ( ptr != span_nullptr && detail::is_positive( count ) )
        );
    }
#endif

#if span_HAVE( ITERATOR_CTOR )
    template< typename It, typename End
        span_REQUIRES_T(( ! std::is_convertible<End, std::size_t>::value ))
     >
    span_constexpr_exp span( It first, End last )
        : data_( to_address( first ) )
        , size_( to_size( last - first ) )
    {
        span_EXPECTS(
             last - first >= 0
        );
    }
#else
    span_constexpr_exp span( pointer first, pointer last )
        : data_( first )
        , size_( to_size( last - first ) )
    {
        span_EXPECTS(
            last - first >= 0
        );
    }
#endif

    template< std::size_t N
        span_REQUIRES_T((
            (Extent == dynamic_extent || Extent == static_cast<extent_t>(N))
            && std::is_convertible< value_type(*)[], element_type(*)[] >::value
        ))
    >
    span_constexpr span( element_type ( &arr )[ N ] ) span_noexcept
        : data_( span_ADDRESSOF( arr[0] ) )
        , size_( N  )
    {}

#if span_HAVE( ARRAY )

    template< std::size_t N
        span_REQUIRES_T((
            (Extent == dynamic_extent || Extent == static_cast<extent_t>(N))
            && std::is_convertible< value_type(*)[], element_type(*)[] >::value
        ))
    >
# if span_FEATURE( CONSTRUCTION_FROM_STDARRAY_ELEMENT_TYPE )
        span_constexpr span( std::array< element_type, N > & arr ) span_noexcept
# else
        span_constexpr span( std::array< value_type, N > & arr ) span_noexcept
# endif
        : data_( arr.data() )
        , size_( to_size( arr.size() ) )
    {}

    template< std::size_t N
# if span_HAVE( DEFAULT_FUNCTION_TEMPLATE_ARG )
        span_REQUIRES_T((
            (Extent == dynamic_extent || Extent == static_cast<extent_t>(N))
            && std::is_convertible< value_type(*)[], element_type(*)[] >::value
        ))
# endif
    >
    span_constexpr span( std::array< value_type, N> const & arr ) span_noexcept
        : data_( arr.data() )
        , size_( to_size( arr.size() ) )
    {}

#endif // span_HAVE( ARRAY )

#if span_HAVE( CONSTRAINED_SPAN_CONTAINER_CTOR )
    template< class Container
        span_REQUIRES_T((
            detail::is_compatible_container< Container, element_type >::value
        ))
    >
    span_constexpr span( Container & cont )
        : data_( std17::data( cont ) )
        , size_( to_size( std17::size( cont ) ) )
    {}

    template< class Container
        span_REQUIRES_T((
            std::is_const< element_type >::value
            && detail::is_compatible_container< Container, element_type >::value
        ))
    >
    span_constexpr span( Container const & cont )
        : data_( std17::data( cont ) )
        , size_( to_size( std17::size( cont ) ) )
    {}

#endif // span_HAVE( CONSTRAINED_SPAN_CONTAINER_CTOR )

#if span_FEATURE( WITH_CONTAINER )

    template< class Container >
    span_constexpr span( with_container_t, Container & cont )
        : data_( cont.size() == 0 ? span_nullptr : span_ADDRESSOF( cont[0] ) )
        , size_( to_size( cont.size() ) )
    {}

    template< class Container >
    span_constexpr span( with_container_t, Container const & cont )
        : data_( cont.size() == 0 ? span_nullptr : const_cast<pointer>( span_ADDRESSOF( cont[0] ) ) )
        , size_( to_size( cont.size() ) )
    {}
#endif

#if span_HAVE( IS_DEFAULT )
    span_constexpr span( span const & other ) span_noexcept = default;

    ~span() span_noexcept = default;

    span_constexpr14 span & operator=( span const & other ) span_noexcept = default;
#else
    span_constexpr span( span const & other ) span_noexcept
        : data_( other.data_ )
        , size_( other.size_ )
    {}

    ~span() span_noexcept
    {}

    span_constexpr14 span & operator=( span const & other ) span_noexcept
    {
        data_ = other.data_;
        size_ = other.size_;

        return *this;
    }
#endif

    template< class OtherElementType, extent_type OtherExtent
        span_REQUIRES_T((
            (Extent == dynamic_extent || Extent == OtherExtent)
            && std::is_convertible<OtherElementType(*)[], element_type(*)[]>::value
        ))
    >
    span_constexpr_exp span( span<OtherElementType, OtherExtent> const & other ) span_noexcept
        : data_( reinterpret_cast<pointer>( other.data() ) )
        , size_( other.size() )
    {
        span_EXPECTS( OtherExtent == dynamic_extent || other.size() == to_size(OtherExtent) );
    }

    // 26.7.3.3 Subviews [span.sub]

    template< extent_type Count >
    span_constexpr_exp span< element_type, Count >
    first() const
    {
        span_EXPECTS( detail::is_positive( Count ) && Count <= size() );

        return span< element_type, Count >( data(), Count );
    }

    template< extent_type Count >
    span_constexpr_exp span< element_type, Count >
    last() const
    {
        span_EXPECTS( detail::is_positive( Count ) && Count <= size() );

        return span< element_type, Count >( data() + (size() - Count), Count );
    }

#if span_HAVE( DEFAULT_FUNCTION_TEMPLATE_ARG )
    template< size_type Offset, extent_type Count = dynamic_extent >
#else
    template< size_type Offset, extent_type Count /*= dynamic_extent*/ >
#endif
    span_constexpr_exp span< element_type, Count >
    subspan() const
    {
        span_EXPECTS(
            ( detail::is_positive( Offset ) && Offset <= size() ) &&
            ( Count == dynamic_extent || (detail::is_positive( Count ) && Count + Offset <= size()) )
        );

        return span< element_type, Count >(
            data() + Offset, Count != dynamic_extent ? Count : (Extent != dynamic_extent ? Extent - Offset : size() - Offset) );
    }

    span_constexpr_exp span< element_type, dynamic_extent >
    first( size_type count ) const
    {
        span_EXPECTS( detail::is_positive( count ) && count <= size() );

        return span< element_type, dynamic_extent >( data(), count );
    }

    span_constexpr_exp span< element_type, dynamic_extent >
    last( size_type count ) const
    {
        span_EXPECTS( detail::is_positive( count ) && count <= size() );

        return span< element_type, dynamic_extent >( data() + ( size() - count ), count );
    }

    span_constexpr_exp span< element_type, dynamic_extent >
    subspan( size_type offset, size_type count = static_cast<size_type>(dynamic_extent) ) const
    {
        span_EXPECTS(
            ( ( detail::is_positive( offset ) && offset <= size() ) ) &&
            ( count == static_cast<size_type>(dynamic_extent) || ( detail::is_positive( count ) && offset + count <= size() ) )
        );

        return span< element_type, dynamic_extent >(
            data() + offset, count == static_cast<size_type>(dynamic_extent) ? size() - offset : count );
    }

    // 26.7.3.4 Observers [span.obs]

    span_constexpr size_type size() const span_noexcept
    {
        return size_;
    }

    span_constexpr std::ptrdiff_t ssize() const span_noexcept
    {
        return static_cast<std::ptrdiff_t>( size_ );
    }

    span_constexpr size_type size_bytes() const span_noexcept
    {
        return size() * to_size( sizeof( element_type ) );
    }

    span_nodiscard span_constexpr bool empty() const span_noexcept
    {
        return size() == 0;
    }

    // 26.7.3.5 Element access [span.elem]

    span_constexpr_exp reference operator[]( size_type idx ) const
    {
        span_EXPECTS( detail::is_positive( idx ) && idx < size() );

        return *( data() + idx );
    }

#if span_FEATURE( MEMBER_CALL_OPERATOR )
    span_deprecated("replace operator() with operator[]")

    span_constexpr_exp reference operator()( size_type idx ) const
    {
        span_EXPECTS( detail::is_positive( idx ) && idx < size() );

        return *( data() + idx );
    }
#endif

#if span_FEATURE( MEMBER_AT )
    span_constexpr14 reference at( size_type idx ) const
    {
#if span_CONFIG( NO_EXCEPTIONS )
        return this->operator[]( idx );
#else
        if ( !detail::is_positive( idx ) || size() <= idx )
        {
            detail::throw_out_of_range( idx, size() );
        }
        return *( data() + idx );
#endif
    }
#endif

    span_constexpr pointer data() const span_noexcept
    {
        return data_;
    }

#if span_FEATURE( MEMBER_BACK_FRONT )

    span_constexpr_exp reference front() const span_noexcept
    {
        span_EXPECTS( ! empty() );

        return *data();
    }

    span_constexpr_exp reference back() const span_noexcept
    {
        span_EXPECTS( ! empty() );

        return *( data() + size() - 1 );
    }

#endif

    // xx.x.x.x Modifiers [span.modifiers]

#if span_FEATURE( MEMBER_SWAP )

    span_constexpr14 void swap( span & other ) span_noexcept
    {
        using std::swap;
        swap( data_, other.data_ );
        swap( size_, other.size_ );
    }
#endif

    // 26.7.3.6 Iterator support [span.iterators]

    span_constexpr iterator begin() const span_noexcept
    {
#if span_CPP11_OR_GREATER
        return { data() };
#else
        return iterator( data() );
#endif
    }

    span_constexpr iterator end() const span_noexcept
    {
#if span_CPP11_OR_GREATER
        return { data() + size() };
#else
        return iterator( data() + size() );
#endif
    }

    span_constexpr const_iterator cbegin() const span_noexcept
    {
#if span_CPP11_OR_GREATER
        return { data() };
#else
        return const_iterator( data() );
#endif
    }

    span_constexpr const_iterator cend() const span_noexcept
    {
#if span_CPP11_OR_GREATER
        return { data() + size() };
#else
        return const_iterator( data() + size() );
#endif
    }

    span_constexpr reverse_iterator rbegin() const span_noexcept
    {
        return reverse_iterator( end() );
    }

    span_constexpr reverse_iterator rend() const span_noexcept
    {
        return reverse_iterator( begin() );
    }

    span_constexpr const_reverse_iterator crbegin() const span_noexcept
    {
        return const_reverse_iterator ( cend() );
    }

    span_constexpr const_reverse_iterator crend() const span_noexcept
    {
        return const_reverse_iterator( cbegin() );
    }

private:

#if span_HAVE( ITERATOR_CTOR )
    static inline span_constexpr pointer to_address( std::nullptr_t ) span_noexcept
    {
        return nullptr;
    }

    template< typename U >
    static inline span_constexpr U * to_address( U * p ) span_noexcept
    {
        return p;
    }

    template< typename Ptr
        span_REQUIRES_T(( ! std::is_pointer<Ptr>::value ))
    >
    static inline span_constexpr pointer to_address( Ptr const & it ) span_noexcept
    {
        return to_address( it.operator->() );
    }
#endif // span_HAVE( ITERATOR_CTOR )

private:
    pointer   data_;
    size_type size_;
};

// class template argument deduction guides:

#if span_HAVE( DEDUCTION_GUIDES )

template< class T, size_t N >
span( T (&)[N] ) -> span<T, static_cast<extent_t>(N)>;

template< class T, size_t N >
span( std::array<T, N> & ) -> span<T, static_cast<extent_t>(N)>;

template< class T, size_t N >
span( std::array<T, N> const & ) -> span<const T, static_cast<extent_t>(N)>;

template< class Container >
span( Container& ) -> span<typename Container::value_type>;

template< class Container >
span( Container const & ) -> span<const typename Container::value_type>;

// iterator: constraints: It satisfies contiguous_­iterator.

template< class It, class EndOrSize >
span( It, EndOrSize ) -> span< typename std11::remove_reference< typename std20::iter_reference_t<It> >::type >;

#endif // span_HAVE( DEDUCTION_GUIDES )

// 26.7.3.7 Comparison operators [span.comparison]

#if span_FEATURE( COMPARISON )
#if span_FEATURE( SAME )

template< class T1, extent_t E1, class T2, extent_t E2  >
inline span_constexpr bool same( span<T1,E1> const & l, span<T2,E2> const & r ) span_noexcept
{
    return std11::is_same<T1, T2>::value
        && l.size() == r.size()
        && static_cast<void const*>( l.data() ) == r.data();
}

#endif

template< class T1, extent_t E1, class T2, extent_t E2  >
inline span_constexpr bool operator==( span<T1,E1> const & l, span<T2,E2> const & r )
{
    return
#if span_FEATURE( SAME )
        same( l, r ) ||
#endif
        ( l.size() == r.size() && std::equal( l.begin(), l.end(), r.begin() ) );
}

template< class T1, extent_t E1, class T2, extent_t E2  >
inline span_constexpr bool operator<( span<T1,E1> const & l, span<T2,E2> const & r )
{
    return std::lexicographical_compare( l.begin(), l.end(), r.begin(), r.end() );
}

template< class T1, extent_t E1, class T2, extent_t E2  >
inline span_constexpr bool operator!=( span<T1,E1> const & l, span<T2,E2> const & r )
{
    return !( l == r );
}

template< class T1, extent_t E1, class T2, extent_t E2  >
inline span_constexpr bool operator<=( span<T1,E1> const & l, span<T2,E2> const & r )
{
    return !( r < l );
}

template< class T1, extent_t E1, class T2, extent_t E2  >
inline span_constexpr bool operator>( span<T1,E1> const & l, span<T2,E2> const & r )
{
    return ( r < l );
}

template< class T1, extent_t E1, class T2, extent_t E2  >
inline span_constexpr bool operator>=( span<T1,E1> const & l, span<T2,E2> const & r )
{
    return !( l < r );
}

#endif // span_FEATURE( COMPARISON )

// 26.7.2.6 views of object representation [span.objectrep]

#if span_HAVE( BYTE ) || span_HAVE( NONSTD_BYTE )

// Avoid MSVC 14.1 (1910), VS 2017: warning C4307: '*': integral constant overflow:

template< typename T, extent_t Extent >
struct BytesExtent
{
#if span_CPP11_OR_GREATER
    enum ET : extent_t { value = span_sizeof(T) * Extent };
#else
    enum ET { value = span_sizeof(T) * Extent };
#endif
};

template< typename T >
struct BytesExtent< T, dynamic_extent >
{
#if span_CPP11_OR_GREATER
    enum ET : extent_t { value = dynamic_extent };
#else
    enum ET { value = dynamic_extent };
#endif
};

template< class T, extent_t Extent >
inline span_constexpr span< const std17::byte, BytesExtent<T, Extent>::value >
as_bytes( span<T,Extent> spn ) span_noexcept
{
#if 0
    return { reinterpret_cast< std17::byte const * >( spn.data() ), spn.size_bytes() };
#else
    return span< const std17::byte, BytesExtent<T, Extent>::value >(
        reinterpret_cast< std17::byte const * >( spn.data() ), spn.size_bytes() );  // NOLINT
#endif
}

template< class T, extent_t Extent >
inline span_constexpr span< std17::byte, BytesExtent<T, Extent>::value >
as_writable_bytes( span<T,Extent> spn ) span_noexcept
{
#if 0
    return { reinterpret_cast< std17::byte * >( spn.data() ), spn.size_bytes() };
#else
    return span< std17::byte, BytesExtent<T, Extent>::value >(
        reinterpret_cast< std17::byte * >( spn.data() ), spn.size_bytes() );  // NOLINT
#endif
}

#endif // span_HAVE( BYTE ) || span_HAVE( NONSTD_BYTE )

// extensions: non-member views:
// this feature implies the presence of make_span()

#if span_FEATURE( NON_MEMBER_FIRST_LAST_SUB ) && span_CPP11_120

template< extent_t Count, class T >
span_constexpr auto
first( T & t ) -> decltype( make_span(t).template first<Count>() )
{
    return make_span( t ).template first<Count>();
}

template< class T >
span_constexpr auto
first( T & t, size_t count ) -> decltype( make_span(t).first(count) )
{
    return make_span( t ).first( count );
}

template< extent_t Count, class T >
span_constexpr auto
last( T & t ) -> decltype( make_span(t).template last<Count>() )
{
    return make_span(t).template last<Count>();
}

template< class T >
span_constexpr auto
last( T & t, extent_t count ) -> decltype( make_span(t).last(count) )
{
    return make_span( t ).last( count );
}

template< size_t Offset, extent_t Count = dynamic_extent, class T >
span_constexpr auto
subspan( T & t ) -> decltype( make_span(t).template subspan<Offset, Count>() )
{
    return make_span( t ).template subspan<Offset, Count>();
}

template< class T >
span_constexpr auto
subspan( T & t, size_t offset, extent_t count = dynamic_extent ) -> decltype( make_span(t).subspan(offset, count) )
{
    return make_span( t ).subspan( offset, count );
}

#endif // span_FEATURE( NON_MEMBER_FIRST_LAST_SUB )

// 27.8 Container and view access [iterator.container]

template< class T, extent_t Extent /*= dynamic_extent*/ >
span_constexpr std::size_t size( span<T,Extent> const & spn )
{
    return static_cast<std::size_t>( spn.size() );
}

template< class T, extent_t Extent /*= dynamic_extent*/ >
span_constexpr std::ptrdiff_t ssize( span<T,Extent> const & spn )
{
    return static_cast<std::ptrdiff_t>( spn.size() );
}

}  // namespace span_lite
}  // namespace nonstd

// make available in nonstd:

namespace nonstd {

using span_lite::dynamic_extent;

using span_lite::span;

using span_lite::with_container;

#if span_FEATURE( COMPARISON )
#if span_FEATURE( SAME )
using span_lite::same;
#endif

using span_lite::operator==;
using span_lite::operator!=;
using span_lite::operator<;
using span_lite::operator<=;
using span_lite::operator>;
using span_lite::operator>=;
#endif

#if span_HAVE( BYTE )
using span_lite::as_bytes;
using span_lite::as_writable_bytes;
#endif

using span_lite::size;
using span_lite::ssize;

}  // namespace nonstd

#endif  // span_USES_STD_SPAN

// make_span() [span-lite extension]:

#if span_FEATURE( MAKE_SPAN ) || span_FEATURE( NON_MEMBER_FIRST_LAST_SUB )

#if span_USES_STD_SPAN
# define  span_constexpr  constexpr
# define  span_noexcept   noexcept
# define  span_nullptr    nullptr
# ifndef  span_CONFIG_EXTENT_TYPE
#  define span_CONFIG_EXTENT_TYPE  std::size_t
# endif
using extent_t = span_CONFIG_EXTENT_TYPE;
#endif  // span_USES_STD_SPAN

namespace nonstd {
namespace span_lite {

template< class T >
inline span_constexpr span<T>
make_span( T * ptr, size_t count ) span_noexcept
{
    return span<T>( ptr, count );
}

template< class T >
inline span_constexpr span<T>
make_span( T * first, T * last ) span_noexcept
{
    return span<T>( first, last );
}

template< class T, std::size_t N >
inline span_constexpr span<T, static_cast<extent_t>(N)>
make_span( T ( &arr )[ N ] ) span_noexcept
{
    return span<T, static_cast<extent_t>(N)>( &arr[ 0 ], N );
}

#if span_USES_STD_SPAN || span_HAVE( ARRAY )

template< class T, std::size_t N >
inline span_constexpr span<T, static_cast<extent_t>(N)>
make_span( std::array< T, N > & arr ) span_noexcept
{
    return span<T, static_cast<extent_t>(N)>( arr );
}

template< class T, std::size_t N >
inline span_constexpr span< const T, static_cast<extent_t>(N) >
make_span( std::array< T, N > const & arr ) span_noexcept
{
    return span<const T, static_cast<extent_t>(N)>( arr );
}

#endif // span_HAVE( ARRAY )

#if span_USES_STD_SPAN

template< class Container, class EP = decltype( std::data(std::declval<Container&>())) >
inline span_constexpr auto
make_span( Container & cont ) span_noexcept -> span< typename std::remove_pointer<EP>::type >
{
    return span< typename std::remove_pointer<EP>::type >( cont );
}

template< class Container, class EP = decltype( std::data(std::declval<Container&>())) >
inline span_constexpr auto
make_span( Container const & cont ) span_noexcept -> span< const typename std::remove_pointer<EP>::type >
{
    return span< const typename std::remove_pointer<EP>::type >( cont );
}

#elif span_HAVE( CONSTRAINED_SPAN_CONTAINER_CTOR ) && span_HAVE( AUTO )

template< class Container, class EP = decltype( std17::data(std::declval<Container&>())) >
inline span_constexpr auto
make_span( Container & cont ) span_noexcept -> span< typename std::remove_pointer<EP>::type >
{
    return span< typename std::remove_pointer<EP>::type >( cont );
}

template< class Container, class EP = decltype( std17::data(std::declval<Container&>())) >
inline span_constexpr auto
make_span( Container const & cont ) span_noexcept -> span< const typename std::remove_pointer<EP>::type >
{
    return span< const typename std::remove_pointer<EP>::type >( cont );
}

#else

template< class T, class Allocator >
inline span_constexpr span<T>
make_span( std::vector<T, Allocator> & cont ) span_noexcept
{
    return span<T>( with_container, cont );
}

template< class T, class Allocator >
inline span_constexpr span<const T>
make_span( std::vector<T, Allocator> const & cont ) span_noexcept
{
    return span<const T>( with_container, cont );
}

#endif // span_USES_STD_SPAN || ( ... )

#if ! span_USES_STD_SPAN && span_FEATURE( WITH_CONTAINER )

template< class Container >
inline span_constexpr span<typename Container::value_type>
make_span( with_container_t, Container & cont ) span_noexcept
{
    return span< typename Container::value_type >( with_container, cont );
}

template< class Container >
inline span_constexpr span<const typename Container::value_type>
make_span( with_container_t, Container const & cont ) span_noexcept
{
    return span< const typename Container::value_type >( with_container, cont );
}

#endif // ! span_USES_STD_SPAN && span_FEATURE( WITH_CONTAINER )


}  // namespace span_lite
}  // namespace nonstd

// make available in nonstd:

namespace nonstd {
using span_lite::make_span;
}  // namespace nonstd

#endif // #if span_FEATURE_TO_STD( MAKE_SPAN )

#if span_CPP11_OR_GREATER && span_FEATURE( BYTE_SPAN ) && ( span_HAVE( BYTE ) || span_HAVE( NONSTD_BYTE ) )

namespace nonstd {
namespace span_lite {

template< class T >
inline span_constexpr auto
byte_span( T & t ) span_noexcept -> span< std17::byte, span_sizeof(T) >
{
    return span< std17::byte, span_sizeof(t) >( reinterpret_cast< std17::byte * >( &t ), span_sizeof(T) );
}

template< class T >
inline span_constexpr auto
byte_span( T const & t ) span_noexcept -> span< const std17::byte, span_sizeof(T) >
{
    return span< const std17::byte, span_sizeof(t) >( reinterpret_cast< std17::byte const * >( &t ), span_sizeof(T) );
}

}  // namespace span_lite
}  // namespace nonstd

// make available in nonstd:

namespace nonstd {
using span_lite::byte_span;
}  // namespace nonstd

#endif // span_FEATURE( BYTE_SPAN )

#if span_HAVE( STRUCT_BINDING )

#if   span_CPP14_OR_GREATER
# include <tuple>
#elif span_CPP11_OR_GREATER
# include <tuple>
namespace std {
    template< std::size_t I, typename T >
    using tuple_element_t = typename tuple_element<I, T>::type;
}
#else
namespace std {
    template< typename T >
    class tuple_size; /*undefined*/

    template< std::size_t I, typename T >
    class tuple_element; /* undefined */
}
#endif // span_CPP14_OR_GREATER

namespace std {

// 26.7.X Tuple interface

// std::tuple_size<>:

template< typename ElementType, nonstd::span_lite::extent_t Extent >
class tuple_size< nonstd::span<ElementType, Extent> > : public integral_constant<size_t, static_cast<size_t>(Extent)> {};

// std::tuple_size<>: Leave undefined for dynamic extent:

template< typename ElementType >
class tuple_size< nonstd::span<ElementType, nonstd::dynamic_extent> >;

// std::tuple_element<>:

template< size_t I, typename ElementType, nonstd::span_lite::extent_t Extent >
class tuple_element< I, nonstd::span<ElementType, Extent> >
{
public:
#if span_HAVE( STATIC_ASSERT )
    static_assert( Extent != nonstd::dynamic_extent && I < Extent, "tuple_element<I,span>: dynamic extent or index out of range" );
#endif
    using type = ElementType;
};

// std::get<>(), 2 variants:

template< size_t I, typename ElementType, nonstd::span_lite::extent_t Extent >
span_constexpr ElementType & get( nonstd::span<ElementType, Extent> & spn ) span_noexcept
{
#if span_HAVE( STATIC_ASSERT )
    static_assert( Extent != nonstd::dynamic_extent && I < Extent, "get<>(span): dynamic extent or index out of range" );
#endif
    return spn[I];
}

template< size_t I, typename ElementType, nonstd::span_lite::extent_t Extent >
span_constexpr ElementType const & get( nonstd::span<ElementType, Extent> const & spn ) span_noexcept
{
#if span_HAVE( STATIC_ASSERT )
    static_assert( Extent != nonstd::dynamic_extent && I < Extent, "get<>(span): dynamic extent or index out of range" );
#endif
    return spn[I];
}

} // end namespace std

#endif // span_HAVE( STRUCT_BINDING )

#if ! span_USES_STD_SPAN
span_RESTORE_WARNINGS()
#endif  // span_USES_STD_SPAN

#endif  // NONSTD_SPAN_HPP_INCLUDED
