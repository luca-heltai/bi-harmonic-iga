set(_target ${TARGET}_test)

file(GLOB _source_tests *.cc)

FOREACH(_build_type ${_d2_build_types})
  # Postfix to use everywhere
  SET(_p "${${_build_type}_postfix}")
  # Only build this type, if deal.II was compiled with it.
  IF(CMAKE_BUILD_TYPE MATCHES "${_build_type}" AND
      DEAL_II_BUILD_TYPE MATCHES "${_build_type}")

    MESSAGE("-- Found ${_build_type} version of deal.II.")

    STRING(TOUPPER "${_build_type}" _BUILD_TYPE)

	SET(_exe "${_target}${${_build_type}_postfix}")
	MESSAGE("-- Configuring executable ${_exe}")
	ADD_EXECUTABLE(${_exe} ${_source_tests})
	
	TARGET_LINK_LIBRARIES(${_exe} ${TEST_LIBRARIES_${_BUILD_TYPE}} ${GTEST_LIBRARY})
	DEAL_II_SETUP_TARGET(${_exe} ${_BUILD_TYPE})
  ENDIF()
ENDFOREACH()