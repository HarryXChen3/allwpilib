// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include <jni.h>

#include <wpi/jni_util.h>

#include "edu_wpi_first_math_WPIMathJNI.h"
#include "frc/controller/ArmFeedforward.h"

using namespace wpi::java;

extern "C" {

/*
 * Class:     edu_wpi_first_math_WPIMathJNI
 * Method:    calculate
 * Signature: (DDDDDDDD)D
 */
JNIEXPORT jdouble JNICALL
Java_edu_wpi_first_math_WPIMathJNI_calculate
  (JNIEnv* env, jclass, jdouble ks, jdouble kv, jdouble ka, jdouble kg,
   jdouble currentAngle, jdouble currentVelocity, jdouble nextVelocity,
   jdouble dt)
{
  return frc::ArmFeedforward{units::volt_t{ks}, units::volt_t{kg},
                             units::unit_t<frc::ArmFeedforward::kv_unit>{kv},
                             units::unit_t<frc::ArmFeedforward::ka_unit>{ka}}
      .Calculate(units::radian_t{currentAngle},
                 units::radians_per_second_t{currentVelocity},
                 units::radians_per_second_t{nextVelocity}, units::second_t{dt})
      .value();
}

}  // extern "C"
