// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#pragma once

#include <bitset>
#include <string>
#include <tuple>
#include <vector>

#include "sysid/analysis/AnalysisType.h"
#include "sysid/analysis/OLS.h"
#include "sysid/analysis/Storage.h"

namespace sysid {

/**
 * Exception for data that doesn't sample enough of the state-input space.
 */
class InsufficientSamplesError : public std::exception {
 public:
  /**
   * Constructs an InsufficientSamplesError.
   *
   * @param message The error message
   */
  explicit InsufficientSamplesError(std::string_view message) {
    m_message = message;
  }

  const char* what() const noexcept override { return m_message.c_str(); }

 private:
  /**
   * Stores the error message
   */
  std::string m_message;
};

struct OLSDataQuality {
  static constexpr std::array gainNames{"Ks", "Kv", "Ka", "Kg", "offset"};

  /**
   * Bitset describing which feedforward gains are impacted by the bad fit
   * coeffs; Bits are Ks, Kv, Ka, Kg, offset
   */
  std::bitset<5> badGains = {};
  /**
   * Minimum absolute eigenvector component along the regression variable's
   * direction.
   */
  double minAbsEigVecComponent = -1;
  /**
   * Describes if the OLS data contains a worst fit coefficient.
   */
  bool hasWorstCoeff = false;
  /**
   * Index describing the worst fit for α, β, γ and optionally δ and ε (elevator
   * and arm)
   */
  int worstFitCoeffIndex = -1;
};

struct FeedforwardGain {
  /**
   * The feedforward gain.
   */
  double gain = 1;

  /**
   * Descriptor attached to the feedforward gain.
   */
  std::string descriptor = "Feedforward gain.";

  /**
   * Whether the feedforward gain is valid.
   */
  bool isValidGain = true;

  /**
   * Error message attached to the feedforward gain.
   */
  std::string errorMessage = "No error.";
};

/**
 * Stores feedforward gains.
 */
struct FeedforwardGains {
  /**
   * Stores the raw OLSResult from analysis.
   */
  OLSResult olsResult;

  /**
   * The static gain Ks.
   */
  FeedforwardGain Ks = {};

  /**
   * The velocity gain kV.
   */
  FeedforwardGain Kv = {};

  /**
   * The acceleration gain kA.
   */
  FeedforwardGain Ka = {};

  /**
   * The gravity gain Kg.
   */
  FeedforwardGain Kg = {};

  /**
   * The offset (arm).
   */
  FeedforwardGain offset = {};
};

/**
 * Calculates feedforward gains given the data and the type of analysis to
 * perform.
 *
 * @param data The OLS input data.
 * @param type The analysis type.
 */
FeedforwardGains CalculateFeedforwardGains(const Storage& data,
                                           const AnalysisType& type);

}  // namespace sysid
