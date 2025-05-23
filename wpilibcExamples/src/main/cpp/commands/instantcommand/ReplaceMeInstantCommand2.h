// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#pragma once

#include <frc2/command/CommandHelper.h>
#include <frc2/command/InstantCommand.h>

// NOTE:  Consider using this command inline, rather than writing a subclass.
// For more information, see:
// https://docs.wpilib.org/en/stable/docs/software/commandbased/convenience-features.html
class ReplaceMeInstantCommand2
    : public frc2::CommandHelper<frc2::InstantCommand,
                                 ReplaceMeInstantCommand2> {
 public:
  ReplaceMeInstantCommand2();

  void Initialize() override;
};
