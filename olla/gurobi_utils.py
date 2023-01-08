
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gurobipy
import os
import logging


def get_gurobi_env():
    try:
        # Try using an ISV license based on the supplied environment variables
        env = gurobipy.Env.OtherEnv(
            os.getenv("OLLA_GUROBI_LOGFILE", ""),
            os.getenv("OLLA_GUROBI_ISV_NAME"),
            os.getenv("OLLA_GUROBI_ISV_APP_NAME"),
            int(os.getenv("OLLA_GUROBI_ISV_EXPIRATION")),
            os.getenv("OLLA_GUROBI_ISV_CODE"),
        )
        logging.info("Successfully created Gurobi env with ISV license")
    except:
        logging.warning(
            "Failed to create env based on ISV license. If you intended to use",
            " an ISV license, set the OLLA_GUROBI_ISV_NAME",
            " OLLA_GUROBI_ISV_APP_NAME, OLLA_GUROBI_ISV_EXPIRATION,",
            " and OLLA_GUROBI_ISV_CODE to their corresponding values."
            " Falling back to default Gurobi environment initialization.",
        )
        # Fall back to default env
        env = gurobipy.Env()

    return env
