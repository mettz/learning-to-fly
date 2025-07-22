#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include <learning_to_fly/simulator/operations_cpu.h>
#include <learning_to_fly/simulator/metrics.h>

#include "config/config.h"

#include <rl_tools/rl/algorithms/td3/loop.h>


#include "training_state.h"

#include "steps/checkpoint.h"
#include "steps/critic_reset.h"
#include "steps/curriculum.h"
#include "steps/log_reward.h"
#include "steps/logger.h"
#include "steps/trajectory_collection.h"
#include "steps/validation.h"

#include "helpers.h"

#include <filesystem>
#include <fstream>


namespace learning_to_fly{

    template <typename T_CONFIG>
    void init(TrainingState<T_CONFIG>& ts, typename T_CONFIG::TI seed = 0){
        using CONFIG = T_CONFIG;
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        using ABLATION_SPEC = typename CONFIG::ABLATION_SPEC;
        auto env_parameters = parameters::environment<T, TI, ABLATION_SPEC>::parameters;
        auto env_parameters_eval = parameters::environment<T, TI, config::template ABLATION_SPEC_EVAL<ABLATION_SPEC>>::parameters;
        for (auto& env : ts.envs) {
            env.parameters = env_parameters;
        }
        ts.env_eval.parameters = env_parameters_eval;
        TI effective_seed = CONFIG::BASE_SEED + seed;
        ts.run_name = helpers::run_name<ABLATION_SPEC, CONFIG>(effective_seed);
        rlt::construct(ts.device, ts.device.logger, std::string("logs"), ts.run_name);

        rlt::set_step(ts.device, ts.device.logger, 0);
        rlt::add_scalar(ts.device, ts.device.logger, "loop/seed", effective_seed);
        rlt::rl::algorithms::td3::loop::init(ts, effective_seed);
        ts.off_policy_runner.parameters = CONFIG::off_policy_runner_parameters;

        for(typename CONFIG::ENVIRONMENT& env: ts.validation_envs){
            env.parameters = parameters::environment<typename CONFIG::T, TI, ABLATION_SPEC>::parameters;
        }
        rlt::malloc(ts.device, ts.validation_actor_buffers);
        rlt::init(ts.device, ts.task, ts.validation_envs, ts.rng_eval);

        // info

        std::cout << "Environment Info: \n";
        std::cout << "\t" << "Observation dim: " << CONFIG::ENVIRONMENT::OBSERVATION_DIM << std::endl;
        std::cout << "\t" << "Observation dim privileged: " << CONFIG::ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED << std::endl;
        std::cout << "\t" << "Action dim: " << CONFIG::ENVIRONMENT::ACTION_DIM << std::endl;
        std::cout << "Timesteps: " << CONFIG::STEP_LIMIT << std::endl;

        std::cout << "Ablation Spec: \n";
        std::cout << "\t" << "DISTURBANCE: " << ABLATION_SPEC::DISTURBANCE << std::endl;
        std::cout << "\t" << "OBSERVATION_NOISE: " << ABLATION_SPEC::OBSERVATION_NOISE << std::endl;
        std::cout << "\t" << "ASYMMETRIC_ACTOR_CRITIC: " << ABLATION_SPEC::ASYMMETRIC_ACTOR_CRITIC << std::endl;
        std::cout << "\t" << "ROTOR_DELAY: " << ABLATION_SPEC::ROTOR_DELAY << std::endl;
        std::cout << "\t" << "ACTION_HISTORY: " << ABLATION_SPEC::ACTION_HISTORY << std::endl;
        std::cout << "\t" << "ENABLE_CURRICULUM: " << ABLATION_SPEC::ENABLE_CURRICULUM << std::endl;
        std::cout << "\t" << "RECALCULATE_REWARDS: " << ABLATION_SPEC::RECALCULATE_REWARDS << std::endl;
        std::cout << "\t" << "USE_INITIAL_REWARD_FUNCTION: " << ABLATION_SPEC::USE_INITIAL_REWARD_FUNCTION << std::endl;
        std::cout << "\t" << "INIT_NORMAL: " << ABLATION_SPEC::INIT_NORMAL << std::endl;
        std::cout << "\t" << "EXPLORATION_NOISE_DECAY: " << ABLATION_SPEC::EXPLORATION_NOISE_DECAY << std::endl;
    }

    template <typename CONFIG>
    void step(TrainingState<CONFIG>& ts){
        using TI = typename CONFIG::TI;
        using T = typename CONFIG::T;
        if(ts.step % 10000 == 0){
            std::cout << "Step: " << ts.step << std::endl;
        }
        steps::logger(ts);
        steps::checkpoint(ts);
        steps::validation(ts);
        steps::curriculum(ts);
        rlt::rl::algorithms::td3::loop::step(ts);
        steps::trajectory_collection(ts);
    }
    template <typename CONFIG>
    void destroy(TrainingState<CONFIG>& ts){
        rlt::rl::algorithms::td3::loop::destroy(ts);
        rlt::destroy(ts.device, ts.task);
        rlt::free(ts.device, ts.validation_actor_buffers);
    }
}
