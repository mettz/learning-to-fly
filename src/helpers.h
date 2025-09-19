namespace learning_to_fly{
    namespace helpers{
        // Pretty-print helpers for environment parameters
        template <typename T, std::size_t N>
        void print_array_inline(const char* label, const T (&arr)[N]){
            std::cout << "\t" << label << ": [";
            for(std::size_t i = 0; i < N; i++){
                if(i){ std::cout << ", "; }
                std::cout << arr[i];
            }
            std::cout << "]\n";
        }
        template <typename T, std::size_t N, std::size_t M>
        void print_array2_inline(const char* label, const T (&arr)[N][M]){
            std::cout << "\t" << label << ":\n";
            for(std::size_t i = 0; i < N; i++){
                std::cout << "\t  [";
                for(std::size_t j = 0; j < M; j++){
                    if(j){ std::cout << ", "; }
                    std::cout << arr[i][j];
                }
                std::cout << "]\n";
            }
        }

        // Reward printers (overloads for known reward types)
        namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;
        template <typename T>
        void dump_reward(const rlt::rl::environments::multirotor::parameters::reward_functions::Squared<T>& r){
            std::cout << "\tmdp.reward (Squared):\n";
            std::cout << "\t  non_negative: " << r.non_negative << "\n";
            std::cout << "\t  scale: " << r.scale << "\n";
            std::cout << "\t  constant: " << r.constant << "\n";
            std::cout << "\t  termination_penalty: " << r.termination_penalty << "\n";
            std::cout << "\t  weights: position=" << r.position
                      << ", orientation=" << r.orientation
                      << ", linear_velocity=" << r.linear_velocity
                      << ", angular_velocity=" << r.angular_velocity
                      << ", linear_acceleration=" << r.linear_acceleration
                      << ", angular_acceleration=" << r.angular_acceleration
                      << ", action=" << r.action << "\n";
            std::cout << "\t  action_baseline: " << r.action_baseline << "\n";
        }
        template <typename T>
        void dump_reward(const rlt::rl::environments::multirotor::parameters::reward_functions::AbsExp<T>& r){
            std::cout << "\tmdp.reward (AbsExp):\n";
            std::cout << "\t  scale: " << r.scale << ", scale_inner: " << r.scale_inner << "\n";
            std::cout << "\t  weights: position=" << r.position
                      << ", orientation=" << r.orientation
                      << ", linear_velocity=" << r.linear_velocity
                      << ", angular_velocity=" << r.angular_velocity
                      << ", linear_acceleration=" << r.linear_acceleration
                      << ", angular_acceleration=" << r.angular_acceleration
                      << ", action=" << r.action << "\n";
            std::cout << "\t  action_baseline: " << r.action_baseline << "\n";
        }
        template <typename T, typename TI, TI N>
        void dump_reward(const rlt::rl::environments::multirotor::parameters::reward_functions::AbsExpMultiModal<T, TI, N>& r){
            std::cout << "\tmdp.reward (AbsExpMultiModal, modes=" << (int)N << "):\n";
            for(TI i=0;i<N;i++){
                std::cout << "\t  mode[" << (int)i << "]:\n";
                dump_reward(r.modes[i]);
            }
        }
        template <typename T>
        void dump_reward(const rlt::rl::environments::multirotor::parameters::reward_functions::SqExp<T>& r){
            std::cout << "\tmdp.reward (SqExp):\n";
            std::cout << "\t  additive_constant: " << r.additive_constant << ", scale: " << r.scale << ", scale_inner: " << r.scale_inner << "\n";
            std::cout << "\t  weights: position=" << r.position
                      << ", orientation=" << r.orientation
                      << ", linear_velocity=" << r.linear_velocity
                      << ", angular_velocity=" << r.angular_velocity
                      << ", linear_acceleration=" << r.linear_acceleration
                      << ", angular_acceleration=" << r.angular_acceleration
                      << ", action=" << r.action << "\n";
            std::cout << "\t  action_baseline: " << r.action_baseline << "\n";
        }
        template <typename T, typename TI, TI N>
        void dump_reward(const rlt::rl::environments::multirotor::parameters::reward_functions::SqExpMultiModal<T, TI, N>& r){
            std::cout << "\tmdp.reward (SqExpMultiModal, modes=" << (int)N << "):\n";
            for(TI i=0;i<N;i++){
                std::cout << "\t  mode[" << (int)i << "]:\n";
                dump_reward(r.modes[i]);
            }
        }
        template <typename T>
        void dump_reward(const rlt::rl::environments::multirotor::parameters::reward_functions::Absolute<T>& r){
            std::cout << "\tmdp.reward (Absolute):\n";
            std::cout << "\t  non_negative: " << r.non_negative << ", scale: " << r.scale << ", constant: " << r.constant << ", termination_penalty: " << r.termination_penalty << "\n";
            std::cout << "\t  weights: position=" << r.position
                      << ", orientation=" << r.orientation
                      << ", linear_velocity=" << r.linear_velocity
                      << ", angular_velocity=" << r.angular_velocity
                      << ", linear_acceleration=" << r.linear_acceleration
                      << ", angular_acceleration=" << r.angular_acceleration
                      << ", action=" << r.action << "\n";
            std::cout << "\t  action_baseline: " << r.action_baseline << "\n";
        }
        // Fallback: unknown reward type
        template <typename REWARD>
        void dump_reward(const REWARD&){
            std::cout << "\tmdp.reward: <unknown reward type>\n";
        }

        // Print the full environment parameters (dynamics, integration, mdp, disturbances)
        template <typename PARAMETERS>
        void dump_environment_parameters(const PARAMETERS& p){
            std::cout << "Integration:\n";
            std::cout << "\tdt: " << p.integration.dt << "\n";

            std::cout << "Dynamics:\n";
            std::cout << "\tmass: " << p.dynamics.mass << "\n";
            print_array_inline("gravity [x,y,z]", p.dynamics.gravity);
            print_array2_inline("J", p.dynamics.J);
            print_array2_inline("J_inv", p.dynamics.J_inv);
            std::cout << "\trpm_time_constant: " << p.dynamics.rpm_time_constant << "\n";
            std::cout << "\taction_limit.min: " << p.dynamics.action_limit.min << ", max: " << p.dynamics.action_limit.max << "\n";
            print_array2_inline("rotor_positions [N x 3]", p.dynamics.rotor_positions);
            print_array2_inline("rotor_thrust_directions [N x 3]", p.dynamics.rotor_thrust_directions);
            print_array2_inline("rotor_torque_directions [N x 3]", p.dynamics.rotor_torque_directions);
            print_array_inline("thrust_constants [3]", p.dynamics.thrust_constants);
            std::cout << "\ttorque_constant: " << p.dynamics.torque_constant << "\n";

            std::cout << "MDP:\n";
            std::cout << "\tinit.guidance: " << p.mdp.init.guidance << "\n";
            std::cout << "\tinit.max_position: " << p.mdp.init.max_position << ", max_angle: " << p.mdp.init.max_angle << "\n";
            std::cout << "\tinit.max_linear_velocity: " << p.mdp.init.max_linear_velocity << ", max_angular_velocity: " << p.mdp.init.max_angular_velocity << "\n";
            std::cout << "\tinit.relative_rpm: " << p.mdp.init.relative_rpm << ", min_rpm: " << p.mdp.init.min_rpm << ", max_rpm: " << p.mdp.init.max_rpm << "\n";
            std::cout << "\tobservation_noise: pos=" << p.mdp.observation_noise.position
                      << ", orient=" << p.mdp.observation_noise.orientation
                      << ", lin_vel=" << p.mdp.observation_noise.linear_velocity
                      << ", ang_vel=" << p.mdp.observation_noise.angular_velocity << "\n";
            std::cout << "\taction_noise.normalized_rpm: " << p.mdp.action_noise.normalized_rpm << "\n";
            std::cout << "\ttermination.enabled: " << p.mdp.termination.enabled
                      << ", pos_thr: " << p.mdp.termination.position_threshold
                      << ", lin_vel_thr: " << p.mdp.termination.linear_velocity_threshold
                      << ", ang_vel_thr: " << p.mdp.termination.angular_velocity_threshold << "\n";
            dump_reward(p.mdp.reward);

            std::cout << "Disturbances:\n";
            std::cout << "\trandom_force.mean/std: " << p.disturbances.random_force.mean << "/" << p.disturbances.random_force.std << "\n";
            std::cout << "\trandom_torque.mean/std: " << p.disturbances.random_torque.mean << "/" << p.disturbances.random_torque.std << "\n";
        }
        template <typename ABLATION_SPEC>
        std::string ablation_name(){
            std::string n = "";
            n += std::string("d") + (ABLATION_SPEC::DISTURBANCE ? "+"  : "-");
            n += std::string("o") + (ABLATION_SPEC::OBSERVATION_NOISE ? "+"  : "-");
            n += std::string("a") + (ABLATION_SPEC::ASYMMETRIC_ACTOR_CRITIC ? "+"  : "-");
            n += std::string("r") + (ABLATION_SPEC::ROTOR_DELAY ? "+"  : "-");
            n += std::string("h") + (ABLATION_SPEC::ACTION_HISTORY ? "+"  : "-");
            n += std::string("c") + (ABLATION_SPEC::ENABLE_CURRICULUM ? "+"  : "-");
            n += std::string("f") + (ABLATION_SPEC::USE_INITIAL_REWARD_FUNCTION ? "+"  : "-");
            n += std::string("w") + (ABLATION_SPEC::RECALCULATE_REWARDS ? "+"  : "-");
            n += std::string("e") + (ABLATION_SPEC::EXPLORATION_NOISE_DECAY ? "+"  : "-");
            return n;
        }
        template <typename ABLATION_SPEC, typename CONFIG>
        std::string run_name(typename CONFIG::TI seed) {
            std::stringstream run_name_ss;
            run_name_ss << "";
            auto now = std::chrono::system_clock::now();
            auto local_time = std::chrono::system_clock::to_time_t(now);
            std::tm *tm = std::localtime(&local_time);
            run_name_ss << "" << std::put_time(tm, "%Y_%m_%d_%H_%M_%S");
            if constexpr (CONFIG::BENCHMARK) {
                run_name_ss << "_BENCHMARK";
            }
            run_name_ss << "_" << helpers::ablation_name<ABLATION_SPEC>();
            run_name_ss << "_" << std::setw(3) << std::setfill('0') << seed;
            return run_name_ss.str();
        }
    }
}
