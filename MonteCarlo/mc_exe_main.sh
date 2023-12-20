#!/bin/bash


for gamm in 0.1 0.25 0.5 0.9 0.99
  do
  for semente in 960703545 1277478588 1936856304 186872697 1859168769 1598189534 1822174485 1871883252 694388766 188312339 773370613 2125204119 2041095833 1384311643 1000004583 358485174 1695858027 762772169 437720306 939612284
    do
      param="gamma"
      echo $semente

     type_e="deterministic"
     reward="unit"
     python3 main.py --seed="${semente}" --value="${gamm}" --testing_agent_param_name="${param}" --type_env="${type_e}" --rewards="${reward}" &
     sleep 5

     type_e="deterministic"
     reward="weighted"
     echo $semente
     python3 main.py --seed="${semente}" --value="${gamm}" --testing_agent_param_name="${param}" --type_env="${type_e}" --rewards="${reward}" &
     sleep 5

      type_e="stochastic"
      reward="unit"
      python3 main.py --seed="${semente}" --value="${gamm}" --testing_agent_param_name="${param}" --type_env="${type_e}" --rewards="${reward}" &
      sleep 5

      type_e="stochastic"
      reward="weighted"
      python3 main.py --seed="${semente}" --value="${gamm}" --testing_agent_param_name="${param}" --type_env="${type_e}" --rewards="${reward}"
      sleep 5
    done
  done


#  for min_eps in 0.05 0.1 0.3 0.5 0.7
#  do
#  for semente in 960703545 1277478588 1936856304 186872697 1859168769 1598189534 1822174485 1871883252 694388766 188312339 773370613 2125204119 2041095833 1384311643 1000004583 358485174 1695858027 762772169 437720306 939612284
#    do
#      param="min_epsilon"
#      echo $semente
#
#      type_e="deterministic"
#      reward="unit"
#      python3 main.py --seed="${semente}" --value="${min_eps}" --testing_agent_param_name="${param}" --type_env="${type_e}" --rewards="${reward}" &
#      sleep 5
#
#      type_e="deterministic"
#      reward="weighted"
#      python3 main.py --seed="${semente}" --value="${min_eps}" --testing_agent_param_name="${param}" --type_env="${type_e}" --rewards="${reward}" &
#      sleep 5

#      type_e="stochastic"
#      reward="unit"
#      python3 main.py --seed="${semente}" --value="${min_eps}" --testing_agent_param_name="${param}" --type_env="${type_e}" --rewards="${reward}" &
#      sleep 5
#
#      type_e="stochastic"
#      reward="weighted"
#      python3 main.py --seed="${semente}" --value="${min_eps}" --testing_agent_param_name="${param}" --type_env="${type_e}" --rewards="${reward}"
#      sleep 5
#    done
#  done
