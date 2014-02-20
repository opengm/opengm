#add_executable(example-gibbs gibbs.cxx ${headers})
add_executable(example-inference-types inference_types.cxx ${headers})
add_executable(example-markov-chain markov-chain.cxx ${headers})
add_executable(example-matching one_to_one_matching.cxx ${headers})
add_executable(example-quick-start quick_start.cxx ${headers})
add_executable(example-space-types space_types.cxx ${headers})
#add_executable(example-swendsenwang swendsenwang.cxx ${headers})

if(WIN32 OR APPLE)

else( )
   find_library(RT_LIBRARY rt)
   target_link_libraries(example-quick-start rt)
   target_link_libraries(example-matching rt)
   target_link_libraries(example-space-types rt)
endif()


