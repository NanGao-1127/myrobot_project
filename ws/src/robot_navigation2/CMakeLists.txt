cmake_minimum_required(VERSION 3.8)
project(robot_navigation2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# 安装 Python 节点（注意路径！）
install(PROGRAMS
  src/explore.py
  DESTINATION lib/${PROJECT_NAME}
)



ament_package()
