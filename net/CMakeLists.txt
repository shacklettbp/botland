add_library(bot-net STATIC
  socket.cpp
  comm.cpp
)

target_link_libraries(bot-net PUBLIC
  bot-rt
)
