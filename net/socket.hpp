#pragma once

#include <stdint.h>
#include <utility>

namespace bot {

// For now, we just support IPv4. TODO: Support IPv6
struct Address {
  uint16_t port;
  uint32_t ipv4Address;
};

using TrajectoryID = uint32_t;

struct Socket {
  enum class Type {
    Stream,
    Datagram
  };

  static Socket make(Type type);
  void setRecvBufferSize(uint32_t size);
  void setSendBufferSize(uint32_t size);

  void setReuseAddr();

  // Returns 0xFFFF upon failure, and host order port number
  uint16_t bindToPort(Address addr);

  std::pair<Socket, Address> acceptConnection();

  void setToListening(uint32_t max_clients);
  void setBlockingMode(bool blocking);

  template <typename T>
  uint32_t receive(T *buf, uint32_t buf_size)
  {
    return receiveImpl((char *)buf, buf_size);
  }

  bool send(const char *buf, uint32_t buf_size);

  bool connectTo(const char *addr_name, uint16_t port);



  int32_t hdl;
  Type type;

private:
  uint32_t receiveImpl(char *buf, uint32_t buf_size);
};
    
}
