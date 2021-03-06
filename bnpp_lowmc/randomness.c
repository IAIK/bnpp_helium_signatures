/*
 *  This file is part of the optimized implementation of the Picnic signature
 * scheme. See the accompanying documentation for complete details.
 *
 *  The code is provided under the MIT license, see LICENSE for
 *  more details.
 *  SPDX-License-Identifier: MIT
 */

#include "randomness.h"

#if defined(__linux__) && (__GLIBC__ > 2 || __GLIBC_MINOR__ >= 25)
#include <sys/random.h>

int rand_bytes(uint8_t *dst, size_t len) {
  const ssize_t ret = getrandom(dst, len, GRND_NONBLOCK);
  if (ret < 0 || (size_t)ret != len) {
    return 0;
  }
  return 1;
}
#elif defined(__APPLE__) && defined(HAVE_APPLE_FRAMEWORK)
#include <Security/Security.h>

int rand_bytes(uint8_t *dst, size_t len) {
  if (SecRandomCopyBytes(kSecRandomDefault, len, dst) == errSecSuccess) {
    return 1;
  }
  return 0;
}
#elif defined(__linux__) || defined(__APPLE__)
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(__linux__)
#include <linux/random.h>
#include <sys/ioctl.h>
#endif

#if !defined(O_NOFOLLOW)
#define O_NOFOLLOW 0
#endif
#if !defined(O_CLOEXEC)
#define O_CLOEXEC 0
#endif

int rand_bytes(uint8_t *dst, size_t len) {
  int fd;
  while ((fd = open("/dev/urandom", O_RDONLY | O_NOFOLLOW | O_CLOEXEC, 0)) ==
         -1) {
    // check if we should restart
    if (errno != EINTR) {
      return 0;
    }
  }
#if O_CLOEXEC == 0
  fcntl(fd, F_SETFD, fcntl(fd, F_GETFD) | FD_CLOEXEC);
#endif

#if defined(__linux__)
  int cnt = 0;
  if (ioctl(fd, RNDGETENTCNT, &cnt) == -1) {
    // not ready
    close(fd);
    return 0;
  }
#endif

  while (len) {
    const ssize_t ret = read(fd, dst, len);
    if (ret == -1) {
      if (errno == EAGAIN || errno == EINTR) {
        // retry
        continue;
      }
      close(fd);
      return 0;
    }

    dst += ret;
    len -= ret;
  }

  close(fd);
  return 1;
}
#elif defined(_WIN16) || defined(_WIN32) || defined(_WIN64)
#include <windows.h>

int rand_bytes(uint8_t *dst, size_t len) {
  if (len > ULONG_MAX) {
    return 0;
  }
  if (!BCRYPT_SUCCESS(BCryptGenRandom(NULL, dst, (ULONG)len,
                                      BCRYPT_USE_SYSTEM_PREFERRED_RNG))) {
    return 0;
  }
  return 1;
}
#else
#error "Unsupported OS! Please implement rand_bytes."
#endif
