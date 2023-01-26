#include <sys/stat.h>
#include <assert.h>
#include <fstream> 
#include <cstdio>
#include <unistd.h>
#include <fcntl.h>
#include <libgen.h>
#include <string>
#include <cstring>
#include <iostream>
#include <filesystem>

#include "device.hpp"
#include "platform.hpp"

namespace fs = std::filesystem;

bool mkDir(const fs::path &file_path)
{
  size_t pos = 0;
  auto ret_val = true;

  std::string dir_path(file_path);
  if (!fs::is_directory((file_path)))
    dir_path = file_path.parent_path();

  while (ret_val && pos != std::string::npos) {
    pos = dir_path.find('/', pos + 1);
    const auto dir = fs::path(dir_path.substr(0, pos));
    if (!fs::exists(dir)) {
      ret_val = fs::create_directory(dir);
    }
  }

  return ret_val;
}

int fileBcastNodes(const fs::path srcPath,
                   const fs::path dstPath,
                   int rankCompile,
                   MPI_Comm comm,
                   int verbose)
{
  int rank;
  MPI_Comm_rank(comm, &rank);

  int err = 0;
  if (rank == rankCompile && !fs::exists(srcPath)) {
    err++;
    std::cout << __func__ << ": cannot stat "
              << "" << srcPath << ":"
              << " No such file or directory\n";
  }
  MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_INT, MPI_SUM, comm);
  if (err)
    return EXIT_FAILURE;

  const auto path0 = fs::current_path();

  int localRank;
  const int localRankRoot = 0;

  int color = MPI_UNDEFINED;
  MPI_Comm commLocal;
  {
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &commLocal);
    MPI_Comm_rank(commLocal, &localRank);
    if (localRank == localRankRoot)
      color = 1;
    if (rank == rankCompile)
      color = 1;
  }
  MPI_Comm commNode;
  MPI_Comm_split(comm, color, rank, &commNode);

  int nodeRank = -1;
  int nodeRankRoot = -1;
  if (color != MPI_UNDEFINED)
    MPI_Comm_rank(commNode, &nodeRank);
  if (rank == rankCompile)
    nodeRankRoot = nodeRank;
  MPI_Bcast(&nodeRankRoot, 1, MPI_INT, rankCompile, comm);

  // generate file list
  std::vector<std::string> fileList;
  if (nodeRank == nodeRankRoot) {
    if (!fs::is_directory((srcPath))) {
      fileList.push_back(srcPath);
    }
    else {
      for (const auto &dirEntry : fs::recursive_directory_iterator(srcPath)) {
        if (dirEntry.is_regular_file())
          fileList.push_back(dirEntry.path());
      }
    }
  }
  int nFiles = (nodeRank == nodeRankRoot) ? fileList.size() : 0;
  MPI_Bcast(&nFiles, 1, MPI_INT, rankCompile, comm);
  if (!nFiles)
    return EXIT_SUCCESS;

  // bcast file list
  for (int i = 0; i < nFiles; i++) {
    int bufSize = (rank == rankCompile) ? fileList.at(i).size() : 0;
    MPI_Bcast(&bufSize, 1, MPI_INT, rankCompile, comm);

    auto buf = (char *)std::malloc(bufSize * sizeof(char));
    if (rank == rankCompile)
      std::strncpy(buf, fileList.at(i).c_str(), bufSize);
    MPI_Bcast(buf, bufSize, MPI_CHAR, rankCompile, comm);
    if (rank != rankCompile)
      fileList.push_back(std::string(buf, 0, bufSize));
    free(buf);
  }

  for (const auto &file : fileList) {
    int bufSize = 0;
    const std::string filePath = dstPath / fs::path(file);

    unsigned char *buf = nullptr;
    if (color != MPI_UNDEFINED) {
      if (nodeRank == nodeRankRoot)
        bufSize = fs::file_size(file);
      MPI_Bcast(&bufSize, 1, MPI_INT, nodeRankRoot, commNode);

      if (bufSize > std::numeric_limits<int>::max()) {
        if (rank == rankCompile)
          std::cout << __func__ << ": file size of "
                    << "" << file << " too large!\n";
        return EXIT_FAILURE;
      }

      buf = (unsigned char *)std::malloc(bufSize * sizeof(unsigned char));

      if (nodeRank == nodeRankRoot) {
        std::ifstream input(file, std::ios::in | std::ios::binary);
        std::stringstream sstr;
        input >> sstr.rdbuf();
        input.close();
        std::memcpy(buf, sstr.str().c_str(), bufSize);
      }
      MPI_Bcast(buf, bufSize, MPI_BYTE, nodeRankRoot, commNode);

      if (nodeRank == nodeRankRoot && verbose)
        std::cout << __func__ << ": " << file << " -> " << filePath << " (" << bufSize << " bytes)"
                  << std::endl;
    }

    // write file to node-local storage;
    if (localRank == localRankRoot)
      mkDir(filePath); // create directory and parents if they don't already exist

    MPI_File fh;
    MPI_File_open(commLocal, filePath.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);

    if (localRank == localRankRoot) {
      MPI_Status status;
      MPI_File_write_at(fh, 0, buf, bufSize, MPI_BYTE, &status);
    }
    free(buf);

    MPI_File_sync(fh);
    MPI_Barrier(commLocal);
    MPI_File_sync(fh);
    MPI_File_close(&fh);
  }

  fs::current_path(path0);

  return EXIT_SUCCESS;
}

void fileSync(const char * file)
{
  std::string dir;
  {
    // POSIX allows dirname to overwrite input
    const int len = std::char_traits<char>::length(file);
    char *tmp = (char*) malloc((len+1) * sizeof(char));
    strncpy(tmp, file, len);
    tmp[len] = '\0'; 
    dir.assign(dirname(tmp));
    free(tmp);
  }

  int fd; 
  fd = open(file, O_RDONLY);
  fsync(fd);
  close(fd);

  fd = open(dir.c_str(), O_RDONLY);
  fsync(fd);
  close(fd);
}

bool isFileNewer(const char *file1, const char *file2)
{
  struct stat s1, s2;
  if (lstat(file1, &s1) != 0) assert(1);
  if (lstat(file2, &s2) != 0) return true; 
  if (s1.st_mtime > s2.st_mtime) 
    return true;
  else
    return false;	  
}

void copyFile(const char *srcFile, const char *dstFile)
{
  std::ifstream src(srcFile, std::ios::binary);
  std::ofstream dst(dstFile, std::ios::trunc | std::ios::binary);
  dst << src.rdbuf();
  src.close();
  dst.close();
  fileSync(dstFile);
}

bool fileExists(const char *file)
{
  return realpath(file, NULL);
}

bool isFileEmpty(const char *file)
{
  std::ifstream f(file);
  const bool isEmpty = f.peek() == std::ifstream::traits_type::eof();
  f.close();
  return isEmpty;
}
