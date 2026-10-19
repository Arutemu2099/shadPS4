PGO profile of the inFAMOUS Second Son build v1.0 (release EXE 9E93FFDD, built from commit 6029396d).

Training: the game's forest, fire and open-world city scenes on an instrumented build (clang-cl, Windows).
To reproduce the release EXE: configure the x64-Clang-RelWithDebInfo preset with
  -DASTRA_AUTO_INLINE=ON -DASTRA_THINLTO=ON -DASTRA_PGO_USE_FILE=<path to this folder>/second-son-v1.0.profdata
and build. Without the profile the build works but is slower.
