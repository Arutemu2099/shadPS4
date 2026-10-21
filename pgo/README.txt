PGO profiles of the inFAMOUS Second Son builds (clang-cl, Windows).

second-son-v1.1.profdata - the v1.1 release (the commit tagged second-son-v1.1).
second-son-v1.0.profdata - the v1.0 release (EXE 9E93FFDD, built from commit 6029396d); v1.0.1 and v1.0.2 used it too.

Training: the game's forest, fire and open-world city scenes on an instrumented build.
To reproduce a release EXE: configure the x64-Clang-RelWithDebInfo preset with
  -DASTRA_AUTO_INLINE=ON -DASTRA_THINLTO=ON -DASTRA_PGO_USE_FILE=<path to this folder>/second-son-v1.1.profdata
and build. Without the profile the build works but is slower.
