! Fortran do concurrent — standard-language GPU offload (no directives).
! build: ifx -fiopenmp -fopenmp-targets=spir64 -fopenmp-target-do-concurrent dc_triad.f90 -o dc_triad
program dc_triad
  implicit none
  integer, parameter :: N = 1024*1024
  integer, parameter :: STEPS = 1000
  real, parameter :: a = 2.0
  real, allocatable :: x(:), y(:), z(:)
  integer :: i, s
  integer(8) :: c0, c1, rate
  real(8) :: ms, best_ms, err

  allocate(x(N), y(N), z(N))
  x = 2.0; y = 1.0; z = 0.0

  call system_clock(count_rate=rate)
  best_ms = huge(1.0d0)
  do s = 1, STEPS
     call system_clock(c0)
     ! Standard Fortran. The compiler offloads this loop to the GPU.
     do concurrent (i = 1:N)
        z(i) = x(i) + a*y(i)
     end do
     call system_clock(c1)
     ms = 1.0d3 * (c1 - c0) / rate
     best_ms = min(best_ms, ms)
  end do

  ! Reduction: residual error sum (z-4)^2.
  err = 0
  do concurrent (i = 1:N) reduce(+:err)
     err = err + (z(i) - 4.0)**2
  end do

  print '(A,I0,A,F0.3,A,ES10.3)', 'do conc.  N=', N, '  min_ms=', best_ms, '  residual=', err
  deallocate(x, y, z)
end program
