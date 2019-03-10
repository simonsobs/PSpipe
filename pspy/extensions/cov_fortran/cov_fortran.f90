! FFLAGS="-fopenmp -fPIC -Ofast -ffree-line-length-none" f2py-2.7 -c -m cov_fortran cov_fortran.f90 wigner3j_sub.f -lgomp

subroutine calc_cov_spin0_single_win(wcl,cov_array)
    implicit none
    real(8), intent(in)    :: wcl(:)
    real(8), intent(inout) :: cov_array(:,:,:)
    real(8), parameter     :: pi = 3.14159265358979323846264d0
    integer :: l1, l2, l3, info, nlmax, lmin, lmax, i
    real(8) :: l1f(2), fac
    real(8) :: thrcof0(2*size(cov_array,1))
    nlmax = size(cov_array,1)-1
    !$omp parallel do private(l3,l2,l1,fac,info,l1f,thrcof0,lmin,lmax,i) schedule(dynamic)
    do l1 = 2, nlmax
        do l2 = 2, nlmax
            call drc3jj(dble(l1),dble(l2),0d0,0d0,l1f(1),l1f(2),thrcof0, size(thrcof0),info)
            lmin=INT(l1f(1))
            !lmax= INT(l1f(2))
            !write(*,*) nlmax,lmax
            lmax=MIN(nlmax+1,INT(l1f(2)))
            do l3=lmin,lmax
                i   = l3-lmin+1
                cov_array(l1-1,l2-1,1) =cov_array(l1-1,l2-1,1)+ (wcl(l3+1)*thrcof0(i)**2d0)
            end do
        end do
    end do
end subroutine

subroutine calc_cov_spin0(ac_bd,ad_bc,cov_array)
    implicit none
    real(8), intent(in)    :: ac_bd(:),ad_bc(:)
    real(8), intent(inout) :: cov_array(:,:,:)
    real(8), parameter     :: pi = 3.14159265358979323846264d0
    integer :: l1, l2, l3, info, nlmax, lmin, lmax, i
    real(8) :: l1f(2), fac
    real(8) :: thrcof0(2*size(cov_array,1))
    nlmax = size(cov_array,1)-1
    !$omp parallel do private(l3,l2,l1,fac,info,l1f,thrcof0,lmin,lmax,i) schedule(dynamic)
    do l1 = 2, nlmax
        do l2 = 2, nlmax
            call drc3jj(dble(l1),dble(l2),0d0,0d0,l1f(1),l1f(2),thrcof0, size(thrcof0),info)
            lmin=INT(l1f(1))
            !lmax= INT(l1f(2))
            !write(*,*) nlmax,lmax
            lmax=MIN(nlmax+1,INT(l1f(2)))
            do l3=lmin,lmax
                i   = l3-lmin+1
                cov_array(l1-1,l2-1,1) =cov_array(l1-1,l2-1,1)+ (ac_bd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1,l2-1,2) =cov_array(l1-1,l2-1,2)+ (ad_bc(l3+1)*thrcof0(i)**2d0)
            end do
        end do
    end do
end subroutine

subroutine calc_cov_spin0and2_single_win(wcl,cov_array)
    implicit none
    real(8), intent(in)    :: wcl(:)
    real(8), intent(inout) :: cov_array(:,:,:)
    real(8), parameter     :: pi = 3.14159265358979323846264d0
    integer :: l1, l2, l3, info, nlmax, lmin, lmax, i
    real(8) :: l1f(2), fac_00,fac_02,fac_20,fac_22
    real(8) :: thrcof0(2*size(cov_array,1)),thrcof1(2*size(cov_array,1))
    nlmax = size(cov_array,1)-1
    !$omp parallel do private(l3,l2,l1,fac_00,fac_02,fac_20,fac_22,info,l1f,thrcof0,thrcof1,lmin,lmax,i) schedule(dynamic)
    do l1 = 2, nlmax
        do l2 = 2, nlmax
            call drc3jj(dble(l1),dble(l2),0d0,0d0,l1f(1),l1f(2),thrcof0, size(thrcof0),info)
            call drc3jj(dble(l1),dble(l2),-2d0,2d0,l1f(1),l1f(2),thrcof1, size(thrcof1),info)
            lmin=INT(l1f(1))
            !lmax= INT(l1f(2))
            lmax=MIN(nlmax+1,INT(l1f(2)))
            do l3=lmin,lmax
                i   = l3-lmin+1
                cov_array(l1-1,l2-1,1) =cov_array(l1-1,l2-1,1)+ (wcl(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1,l2-1,2) =cov_array(l1-1,l2-1,2)+ (wcl(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)
                cov_array(l1-1,l2-1,3) =cov_array(l1-1,l2-1,3)+ (wcl(l3+1)*thrcof0(i)*thrcof1(i))
            end do
        end do
    end do

end subroutine

subroutine calc_cov_spin0and2(TaTc_TbTd,TaTd_TbTc,PaPc_PbPd,PaPd_PbPc,TaTc_PbPd,TaPd_PbTc,TaTc_TbPd,TaPd_TbTc,TaPc_TbPd,TaPd_TbPc,PaTc_PbPd,PaPd_PbTc,cov_array)
    implicit none
    real(8), intent(in)    :: TaTc_TbTd(:),TaTd_TbTc(:),PaPc_PbPd(:),PaPd_PbPc(:)
    real(8), intent(in)    :: TaTc_PbPd(:),TaPd_PbTc(:),TaTc_TbPd(:),TaPd_TbTc(:)
    real(8), intent(in)    :: TaPc_TbPd(:),TaPd_TbPc(:),PaTc_PbPd(:),PaPd_PbTc(:)
    real(8), intent(inout) :: cov_array(:,:,:)
    real(8), parameter     :: pi = 3.14159265358979323846264d0
    integer :: l1, l2, l3, info, nlmax, lmin, lmax, i
    real(8) :: l1f(2), fac_00,fac_02,fac_20,fac_22
    real(8) :: thrcof0(2*size(cov_array,1)),thrcof1(2*size(cov_array,1))
    nlmax = size(cov_array,1)-1
    !$omp parallel do private(l3,l2,l1,fac_00,fac_02,fac_20,fac_22,info,l1f,thrcof0,thrcof1,lmin,lmax,i) schedule(dynamic)
    do l1 = 2, nlmax
        do l2 = 2, nlmax
            call drc3jj(dble(l1),dble(l2),0d0,0d0,l1f(1),l1f(2),thrcof0, size(thrcof0),info)
            call drc3jj(dble(l1),dble(l2),-2d0,2d0,l1f(1),l1f(2),thrcof1, size(thrcof1),info)
            lmin=INT(l1f(1))
            !lmax= INT(l1f(2))
            lmax=MIN(nlmax+1,INT(l1f(2)))
            do l3=lmin,lmax
                i   = l3-lmin+1
                cov_array(l1-1,l2-1,1) =cov_array(l1-1,l2-1,1)+ (TaTc_TbTd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1,l2-1,2) =cov_array(l1-1,l2-1,2)+ (TaTd_TbTc(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1,l2-1,3) =cov_array(l1-1,l2-1,3)+ (PaPc_PbPd(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)
                cov_array(l1-1,l2-1,4) =cov_array(l1-1,l2-1,4)+ (PaPd_PbPc(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)
                cov_array(l1-1,l2-1,5) =cov_array(l1-1,l2-1,5)+ (TaTc_PbPd(l3+1)*thrcof0(i)*thrcof1(i))
                cov_array(l1-1,l2-1,6) =cov_array(l1-1,l2-1,6)+ (TaPd_PbTc(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1,l2-1,7) =cov_array(l1-1,l2-1,7)+ (TaTc_TbPd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1,l2-1,8) =cov_array(l1-1,l2-1,8)+ (TaPd_TbTc(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1,l2-1,9) =cov_array(l1-1,l2-1,9)+ (TaPc_TbPd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1,l2-1,10) =cov_array(l1-1,l2-1,10)+ (TaPd_TbPc(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1,l2-1,11) =cov_array(l1-1,l2-1,11)+ (PaTc_PbPd(l3+1)*thrcof0(i)*thrcof1(i))
                cov_array(l1-1,l2-1,12) =cov_array(l1-1,l2-1,12)+ (PaPd_PbTc(l3+1)*thrcof0(i)*thrcof1(i))

            end do
        end do
    end do

end subroutine






