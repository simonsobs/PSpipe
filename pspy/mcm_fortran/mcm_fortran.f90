! FFLAGS="-fopenmp -fPIC -Ofast -ffree-line-length-none" f2py-2.7 -c -m mcm_fortran mcm_fortran.f90 wigner3j_sub.f -lgomp

subroutine calc_mcm_spin0(wcl,wbl, mcm)
    implicit none
    real(8), intent(in)    :: wcl(:),wbl(:)
    real(8), intent(inout) :: mcm(:,:)
    real(8), parameter     :: pi = 3.14159265358979323846264d0
    integer :: l1, l2, l3, info, nlmax, lmin, lmax, i
    real(8) :: l1f(2), fac
    real(8) :: thrcof0(2*size(mcm,1))
    nlmax = size(mcm,1)-1
    !$omp parallel do private(l3,l2,l1,fac,info,l1f,thrcof0,lmin,lmax,i) schedule(dynamic)
    do l1 = 2, nlmax
        fac=(2*l1+1)/(4*pi)*wbl(l1+1)
        do l2 = 2, nlmax
            call drc3jj(dble(l1),dble(l2),0d0,0d0,l1f(1),l1f(2),thrcof0, size(thrcof0),info)
            lmin=INT(l1f(1))
            !lmax= INT(l1f(2))
            !write(*,*) nlmax,lmax
            lmax=MIN(nlmax+1,INT(l1f(2)))
            do l3=lmin,lmax
                i   = l3-lmin+1
                mcm(l1-1,l2-1) =mcm(l1-1,l2-1)+ fac*(wcl(l3+1)*thrcof0(i)**2d0)
            end do
        end do
    end do
end subroutine

subroutine calc_mcm_spin0and2(wcl_00,wcl_02, wcl_20, wcl_22, wbl_00,wbl_02, wbl_20, wbl_22 , mcm_array)
    implicit none
    real(8), intent(in)    :: wcl_00(:),wcl_02(:),wcl_20(:),wcl_22(:),wbl_00(:),wbl_02(:), wbl_20(:), wbl_22(:)
    real(8), intent(inout) :: mcm_array(:,:,:)
    real(8), parameter     :: pi = 3.14159265358979323846264d0
    integer :: l1, l2, l3, info, nlmax, lmin, lmax, i
    real(8) :: l1f(2), fac_00,fac_02,fac_20,fac_22
    real(8) :: thrcof0(2*size(mcm_array,1)),thrcof1(2*size(mcm_array,1))
    nlmax = size(mcm_array,1)-1
    !$omp parallel do private(l3,l2,l1,fac_00,fac_02,fac_20,fac_22,info,l1f,thrcof0,thrcof1,lmin,lmax,i) schedule(dynamic)
    do l1 = 2, nlmax
        fac_00=(2*l1+1)/(4*pi)*wbl_00(l1+1)
        fac_02=(2*l1+1)/(4*pi)*wbl_02(l1+1)
        fac_20=(2*l1+1)/(4*pi)*wbl_20(l1+1)
        fac_22=(2*l1+1)/(4*pi)*wbl_22(l1+1)

        do l2 = 2, nlmax
            call drc3jj(dble(l1),dble(l2),0d0,0d0,l1f(1),l1f(2),thrcof0, size(thrcof0),info)
            call drc3jj(dble(l1),dble(l2),-2d0,2d0,l1f(1),l1f(2),thrcof1, size(thrcof1),info)
            lmin=INT(l1f(1))
            !lmax= INT(l1f(2))
            lmax=MIN(nlmax+1,INT(l1f(2)))
            do l3=lmin,lmax
                i   = l3-lmin+1
                mcm_array(l1-1,l2-1,1) =mcm_array(l1-1,l2-1,1)+ fac_00*(wcl_00(l3+1)*thrcof0(i)**2d0)
                mcm_array(l1-1,l2-1,2) =mcm_array(l1-1,l2-1,2)+ fac_02*(wcl_02(l3+1)*thrcof0(i)*thrcof1(i))
                mcm_array(l1-1,l2-1,3) =mcm_array(l1-1,l2-1,3)+ fac_20*(wcl_20(l3+1)*thrcof0(i)*thrcof1(i))
                mcm_array(l1-1,l2-1,4) =mcm_array(l1-1,l2-1,4)+ fac_22*(wcl_22(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)
                mcm_array(l1-1,l2-1,5) =mcm_array(l1-1,l2-1,5)+ fac_22*(wcl_22(l3+1)*thrcof1(i)**2*(1-(-1)**(l1+l2+l3))/2)
            end do
        end do
    end do

end subroutine


subroutine bin_mcm(mcm, binLo,binHi, binsize, mbb,doDl)
    ! Bin the given mode coupling matrix mcm(0:lmax,0:lmax) into
    ! mbb(nbin,nbin) using bins of the given binsize
    implicit none
    real(8), intent(in)    :: mcm(:,:)
    integer, intent(in)    :: binLo(:),binHi(:),binsize(:),doDl
    real(8), intent(inout) :: mbb(:,:)
    integer :: b1, b2, l1, l2, lmax
    lmax = size(mcm,1)-1
    mbb  = 0
    do b2=1,size(mbb,1)
        do b1=1,size(mbb,1)
            do l2=binLo(b2),binHi(b2)
                do l1=binLo(b1),binHi(b1)
                    if (doDl .eq. 1) then
                        mbb(b1,b2)=mbb(b1,b2) + mcm(l1-1,l2-1)*l2*(l2+1d0)/(l1*(l1+1d0)) !*mcm(l2-1,l3-1)
                    else
                        mbb(b1,b2)=mbb(b1,b2) + mcm(l1-1,l2-1)
                    end if
                end do
            end do
            mbb(b1,b2) = mbb(b1,b2) / binsize(b2)

        end do
    end do
end subroutine

subroutine binning_matrix(mcm, binLo,binHi, binsize, bbl,doDl)
    implicit none
    real(8), intent(in)    :: mcm(:,:)
    integer(8), intent(in)    :: binLo(:),binHi(:),binsize(:),doDl
    real(8), intent(inout) :: bbl(:,:)
    integer(8) :: b2, l1, l2,lmax

    lmax = size(mcm,1)-1
    ! mcm is transposed
    ! compute \sum_{l'} M_l'l
    do l1=2,lmax
        do b2=1,size(binLo)
            do l2=binLo(b2),binHi(b2)
                if (doDl .eq. 1) then
                    bbl(l1-1,b2)=bbl(l1-1,b2)+mcm(l1-1,l2-1)*l2*(l2+1d0)/(l1*(l1+1d0))
                else
                     bbl(l1-1,b2)=bbl(l1-1,b2)+mcm(l1-1,l2-1)
                end if
            end do
            bbl(l1-1,b2)=bbl(l1-1,b2)/(binsize(b2)*1d0)
        end do
    end do
end subroutine




