clc
clear
close all

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Author: Jonathan Bedford, Tectonic Geodesy, Ruhr University Bochum, DE
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% This Octave script generates random rectangular faults as training samples for
% the surrogate model.  If a fault protrudes above z=0, the fault is trimmed
% but the fault width value is left unchanged.

% To generate displacement predictions,
% the script uses the TDdipsHS.m script from the Nikhoo & Walter 2015 paper:

% Nikkhoo, M., Walter, T. R. (2015): Triangular dislocation: an analytical,
% artefact-free solution. - Geophysical Journal International, 201,
% 1117-1139. doi: 10.1093/gji/ggv035

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %


N_times = 250_000;  ### number of training samples to generate


x_cen_bounds = [-1,1];
y_cen_bounds = [-1,1];
z_cen_bounds = [-1,0];  % these three parameters control the bounds of fault centroid


fault_length_bounds = [0,1];
fault_width_bounds = [0,1];
dip_bounds = [(0+eps),(360-eps)];# eps to avoid apparent singularities
strike_bounds = [(0+eps),[360-eps]];
rake_bounds = [(0+eps),[360-eps]];

[X,Y] = meshgrid(linspace(-1,1,32),linspace(-1,1,32)); % surface bounds and discretization
Z = 0*X;


### empty marices to fill as we proceed through the for-loop below
targets = zeros(N_times,size(X,1),size(X,2),3);

x_cen_inputs = nan(N_times,1);
y_cen_inputs = nan(N_times,1);
z_cen_inputs = nan(N_times,1);
fault_length_inputs = nan(N_times,1);
fault_width_inputs = nan(N_times,1);
dip_inputs = nan(N_times,1);
strike_inputs = nan(N_times,1);
rake_inputs = nan(N_times,1);


for i = 1:N_times

  t1 = time;
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  % %  Generating random fault parameters withihn pre-defined bounds.
  x_cen = min(x_cen_bounds)+range(x_cen_bounds)*rand(1);
  y_cen = min(y_cen_bounds)+range(y_cen_bounds)*rand(1);
  z_cen = min(z_cen_bounds)+range(z_cen_bounds)*rand(1);

  fault_length = min(fault_length_bounds)+range(fault_length_bounds)*rand(1);
  fault_width = min(fault_width_bounds)+range(fault_width_bounds)*rand(1);
  dip = min(dip_bounds)+range(dip_bounds)*rand(1);
  strike = min(strike_bounds)+range(strike_bounds)*rand(1);
  rake = min(rake_bounds)+range(rake_bounds)*rand(1);

  % If the dip is orthogonal or parallel to Z axis, fixing this (to avoid singularities).
  modulo_dip_90 = mod(dip,90);
  if modulo_dip_90==0
    dip+=0.01;
  endif

  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  % % Building fault at 0 strike
  % We have fault_length, fault_width, and dip.
  % Use fault_length, fault_width, and dip to specify the
  % x,y,z positions of B, C, and D.
  A = [0,0,0];
  B = [fault_width*cosd(dip),0,-fault_width*sind(dip)];
  C = [fault_width*cosd(dip),-fault_length,-fault_width*sind(dip)];
  D = [0,-fault_length,0];
  rect = [A;B;C;D];% each row is a corner of the rectangular fault

  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  % % % Removing the mean location of the rectangle
  rect = rect - mean(rect,1);

  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  % % % Rotating about the origin in x,y space
  R = [cosd(-strike), -sind(-strike);...
       sind(-strike), cosd(-strike)];
  rect(:,1:2) = (R*(rect(:,1:2)'))';


  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  % % Moving the fault centre to the location of choice
  rect(:,1) = rect(:,1) + x_cen;
  rect(:,2) = rect(:,2) + y_cen;
  rect(:,3) = rect(:,3) + z_cen;


  ### if rectangle is poking out the surface, pulling the protruding points back
  ### slightly below the surface
  max_z = max(rect(:,3));
  if max_z>0
    h = max(rect(:,3));
    H = range(rect(:,3));
    updip_length = (1-h/H)*fault_width;
    % Depending on dip, downdip edges are at (B and C) or (D and A)
    % which corresponds to indices............(2 and 3) or (
    if rect(1,3) > rect(2,3)
      is_case = 1
    else
      is_case = 2
    end

    if is_case == 1;
      vec_B_A = rect(1,:)-rect(2,:); vec_B_A = vec_B_A./norm(vec_B_A);
      vec_C_D = rect(4,:)-rect(3,:); vec_C_D = vec_C_D./norm(vec_C_D);
      % putting points A and D just below the surface
      rect(1,:) = rect(2,:)+vec_B_A*(updip_length-1e-9);
      rect(4,:) = rect(3,:)+vec_C_D*(updip_length-1e-9);
    else
      vec_A_B = rect(2,:)-rect(1,:); vec_A_B = vec_A_B./norm(vec_A_B);
      vec_D_C = rect(3,:)-rect(4,:); vec_D_C = vec_D_C./norm(vec_D_C);
      % putting points B and C just below the surface
      rect(2,:) = rect(1,:)+vec_A_B*(updip_length-1e-9);
      rect(3,:) = rect(4,:)+vec_D_C*(updip_length-1e-9);
    endif
  end


  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  % % Making two triangles from the four vertices of the rectangle
  tri_1 = [rect(1,:);rect(3,:);rect(2,:)];
  tri_2 = [rect(1,:);rect(4,:);rect(3,:)];

  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  %  Making a dip multiplier (to make sure output of dip angle is circularly continuous)
  DM = 1;
  if (dip > 180)*(dip < 360) == 1
    DM = -1;
  endif


  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  % % Making Green's functions for each triangle
  [SS1_e,SS1_n,SS1_u] = TDdispHS(X,Y,Z,tri_1(1,:),tri_1(2,:),...
                          tri_1(3,:),DM*1,0,0,0.25);

  [DS1_e,DS1_n,DS1_u] = TDdispHS(X,Y,Z,tri_1(1,:),tri_1(2,:),...
                          tri_1(3,:),0,DM*1,0,0.25);


  [SS2_e,SS2_n,SS2_u] = TDdispHS(X,Y,Z,tri_2(1,:),tri_2(2,:),...
                          tri_2(3,:),DM*1,0,0,0.25);

  [DS2_e,DS2_n,DS2_u] = TDdispHS(X,Y,Z,tri_2(1,:),tri_2(2,:),...
                          tri_2(3,:),0,DM*1,0,0.25);

  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  % % Making Green's functions for rectangle by summing predictions of
  %  each triangle

  SS_e = SS1_e+SS2_e;
  SS_n = SS1_n+SS2_n;
  SS_u = SS1_u+SS2_u;

  DS_e = DS1_e+DS2_e;
  DS_n = DS1_n+DS2_n;
  DS_u = DS1_u+DS2_u;

  SS_e = reshape(SS_e,size(X));
  SS_n = reshape(SS_n,size(X));
  SS_u = reshape(SS_u,size(X));

  DS_e = reshape(DS_e,size(X));
  DS_n = reshape(DS_n,size(X));
  DS_u = reshape(DS_u,size(X));

  #### Making the X and Y for the ML
  targets(i,:,:,1) = sind(rake)*DS_e+cosd(rake)*SS_e;
  targets(i,:,:,2) = sind(rake)*DS_n+cosd(rake)*SS_n;
  targets(i,:,:,3) = sind(rake)*DS_u+cosd(rake)*SS_u;

  x_cen_inputs(i) = x_cen;
  y_cen_inputs(i) = y_cen;
  z_cen_inputs(i) = z_cen;
  fault_length_inputs(i) = fault_length;
  fault_width_inputs(i) = fault_width;
  dip_inputs(i) = dip;
  strike_inputs(i) = strike;
  rake_inputs(i) = rake;

  to_disp = [num2str(i),', ',num2str(time-t1)];
  disp(to_disp)


end


### If generation of samples is interrupted and we want to save what we have...
if i < N_times
  keep = 1:(i-1);
  x_cen_inputs = x_cen_inputs(keep);
  y_cen_inputs = y_cen_inputs(keep);
  z_cen_inputs = z_cen_inputs(keep);
  fault_length_inputs = fault_length_inputs(keep);
  fault_width_inputs = fault_width_inputs(keep);
  dip_inputs = dip_inputs(keep);
  strike_inputs = strike_inputs(keep);
  rake_inputs = rake_inputs(keep);
  targets = targets(keep,:,:,:);
end


### Cleaning up
clearvars -except x_cen_inputs y_cen_inputs z_cen_inputs ...
          fault_length_inputs fault_width_inputs dip_inputs ...
          strike_inputs rake_inputs targets

### Saving the samples in a .mat file so that they can later be used for
### surrogate model training.
save_name = ['random_faults_',num2str(int32(10_000*rand(1)*5)),'.mat'];
save(save_name,"-6")
















