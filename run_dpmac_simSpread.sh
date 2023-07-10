# ---------------------------------------------------------------------------------
# w/o dp

source ./scripts/train_mpe_nohup.sh 0 rdpmaddpg simple_spread 0 0 0 mark 4 0

source ./scripts/train_mpe_nohup.sh 1 rdpmaddpg simple_spread 0 0 0 mark 4 0

source ./scripts/train_mpe_nohup.sh 2 rdpmaddpg simple_spread 0 0 0 mark 4 0

source ./scripts/train_mpe_nohup.sh 3 rdpmaddpg simple_spread 0 0 0 mark 4 0

source ./scripts/train_mpe_nohup.sh 4 rdpmaddpg simple_spread 0 0 0 mark 4 0

# ---------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------
# w/ 1.0 dp

source ./scripts/train_mpe_nohup.sh 0 rdpmaddpg simple_spread 1.0 1 1 mark 5 0

source ./scripts/train_mpe_nohup.sh 1 rdpmaddpg simple_spread 1.0 1 1 mark 5 0

source ./scripts/train_mpe_nohup.sh 2 rdpmaddpg simple_spread 1.0 1 1 mark 5 0

source ./scripts/train_mpe_nohup.sh 3 rdpmaddpg simple_spread 1.0 1 1 mark 5 0

source ./scripts/train_mpe_nohup.sh 4 rdpmaddpg simple_spread 1.0 1 1 mark 5 0

# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# w/ 0.1 dp

source ./scripts/train_mpe_nohup.sh 0 rdpmaddpg simple_spread 0.1 1 1 mark 6 0

source ./scripts/train_mpe_nohup.sh 1 rdpmaddpg simple_spread 0.1 1 1 mark 6 0

source ./scripts/train_mpe_nohup.sh 2 rdpmaddpg simple_spread 0.1 1 1 mark 6 0

source ./scripts/train_mpe_nohup.sh 3 rdpmaddpg simple_spread 0.1 1 1 mark 6 0

source ./scripts/train_mpe_nohup.sh 4 rdpmaddpg simple_spread 0.1 1 1 mark 6 0

# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# w/ 0.01 dp

source ./scripts/train_mpe_nohup.sh 0 rdpmaddpg simple_spread 0.01 1 1 mark 7 0

source ./scripts/train_mpe_nohup.sh 1 rdpmaddpg simple_spread 0.01 1 1 mark 7 0

source ./scripts/train_mpe_nohup.sh 2 rdpmaddpg simple_spread 0.01 1 1 mark 7 0

source ./scripts/train_mpe_nohup.sh 3 rdpmaddpg simple_spread 0.01 1 1 mark 7 0

source ./scripts/train_mpe_nohup.sh 4 rdpmaddpg simple_spread 0.01 1 1 mark 7 0

# ---------------------------------------------------------------------------------