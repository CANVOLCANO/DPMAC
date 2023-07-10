# ---------------------------------------------------------------------------------
# w/o dp

source ./scripts/train_mpe_nohup.sh 0 rdpmaddpg po_reference 0 0 0 mark 0 0

source ./scripts/train_mpe_nohup.sh 1 rdpmaddpg po_reference 0 0 0 mark 0 0

source ./scripts/train_mpe_nohup.sh 2 rdpmaddpg po_reference 0 0 0 mark 0 0

source ./scripts/train_mpe_nohup.sh 3 rdpmaddpg po_reference 0 0 0 mark 0 0

source ./scripts/train_mpe_nohup.sh 4 rdpmaddpg po_reference 0 0 0 mark 0 0

# ---------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------
# w/ 1.0 dp

source ./scripts/train_mpe_nohup.sh 0 rdpmaddpg po_reference 1.0 1 1 mark 1 0

source ./scripts/train_mpe_nohup.sh 1 rdpmaddpg po_reference 1.0 1 1 mark 1 0

source ./scripts/train_mpe_nohup.sh 2 rdpmaddpg po_reference 1.0 1 1 mark 1 0

source ./scripts/train_mpe_nohup.sh 3 rdpmaddpg po_reference 1.0 1 1 mark 1 0

source ./scripts/train_mpe_nohup.sh 4 rdpmaddpg po_reference 1.0 1 1 mark 1 0

# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# w/ 0.1 dp

source ./scripts/train_mpe_nohup.sh 0 rdpmaddpg po_reference 0.1 1 1 mark 2 0

source ./scripts/train_mpe_nohup.sh 1 rdpmaddpg po_reference 0.1 1 1 mark 2 0

source ./scripts/train_mpe_nohup.sh 2 rdpmaddpg po_reference 0.1 1 1 mark 2 0

source ./scripts/train_mpe_nohup.sh 3 rdpmaddpg po_reference 0.1 1 1 mark 2 0

source ./scripts/train_mpe_nohup.sh 4 rdpmaddpg po_reference 0.1 1 1 mark 2 0

# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# w/ 0.01 dp

source ./scripts/train_mpe_nohup.sh 0 rdpmaddpg po_reference 0.01 1 1 mark 3 0

source ./scripts/train_mpe_nohup.sh 1 rdpmaddpg po_reference 0.01 1 1 mark 3 0

source ./scripts/train_mpe_nohup.sh 2 rdpmaddpg po_reference 0.01 1 1 mark 3 0

source ./scripts/train_mpe_nohup.sh 3 rdpmaddpg po_reference 0.01 1 1 mark 3 0

source ./scripts/train_mpe_nohup.sh 4 rdpmaddpg po_reference 0.01 1 1 mark 3 0

# ---------------------------------------------------------------------------------