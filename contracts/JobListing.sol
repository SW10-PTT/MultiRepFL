pragma solidity ^0.8.0;

import "./Types.sol";
import "./OpenFLManager.sol";

contract JobListing {
    uint256 private constant MAX_BIDS = 10;

    uint256 internal constant WAD = 1e18;

    modifier onlyNotYetRegisteredUsers() {
        require(applicants[msg.sender].addr == address(0), "SAR");
        _;
    }

    modifier applicationWindowClosed() {
        require(block.timestamp >= applicationWindowCloseTime, "AWO");
        _;
    }

    event SelectionComplete(address[] participants);
    event ChallengeRegistered(address challengeAddress, bool success);

    struct User {
        uint globalTaskRep; // 32
        uint globalIntegrity; // 32
        uint qValue; // 32  — Q-value from manager at registration time
        bytes32 tiebreaker; // 32  — deterministic tie-breaker (hash of off-chain fingerprint)
        address addr; // 20
        bool isSelected; // 1
    }
    mapping(address => User) public applicants;

    uint public applicationWindowCloseTime;
    OpenFLManager manager;
    address[] applicantAddresses;
    uint16 nrOfApplicants;
    address managerAddress;
    bytes32 challengeCodeHash;
    address public challengeAddress;
    address public publisher;

    TrainingSpecifications trainingSpecs;
    address[] selectedParticipants;

    constructor(
        uint _min_collateral,
        uint _max_collateral,
        uint _reward,
        uint8 _min_rounds,
        uint8 _punishfactor,
        uint8 _punishfactorContrib,
        uint8 _freeriderPenalty,
        address _managerAddress,
        TaskType _taskType,
        uint256 _qWeight,
        uint256 _trWeight,
        uint256 _girWeight
    ) payable {
        require(
            _trWeight + _girWeight > 0,
            "JL: trWeight + girWeight must be > 0"
        );
        managerAddress = _managerAddress;
        manager = OpenFLManager(_managerAddress);
        publisher = msg.sender;
        applicationWindowCloseTime = block.timestamp + 0 seconds;

        trainingSpecs.freeriderPenalty = _freeriderPenalty;
        trainingSpecs.managerAddress = _managerAddress;
        trainingSpecs.max_collateral = _max_collateral;
        trainingSpecs.min_collateral = _min_collateral;
        trainingSpecs.min_rounds = _min_rounds;
        trainingSpecs.punishfactor = _punishfactor;
        trainingSpecs.punishfactorContrib = _punishfactorContrib;
        trainingSpecs.reward = _reward;
        trainingSpecs.taskType = _taskType;
        trainingSpecs.qWeight = _qWeight;
        trainingSpecs.trWeight = _trWeight;
        trainingSpecs.girWeight = _girWeight;

        challengeCodeHash = manager.getChallengeCodeHash();
    }
    function debugTimes() external view returns (uint nowTs, uint closeTs) {
        return (block.timestamp, applicationWindowCloseTime);
    }

    function configHash() public view returns (bytes32) {
        return
            keccak256(
                abi.encode(
                    managerAddress,
                    applicationWindowCloseTime,
                    trainingSpecs.freeriderPenalty,
                    trainingSpecs.managerAddress,
                    trainingSpecs.max_collateral,
                    trainingSpecs.min_collateral,
                    trainingSpecs.min_rounds,
                    //trainingSpecs.modelHash,
                    trainingSpecs.punishfactor,
                    trainingSpecs.punishfactorContrib,
                    trainingSpecs.reward,
                    trainingSpecs.taskType,
                    selectedParticipants
                )
            );
    }

    function getSelectedParticipants() public view returns (address[] memory) {
        return selectedParticipants;
    }

    function register(
        bytes32 _tiebreaker
    ) public payable onlyNotYetRegisteredUsers {
        require(
            msg.value >= trainingSpecs.min_collateral &&
                msg.value <= trainingSpecs.max_collateral,
            "NWR"
        );
        registrationProcess(msg.sender, _tiebreaker);
    }

    function registrationProcess(
        address userAddr,
        bytes32 _tiebreaker
    ) internal {
        User storage user = applicants[userAddr];

        // TaskRep is per-task (TaskType acts as dataset key) — getUserRep
        // returns the user's TaskRep specifically for this job's task.
        (uint taskRep, uint globalIntegrity, uint qValue) = manager.getUserRep(
            userAddr,
            trainingSpecs.taskType
        );

        user.globalTaskRep = taskRep;
        user.globalIntegrity = globalIntegrity;
        user.qValue = qValue;
        user.tiebreaker = _tiebreaker;
        user.addr = userAddr;
        user.isSelected = false;

        nrOfApplicants += 1;
        applicantAddresses.push(userAddr);
    }

    function registerChallenge(address challengeContractAddr) public {
        require(challengeAddress == address(0), "IAE");

        bytes32 codeHash;

        assembly {
            codeHash := extcodehash(challengeContractAddr)
        }
        if (codeHash == challengeCodeHash) {
            challengeAddress = challengeContractAddr;
            emit ChallengeRegistered(challengeAddress, true);
        }
        emit ChallengeRegistered(challengeAddress, false);
    }

    function decideOnParticpants(
        uint8 amount
    ) public payable applicationWindowClosed {
        address[] memory selected = getTopN(amount);

        for (uint i = 0; i < selected.length; i++) {
            applicants[selected[i]].isSelected = true;
        }

        selectedParticipants = selected;

        // Update Q-values: patience bonus for non-selected, reset for selected.
        manager.updateQValuesAfterSelection(
            applicantAddresses,
            selected,
            trainingSpecs.taskType
        );

        emit SelectionComplete(selected);
    }

    function _selectionScore(User storage u) internal view returns (uint) {
        // score = (taskRep * trWeight + gir * girWeight) / (trWeight + girWeight) + qWeight * q / WAD
        uint denom = trainingSpecs.trWeight + trainingSpecs.girWeight;
        uint normalWeight = (u.globalTaskRep *
            trainingSpecs.trWeight +
            u.globalIntegrity *
            trainingSpecs.girWeight) / denom;
        uint qBonus = (trainingSpecs.qWeight * u.qValue) / WAD;
        return normalWeight + qBonus;
    }

    // Returns true when candidate A is strictly weaker than B and should be
    // evicted first. Weaker = lower score, or same score with higher tiebreaker.
    function _isWeaker(
        uint sA,
        bytes32 tbA,
        uint sB,
        bytes32 tbB
    ) internal pure returns (bool) {
        if (sA != sB) return sA < sB;
        return tbA > tbB;
    }

    function getTopN(uint N) public view returns (address[] memory) {
        address[] memory heapUsers = new address[](N);
        uint[] memory heapScores = new uint[](N);
        bytes32[] memory heapTBs = new bytes32[](N);

        uint size = 0;

        for (uint i = 0; i < applicantAddresses.length; i++) {
            address addr = applicantAddresses[i];
            uint score = _selectionScore(applicants[addr]);
            bytes32 tb = applicants[addr].tiebreaker;

            if (size < N) {
                heapUsers[size] = addr;
                heapScores[size] = score;
                heapTBs[size] = tb;

                // heapify up — bubble while child is strictly weaker than parent
                // (min-heap invariant: parent ≤ child, i.e. parent is weaker or equal)
                uint idx = size;
                while (idx > 0) {
                    uint parent = (idx - 1) / 2;
                    // stop when parent IS weaker than child (heap property OK for min-heap)
                    if (
                        _isWeaker(
                            heapScores[parent],
                            heapTBs[parent],
                            heapScores[idx],
                            heapTBs[idx]
                        )
                    ) break;

                    (heapScores[parent], heapScores[idx]) = (
                        heapScores[idx],
                        heapScores[parent]
                    );
                    (heapTBs[parent], heapTBs[idx]) = (
                        heapTBs[idx],
                        heapTBs[parent]
                    );
                    (heapUsers[parent], heapUsers[idx]) = (
                        heapUsers[idx],
                        heapUsers[parent]
                    );

                    idx = parent;
                }

                size++;
            } else if (!_isWeaker(score, tb, heapScores[0], heapTBs[0])) {
                // new candidate is not weaker than heap minimum → evict minimum
                heapUsers[0] = addr;
                heapScores[0] = score;
                heapTBs[0] = tb;

                // heapify down — sink the new root to its correct position
                uint idx = 0;
                while (true) {
                    uint left = 2 * idx + 1;
                    uint right = 2 * idx + 2;
                    uint weakest = idx;

                    if (
                        left < N &&
                        _isWeaker(
                            heapScores[left],
                            heapTBs[left],
                            heapScores[weakest],
                            heapTBs[weakest]
                        )
                    ) weakest = left;
                    if (
                        right < N &&
                        _isWeaker(
                            heapScores[right],
                            heapTBs[right],
                            heapScores[weakest],
                            heapTBs[weakest]
                        )
                    ) weakest = right;

                    if (weakest == idx) break;

                    (heapScores[idx], heapScores[weakest]) = (
                        heapScores[weakest],
                        heapScores[idx]
                    );
                    (heapTBs[idx], heapTBs[weakest]) = (
                        heapTBs[weakest],
                        heapTBs[idx]
                    );
                    (heapUsers[idx], heapUsers[weakest]) = (
                        heapUsers[weakest],
                        heapUsers[idx]
                    );

                    idx = weakest;
                }
            }
        }

        return heapUsers;
    }

    // Returns the TaskType (= dataset) bound to this JobListing — convenience
    // getter for off-chain callers and verification.
    function getTaskType() external view returns (TaskType) {
        return trainingSpecs.taskType;
    }

}
