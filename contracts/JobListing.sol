pragma solidity ^0.8.0;

import "./Types.sol";
import "./OpenFLManager.sol";

// Minimal interface to read TaskRep round results from the challenge contract.
// Mirrors OpenFLChallenge.TaskRep + getTaskRepDeltaAndGRS().
interface IOpenFLChallengeTaskRep {
    struct TaskRep {
        address user;
        int256 delta;
        uint globalReputationScore;
    }

    function getTaskRepDeltaAndGRS() external view returns (TaskRep[] memory);

    function taskType() external view returns (TaskType);
}

contract JobListing {
    modifier onlyNotYetRegisteredUsers() {
        require(applicants[msg.sender].addr == address(0), "SAR");
        _;
    }

    modifier applicationWindowClosed() {
        require(block.timestamp >= applicationWindowCloseTime, "AWO");
        _;
    }

    // Authorized callers for updateUserTaskReps:
    //   - the registered challenge contract (normal runs, called from FLChallenge)
    //   - the publisher EOA who deployed this JobListing (replay runs, called from Python)
    modifier onlyTaskRepUpdater() {
        require(
            msg.sender == challengeAddress || msg.sender == publisher,
            "JL: not authorized for task rep update"
        );
        _;
    }

    event SelectionComplete(address[] participants);
    event ChallengeRegistered(address challengeAddress, bool success);
    event TaskRepsApplied(
        address indexed challengeAddress,
        TaskType indexed taskType,
        uint participantCount
    );

    struct User {
        uint globalTaskRep; // 32
        uint globalIntegrity; // 32
        uint nrOfTasksParticipated; // 1
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

    // True once updateUserTaskReps has run for this JobListing's challenge.
    // Idempotency guard: TaskRep deltas must apply at most once per challenge.
    bool public taskRepsApplied;

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
        TaskType _taskType
    ) payable {
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

    function getSelectedParticipants() public returns (address[] memory) {
        return selectedParticipants;
    }

    function register() public payable onlyNotYetRegisteredUsers {
        require(
            msg.value >= trainingSpecs.min_collateral &&
                msg.value <= trainingSpecs.max_collateral,
            "NWR"
        );
        registrationProcess(msg.sender);
    }

    function registrationProcess(address userAddr) internal {
        User storage user = applicants[userAddr];

        // TaskRep is per-task (TaskType acts as dataset key) — getUserRep
        // returns the user's TaskRep specifically for this job's task.
        (
            uint taskRep,
            uint globalIntegrity,
            uint nrOfTasksParticipated
        ) = manager.getUserRep(userAddr, trainingSpecs.taskType);

        user.globalTaskRep = taskRep;
        user.globalIntegrity = globalIntegrity;
        user.nrOfTasksParticipated = nrOfTasksParticipated;
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

        emit SelectionComplete(selected);
    }


    function getTopN(uint N) public view returns (address[] memory) {
        address[] memory heapUsers = new address[](N);
        uint[] memory heapScores = new uint[](N);

        uint size = 0;

        for (uint i = 0; i < applicantAddresses.length; i++) {
            uint score = applicants[applicantAddresses[i]].globalTaskRep;

            if (size < N) {
                heapUsers[size] = applicantAddresses[i];
                heapScores[size] = score;

                // heapify up
                uint idx = size;
                while (idx > 0) {
                    uint parent = (idx - 1) / 2;
                    if (heapScores[parent] <= heapScores[idx]) break;

                    (heapScores[parent], heapScores[idx]) = (
                        heapScores[idx],
                        heapScores[parent]
                    );
                    (heapUsers[parent], heapUsers[idx]) = (
                        heapUsers[idx],
                        heapUsers[parent]
                    );

                    idx = parent;
                }

                size++;
            } else if (score > heapScores[0]) {
                heapUsers[0] = applicantAddresses[i];
                heapScores[0] = score;

                // heapify down
                uint idx = 0;
                while (true) {
                    uint left = 2 * idx + 1;
                    uint right = 2 * idx + 2;
                    uint smallest = idx;

                    if (left < N && heapScores[left] < heapScores[smallest])
                        smallest = left;

                    if (right < N && heapScores[right] < heapScores[smallest])
                        smallest = right;

                    if (smallest == idx) break;

                    (heapScores[idx], heapScores[smallest]) = (
                        heapScores[smallest],
                        heapScores[idx]
                    );
                    (heapUsers[idx], heapUsers[smallest]) = (
                        heapUsers[smallest],
                        heapUsers[idx]
                    );

                    idx = smallest;
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

    // Pass-through view helper: forwards the round's TaskRep deltas + GRS from
    // the registered challenge contract. Useful for Python/off-chain code that
    // wants to inspect what updateUserTaskReps would consume without making a
    // separate call into the challenge.
    function getChallengeTaskReps()
        external
        view
        returns (IOpenFLChallengeTaskRep.TaskRep[] memory)
    {
        require(challengeAddress != address(0), "JL: challenge not registered");
        return IOpenFLChallengeTaskRep(challengeAddress).getTaskRepDeltaAndGRS();
    }

    // Calculate and apply updated per-task (= per-dataset) TaskRep for every
    // participant of the registered challenge. Called once per JobListing
    // lifecycle.
    //
    // Inputs available to the formula (per participant `rep`):
    //   - rep.user                    : participant address
    //   - rep.delta                   : signed taskRepDelta produced by this challenge round
    //   - rep.globalReputationScore   : participant's current GRS (collateral, not integrity rep)
    //   - tt                          : TaskType (= dataset) bound to this job
    //   - priorTaskRep                : participant's existing TaskRep for tt
    //                                   (from manager.getUserRep)
    //   - nrOfTasksParticipated       : from manager.getUserRep
    //
    // The formula must produce a *signed delta* (`int256 newDelta`) to apply on
    // top of the existing TaskRep. Negative deltas are saturated at zero by
    // OpenFLManager.applyUserTaskRepDelta.
    //
    // Auth: callable by the registered challenge contract (normal flow) or by
    // the JobListing publisher EOA (replay flow). Idempotent — taskRepsApplied
    // flag prevents double application.
    function updateUserTaskReps() external onlyTaskRepUpdater {
        require(!taskRepsApplied, "JL: task reps already applied");
        require(challengeAddress != address(0), "JL: challenge not registered");

        IOpenFLChallengeTaskRep challenge = IOpenFLChallengeTaskRep(
            challengeAddress
        );
        IOpenFLChallengeTaskRep.TaskRep[] memory reps = challenge
            .getTaskRepDeltaAndGRS();

        TaskType tt = trainingSpecs.taskType;

        for (uint i = 0; i < reps.length; i++) {
            IOpenFLChallengeTaskRep.TaskRep memory rep = reps[i];

            // ============================================================
            // TASK REP FORMULA — fill in below.
            //
            // Available locals: rep.user, rep.delta, rep.globalReputationScore,
            //                   tt, priorTaskRep, nrOfTasksParticipated
            //
            // Compute `int256 newDelta` representing the signed change to apply
            // to manager's GlobalTaskRep[user][tt].
            // ============================================================
            (
                uint priorTaskRep,
                ,
                uint nrOfTasksParticipated
            ) = manager.getUserRep(rep.user, tt);

            // PLACEHOLDER — replace with real formula.
            int256 newDelta = rep.delta;

            // Suppress unused-variable warnings until formula uses them.
            priorTaskRep;
            nrOfTasksParticipated;
            // ============================================================

            manager.applyUserTaskRepDelta(rep.user, tt, newDelta);
        }

        taskRepsApplied = true;
        emit TaskRepsApplied(challengeAddress, tt, reps.length);
    }
}
