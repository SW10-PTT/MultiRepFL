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

    function nrOfActiveParticipants() external view returns (uint);
}

contract JobListing {
    uint256 private constant MAX_BIDS = 10;

    // ---- TaskRepCalc fixed-point constants (WAD = 1e18) ----
    // Values map 1:1 to the ContribScoreCalc.xlsx workbook:
    //   ALPHA   = M3 = 0.2  (EWMA forgetting factor for mean + variance)
    //   N_0     = M6 = 5    (maturity offset)
    //   LAMBDA  = M9 = 20   (variance penalty weight; dimensionless on s_k)
    //   N_BLEND = M12 = 0.2 (smoothing on the final ContribScore)
    // STAKE_WAD is hardcoded to 1 ETH for now — should later read the actual
    // collateral the participant locked when registering.
    uint256 internal constant WAD = 1e18;
    uint256 internal constant ALPHA = 2e17;
    uint256 internal constant N_BLEND = 2e17;
    uint256 internal constant N_0 = 5;
    uint256 internal constant LAMBDA = 20;
    uint256 internal constant STAKE_WAD = 1e18;
    // Upper bound on per-task positive delta = GAIN_CAP_MULTIPLIER * (reward /
    // nrActive). 1x flattens top performers (their J caps at the equal-share
    // baseline). 2x gives outperformers headroom while keeping the average
    // participant near the middle of [0, WAD]. Tune as needed.
    uint256 internal constant GAIN_CAP_MULTIPLIER = 2;

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

    // Calculate and apply the updated per-task (= per-dataset) TaskRep for
    // every participant of the registered challenge. TaskRep is updated once
    // per task on completion (not per round), so this runs at end-of-task.
    //
    // The formula computes the *new absolute TaskRep* for each participant
    // (typically a weighted blend of their prior TaskRep and the rep earned
    // for this task). The new value overwrites the previous TaskRep stored on
    // OpenFLManager via setUserTaskRep.
    //
    // Inputs available to the formula (per participant `rep`):
    //   - rep.user                    : participant address
    //   - rep.delta                   : signed taskRepDelta produced by this task
    //                                   (treat as "rep earned for this task")
    //   - rep.globalReputationScore   : participant's current GRS (collateral, not integrity rep)
    //   - tt                          : TaskType (= dataset) bound to this job
    //   - priorTaskRep                : participant's existing TaskRep for tt
    //                                   (from manager.getUserRep)
    //   - nrOfTasksParticipated       : from manager.getUserRep
    //
    // Output: `uint256 newTaskRep` — the absolute replacement value.
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
        uint256 reward = trainingSpecs.reward;
        // Use the challenge's active (non-disqualified) count so the
        // per-participant reward cap reflects the true distribution pool,
        // not the inflated raw participant count.
        uint256 nrActive = challenge.nrOfActiveParticipants();

        for (uint i = 0; i < reps.length; i++) {
            _applyTaskRepCalc(reps[i], tt, reward, nrActive);
        }

        taskRepsApplied = true;
        emit TaskRepsApplied(challengeAddress, tt, reps.length);
    }

    // Per-participant body of updateUserTaskReps, extracted so the loop's
    // stack stays shallow enough for the compiler.
    function _applyTaskRepCalc(
        IOpenFLChallengeTaskRep.TaskRep memory rep,
        TaskType tt,
        uint256 reward,
        uint256 nrActive
    ) internal {
        (uint256 priorK, , uint256 priorTaskCount) = manager.getUserRep(
            rep.user,
            tt
        );
        (uint256 priorE, uint256 priorF) = manager.getTaskRepCalcState(
            rep.user,
            tt
        );

        // k is the current task index (1-based). Matches spreadsheet row-3 = k=1.
        uint256 k = priorTaskCount + 1;
        uint256 J = _transformDelta(rep.delta, STAKE_WAD, reward, nrActive);

        (uint256 newE, uint256 newF) = _updateRunningStats(
            J,
            priorE,
            priorF,
            k
        );
        uint256 newK = _updateContribScore(
            priorK,
            _computeConfidence(k, newF),
            J
        );

        manager.setTaskRepCalcState(rep.user, tt, newE, newF);
        manager.setUserTaskRep(rep.user, tt, newK);
        manager.incrementNumberOfTasksJoined(rep.user);
    }

    // Linearly maps a signed per-task reputation delta (wei) into a raw
    // contribution score J_k in [0, WAD]. Worst case (-stake) maps to 0,
    // ceiling is GAIN_CAP_MULTIPLIER * (reward / nrActive) — anything above
    // is clipped to WAD. nrActive is the non-disqualified count, so kicked
    // participants do not deflate the per-participant reward share.
    function _transformDelta(
        int256 delta,
        uint256 stake,
        uint256 reward,
        uint256 nrActive
    ) internal pure returns (uint256) {
        uint256 maxGain = nrActive == 0
            ? 0
            : (GAIN_CAP_MULTIPLIER * reward) / nrActive;
        uint256 range = stake + maxGain;
        if (range == 0) return 0;

        int256 shifted = delta + int256(stake);
        if (shifted <= 0) return 0;

        uint256 num = uint256(shifted);
        if (num >= range) return WAD;
        return (num * WAD) / range;
    }

    // EWMA running mean (E_k = RunningCMean) and variance proxy
    // (F_k = M2). On the first task (k == 1) E is seeded directly from J
    // (no smoothing) — matches the workbook's row-3 seed semantics; F stays 0
    // because Delta2 = J - E_k = 0.
    //
    // Welford identity: D = (1-ALPHA)*C, so C*D >= 0 always — safe to compute
    // with unsigned absolutes.
    function _updateRunningStats(
        uint256 J,
        uint256 priorE,
        uint256 priorF,
        uint256 k
    ) internal pure returns (uint256 newE, uint256 newF) {
        if (k <= 1) {
            newE = J;
        } else {
            newE = ((WAD - ALPHA) * priorE + ALPHA * J) / WAD;
        }

        uint256 absDelta = J > priorE ? J - priorE : priorE - J;
        uint256 absDelta2 = J > newE ? J - newE : newE - J;

        newF =
            ((WAD - ALPHA) * priorF) /
            WAD +
            (ALPHA * absDelta * absDelta2) /
            (WAD * WAD);
    }

    // Confidence weight H_k in [0, WAD]. Combines maturity k/(k+N_0) with
    // stability 1/(1 + LAMBDA * s_k). Marked `virtual` so a subclass can
    // swap in a different confidence formula without touching the rest of
    // the update logic.
    function _computeConfidence(
        uint256 k,
        uint256 s_k
    ) internal pure virtual returns (uint256) {
        if (k == 0) return 0;
        uint256 maturity = (k * WAD) / (k + N_0);
        uint256 stability = (WAD * WAD) / (WAD + LAMBDA * s_k);
        return (maturity * stability) / WAD;
    }

    // EWMA-smoothed ContribScore (TaskRepCalc = K_k).
    function _updateContribScore(
        uint256 priorK,
        uint256 H,
        uint256 J
    ) internal pure returns (uint256) {
        uint256 weighted = (H * J) / WAD;
        return ((WAD - N_BLEND) * priorK + N_BLEND * weighted) / WAD;
    }
}
