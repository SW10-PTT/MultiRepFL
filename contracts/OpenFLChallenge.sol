// SPDX-License-Identifier: Apache-2.0
//  ___ _   _ ____       ____  _____ _
// |_ _| | | |  _ \     |  _ \|  ___| |
//  | || |_| | |_) |____| | | | |_  | |
//  | ||  _  |  __/_____| |_| |  _| | |___
// |___|_| |_|_|        |____/|_|   |_____|
// OpenFL is a Ethereum-based reputation system to facilitate federated learning.
// This contract is part of the OpenFL research paper by Anton Wahrstätter. The contracts do only
// represent Proof-of-Concepts and have not been developed to be used in productive
// environments. Do not use them, except for testing purpose.

pragma solidity ^0.8.0;

import "./Types.sol";

interface IJobListing {
    function getSelectedParticipants() external view returns (address[] memory);
}

interface IOpenFLManager {
    function updateReputationsFromChallenge(
        address challengeAddr,
        TaskType taskType
    ) external;
}

contract OpenFLChallenge {
    bytes32 public modelHash;

    uint8 public round = 0;
    uint8 public votesPerRound;
    uint8 public punishfactor;
    uint8 public min_rounds;
    uint8 public punishfactorContrib;

    uint public nrOfActiveParticipants;
    uint public nrOfProvidedHashedWeights;
    uint public initTS;
    uint public min_collateral;
    uint public max_collateral;
    uint public totalReward;
    uint public rewardPerRound;
    uint public rewardLeft;
    uint public roundStart;
    uint public contributionStart;
    uint public freeriderPenalty;
    TaskType public taskType;
    address public managerAddress;
    uint constant ONE_DAY = 864e2;

    address[] public participants;
    address[] punishedAddresses;

    bool public testing = false;

    // Dont change order, fl_challenge.py relies on order. Maybe use getters if bytecode size allows later
    struct User {
        int256 weightedContribScore; // 32
        uint globalReputationScore; // 32
        int taskRepDelta; // 32
        int256 roundReputation; // 32
        address addr; // 20
        uint8 nrOfRoundsParticipated; // 1
        uint8 nrOfVotesFromUser; // 1
        bool isPunished; // 1
        bool isRegistered; // 1
        bool isSelected; // 1 //OOOPS CHANGED ORDER
        bool whitelistedForRewards; // 1
        bool isDisqualified; // 1
    }

    mapping(address => User) public users;

    mapping(address => mapping(address => bool)) public hasVoted;
    mapping(address => mapping(address => bool)) public votedPositiveFor;
    mapping(address => mapping(uint8 => bytes32)) public secretOf;
    mapping(address => mapping(uint8 => bytes32)) public weightsOf;
    mapping(uint8 => mapping(address => int256)) public contributionScore; // round => user => score
    mapping(uint8 => uint256) public nrOfContributionScores; // round => number of submissions

    // -------- Cross-round vote tallies (used by GIR computation) --------
    // positiveVotesGivenByTo[voter][target] : count of +1 votes voter cast for
    //                                         target, summed across all rounds
    //                                         of this task.
    // totalVotesGivenByTo[voter][target]    : count of all votes (+1/0/-1).
    // positiveVotesReceived[target]         : Σ over voters of the above.
    // totalVotesReceived[target]            : Σ over voters of the above.
    //
    // Receiver tallies are the inputs to GIR's V = (positive/total)^2.
    // When a voter is disqualified in settle(), their per-(voter,target)
    // entries are zeroed and subtracted out of the receiver tallies, so
    // votes from kicked participants do not count.
    mapping(address => mapping(address => uint256))
        public positiveVotesGivenByTo;
    mapping(address => mapping(address => uint256)) public totalVotesGivenByTo;
    mapping(address => uint256) public positiveVotesReceived;
    mapping(address => uint256) public totalVotesReceived;

    struct AccuracyLossSubmission {
        address[] adrs;
        uint16[] acc;
        uint16[] loss;
    }

    struct AccuracySubmission {
        address[] adrs;
        uint16[] acc;
    }

    struct LossSubmission {
        address[] adrs;
        uint16[] loss;
    }

    mapping(uint8 => mapping(address => uint16)) public prev_accs;
    mapping(uint8 => mapping(address => uint16)) public prev_losses;

    // Mapping from sender to all their submissions
    mapping(uint16 => mapping(address => AccuracyLossSubmission[]))
        private accuracyLossSubmissions;

    mapping(uint16 => mapping(address => AccuracySubmission[]))
        private accuracySubmissions;

    mapping(uint16 => mapping(address => LossSubmission[]))
        private lossSubmissions;

    modifier onlyRegisteredUsers() {
        require(users[msg.sender].isRegistered, "SNR");
        _;
    }

    modifier feedbackRoundOpened() {
        require(
            nrOfProvidedHashedWeights == nrOfActiveParticipants ||
                roundStart + ONE_DAY < block.timestamp,
            "FRC"
        );
        _;
    }

    modifier feedbackRoundClosed() {
        require(
            nrOfProvidedHashedWeights != nrOfActiveParticipants &&
                roundStart + ONE_DAY > block.timestamp,
            "NA"
        );
        require(weightsOf[msg.sender][round] == bytes32(0), "WFE");
        _;
    }

    modifier onlyValidTargets(address target) {
        require(target != msg.sender, "SET");
        require(!hasVoted[msg.sender][target], "VAC");
        _;
    }

    modifier onlyNotYetRegisteredUsers(address userAddr) {
        require(!users[userAddr].isRegistered, "SAR");
        _;
    }

    modifier onlySelectedUsers(address userAddr) {
        require(users[userAddr].isSelected, "SUO");
        _;
    }

    modifier hasNotYetProvidedWeights() {
        require(weightsOf[msg.sender][round] == bytes32(0), "SAP");
        _;
    }

    event FederatedLearningModelDeployed(
        uint initTS,
        uint max_collateral,
        uint min_collateral,
        uint total_reward,
        uint8 min_rounds,
        uint freerider_fee,
        bool isTemplate
    );

    event Registered(
        address user,
        uint reputationValue,
        uint totalCollateral,
        uint numberOfContributers
    );

    event Feedback(
        address target,
        address user,
        uint globalReputation,
        int256 newRoundReputation
    );

    event ContributionScoreSubmitted(
        address indexed user,
        int256 contributionScore
    );

    event EndRound(
        uint8 round,
        uint8 validVotes,
        uint sumOfWeightedContribScore,
        uint totalPunishment
    );

    event Punishment(
        address victim,
        int256 roundScore,
        uint loss,
        uint newReputation
    );

    event PassivPunishment(
        address victim,
        int256 roundScore,
        uint loss,
        uint newReputation
    );

    event Disqualification(
        address victim,
        int256 roundScore,
        uint loss,
        uint newReputation
    );

    event Reward(
        address user,
        int256 roundScore,
        uint win,
        uint newReputation,
        bool is_reward
    );

    event SelectedUsers(address[] users);

    constructor(ChallengeSpecifications memory taskSpecs) payable {
        // Initialize Contract
        initTS = block.timestamp;
        roundStart = block.timestamp;
        modelHash = taskSpecs.modelHash;
        min_collateral = taskSpecs.min_collateral;
        max_collateral = taskSpecs.max_collateral;
        totalReward = taskSpecs.reward;
        min_rounds = taskSpecs.min_rounds;
        punishfactor = taskSpecs.punishfactor;
        punishfactorContrib = taskSpecs.punishfactorContrib;
        taskType = taskSpecs.taskType;
        managerAddress = taskSpecs.managerAddress;
        freeriderPenalty = (min_collateral * taskSpecs.freeriderPenalty) / 100;
        rewardPerRound = totalReward / min_rounds;
        rewardLeft = totalReward;

        bool isTemplate = taskSpecs.taskType == TaskType.template;

        if (!isTemplate) {
            require(taskSpecs.jobListingAddress != address(0), "NO_JOBADDR");

            IJobListing job = IJobListing(taskSpecs.jobListingAddress);

            address[] memory selectedUsers = job.getSelectedParticipants();
            emit SelectedUsers(selectedUsers); //TODO: DEBUG
            for (uint i = 0; i < selectedUsers.length; i++) {
                users[selectedUsers[i]].isSelected = true;
            }
        }

        emit FederatedLearningModelDeployed(
            initTS,
            min_collateral,
            max_collateral,
            totalReward,
            min_rounds,
            freeriderPenalty,
            isTemplate
        );
    }

    function setTesting(bool _testing) external {
        testing = _testing;
    }

    // Registration
    function registrationProcess()
        public
        payable
        onlyNotYetRegisteredUsers(msg.sender)
        onlySelectedUsers(msg.sender)
    {
        require(
            msg.value >= min_collateral && msg.value <= max_collateral,
            "NWR"
        );

        User storage user = users[msg.sender];
        user.isRegistered = true;
        user.globalReputationScore = msg.value;
        user.nrOfRoundsParticipated = 1;
        user.addr = msg.sender;
        nrOfActiveParticipants += 1;

        participants.push(msg.sender);
        // emit Registered(
        //     user.addr,
        //     msg.value,
        //     address(this).balance,
        //     nrOfActiveParticipants
        // );
    }

    // Register Slot
    function registerSlot(
        bytes32 _secret
    ) public onlyRegisteredUsers hasNotYetProvidedWeights {
        secretOf[msg.sender][round] = _secret;
    }

    // Timestamp weights to the chain
    function provideHashedWeights(
        bytes32 hashedWeights,
        uint salt
    ) public onlyRegisteredUsers hasNotYetProvidedWeights {
        require(
            secretOf[msg.sender][round] ==
                keccak256(abi.encodePacked(hashedWeights, salt, msg.sender)),
            "NKS"
        );
        weightsOf[msg.sender][round] = hashedWeights;
        nrOfProvidedHashedWeights += 1;
    }

    function feedback(
        address target,
        int256 score
    )
        public
        virtual
        onlyRegisteredUsers
        onlyValidTargets(target)
        feedbackRoundOpened
    {
        //(address target, int score) = abi.decode(data, (address, int));
        hasVoted[msg.sender][target] = true;
        users[msg.sender].nrOfVotesFromUser += 1;
        votesPerRound += 1;

        // Cross-round GIR tallies. Score in {-1, 0, +1}: every call counts
        // toward totalVotes; only score==1 contributes to positiveVotes.
        totalVotesGivenByTo[msg.sender][target] += 1;
        totalVotesReceived[target] += 1;
        if (score == 1) {
            positiveVotesGivenByTo[msg.sender][target] += 1;
            positiveVotesReceived[target] += 1;
        }

        if (score == 1) {
            votedPositiveFor[msg.sender][target] = true;
            users[target].roundReputation +=
                1 *
                int(users[msg.sender].globalReputationScore);
        }
        if (score == -1) {
            votedPositiveFor[msg.sender][target] = false;
            users[target].roundReputation -=
                1 *
                int(users[msg.sender].globalReputationScore);
        }
        if (score == 0) {
            votedPositiveFor[msg.sender][target] = false;
        }
        emit Feedback(
            target,
            msg.sender,
            users[msg.sender].globalReputationScore,
            users[target].roundReputation
        );
    }

    function submitContributionScore(int256 score) external {
        require(users[msg.sender].isRegistered, "User not registered");
        require(
            contributionScore[round][msg.sender] == 0,
            "Score already submitted"
        );

        contributionScore[round][msg.sender] = score;
        nrOfContributionScores[round] += 1;

        emit ContributionScoreSubmitted(msg.sender, score);
    }

    function isFeedBackRoundDone() public view returns (bool roundClosed) {
        if (nrOfActiveParticipants == 0) {
            return false; // no participants => not done
        }

        for (uint i = 0; i < participants.length; i++) {
            User storage user = users[participants[i]];
            // If a particaipant hasnt voted for everyone else wait
            if (user.isRegistered && !user.isDisqualified) {
                if (user.nrOfVotesFromUser < nrOfActiveParticipants - 1) {
                    return false;
                }
            }
        }
        return true;
    }

    function isContributionRoundDone() public view returns (bool roundClosed) {
        // mergedUsers == users that contributed to the global model this
        // round. Python uses `roundReputation >= 0` to pick the merger set
        // (FLChallenge.py: contributors = [u for u in participants if
        // u._roundrep[-1] >= 0]) and submits one contribution score per
        // merger. The on-chain check must mirror that predicate, otherwise
        // the round never closes when |downvoted| != |merged| and settle()
        // gets force-triggered on partial state.
        uint mergedUsers = 0;
        for (uint i = 0; i < participants.length; i++) {
            User storage user = users[participants[i]];
            if (
                user.isRegistered &&
                !user.isDisqualified &&
                user.roundReputation >= 0
            ) {
                mergedUsers++;
            }
        }
        if (nrOfContributionScores[round] < mergedUsers) {
            return false;
        }

        return true;
    }

    function settle() public {
        uint totalPunishment;
        uint freeriderLock; // A global total of sum of freerider penalties

        // First round users pay their anti-freerider fee
        for (uint i = 0; i < participants.length; i++) {
            User storage user = users[participants[i]];
            if (user.nrOfRoundsParticipated == 1) {
                user.globalReputationScore -= freeriderPenalty;
                user.taskRepDelta -= int256(freeriderPenalty);
                freeriderLock += freeriderPenalty;
            }
        }

        // Punish malicious users
        for (uint i = 0; i < participants.length; i++) {
            User storage user = users[participants[i]];
            if (user.isRegistered && !user.isDisqualified) {
                if (user.roundReputation < 0) {
                    votesPerRound -= user.nrOfVotesFromUser;

                    uint punishment = uint(
                        user.globalReputationScore / punishfactor
                    );
                    int256 _rawTaskPunishment = (user.taskRepDelta +
                        int(min_collateral)) / int(uint(punishfactor));
                    uint taskPunishment = _rawTaskPunishment > 0
                        ? uint(_rawTaskPunishment)
                        : 0;

                    if (
                        user.globalReputationScore >
                        min_collateral / punishfactor
                    ) {
                        user.isPunished = true;
                        punishedAddresses.push(participants[i]);
                        user.whitelistedForRewards = false;

                        user.globalReputationScore =
                            user.globalReputationScore -
                            punishment;
                        user.taskRepDelta -= int256(taskPunishment);
                        user.roundReputation =
                            user.roundReputation -
                            int(punishment);
                        totalPunishment += punishment;
                        emit Punishment(
                            participants[i],
                            user.roundReputation,
                            punishment,
                            user.globalReputationScore
                        );
                    } else {
                        user.isRegistered = false;
                        user.isPunished = true;
                        punishedAddresses.push(participants[i]);
                        user.whitelistedForRewards = false;

                        totalPunishment += user.globalReputationScore;
                        user.taskRepDelta = -int256(1e18);

                        emit Disqualification(
                            user.addr,
                            user.roundReputation,
                            user.globalReputationScore,
                            0
                        );
                        user.globalReputationScore = 0;
                        nrOfActiveParticipants -= 1;
                        user.isDisqualified = true;
                        _removeKickedUserVotesFromTallies(user.addr);
                    }
                } else {
                    user.whitelistedForRewards = true;
                }
            }
        }

        // Punish helpers of malicious users.
        //
        // Invariant: votesPerRound == sum(nrOfVotesFromUser) over all voters
        // (each feedback() call increments both by 1). The punish loop above
        // already removed every punished user's outgoing votes (L411), so
        // skipping isPunished users here keeps the subtractions disjoint.
        // Within the helper branch, we hoist the subtraction outside the
        // per-punished-address loop: a helper's votes leave the denominator
        // exactly once regardless of how many malicious users they backed.
        // Old code subtracted nrOfVotesFromUser once per offending vote,
        // which both over-counted the penalty and underflowed votesPerRound
        // whenever a helper voted positive for >= 2 punished users.
        for (uint i = 0; i < participants.length; i++) {
            User storage user = users[participants[i]];
            if (!user.isRegistered || user.isDisqualified || user.isPunished) {
                continue;
            }
            bool isHelper = false;
            for (uint j = 0; j < punishedAddresses.length; j++) {
                if (votedPositiveFor[participants[i]][punishedAddresses[j]]) {
                    votedPositiveFor[participants[i]][
                        punishedAddresses[j]
                    ] = false;
                    isHelper = true;
                    emit PassivPunishment(
                        participants[i],
                        user.roundReputation,
                        0,
                        user.globalReputationScore
                    );
                }
            }
            if (isHelper) {
                votesPerRound -= user.nrOfVotesFromUser;
                user.whitelistedForRewards = false;
            }
        }

        // Pay back freerider 1st round stake to good users
        for (uint i = 0; i < participants.length; i++) {
            User storage user = users[participants[i]];

            if (!user.isRegistered) continue;
            if (user.isDisqualified) continue;
            if (user.nrOfRoundsParticipated != 1) continue;

            if (user.whitelistedForRewards) {
                user.globalReputationScore += freeriderPenalty;
                user.taskRepDelta += int256(freeriderPenalty);
                freeriderLock -= freeriderPenalty;
                continue;
            }

            totalPunishment += freeriderPenalty;
            freeriderLock -= freeriderPenalty;
        }

        // Devide reward between every user who provided (non-malicious) feedback
        // Pay back freeriderLock funds to good users
        // First round users pay their anti-freerider fee
        int256 sumOfWeightedContribScore = 0;
        uint256 positiveSumOfWeightedContribScore;
        if (votesPerRound > 0 && rewardLeft >= rewardPerRound) {
            rewardLeft -= rewardPerRound;

            uint reward = rewardPerRound + totalPunishment;

            // Compute weights
            for (uint i = 0; i < participants.length; i++) {
                User storage user = users[participants[i]];

                if (
                    _isEligibleForRewards(user) &&
                    contributionScore[round][user.addr] > 0
                ) {
                    int256 weight = int256(uint(user.nrOfVotesFromUser)) *
                        contributionScore[round][user.addr];
                    user.weightedContribScore = weight;
                    sumOfWeightedContribScore += weight;
                }
            }
            require(
                sumOfWeightedContribScore > 0,
                "sumOfWeightedContribScore is <= 0 in settle!"
            );
            positiveSumOfWeightedContribScore = uint256(
                sumOfWeightedContribScore
            );

            // check if a user should be disqualified or punished
            for (uint i = 0; i < participants.length; i++) {
                User storage user = users[participants[i]];

                if (
                    _isEligibleForRewards(user) &&
                    contributionScore[round][user.addr] < 0
                ) {
                    require(
                        punishfactorContrib > 0,
                        "punishfactorcontrib <= 0"
                    );
                    require(
                        user.globalReputationScore > 0,
                        "user.globalreputation <= 0"
                    );
                    require(
                        contributionScore[round][user.addr] < 0,
                        "contrib >= 0"
                    );
                    uint punishment = (user.globalReputationScore /
                        punishfactorContrib) *
                        absUint((contributionScore[round][user.addr]));
                    int taskPunishment = (((user.taskRepDelta + int(min_collateral)) /
                        int(uint(punishfactorContrib))) *
                        int(absUint((contributionScore[round][user.addr])))) /
                        int(1e18);
                    require(punishment > 0, "punishment is <= 0 in settle! 1");
                    punishment /= 1e18;
                    require(punishment > 0, "punishment is <= 0 in settle! 2");
                    if (
                        user.globalReputationScore <=
                        min_collateral / punishfactorContrib ||
                        user.globalReputationScore <= punishment
                    ) {
                        reward += user.globalReputationScore;
                        user.taskRepDelta = -1e18;

                        emit Disqualification(
                            participants[i],
                            user.roundReputation,
                            user.globalReputationScore,
                            0
                        );

                        user.globalReputationScore = 0;
                        nrOfActiveParticipants -= 1;
                        user.isDisqualified = true;
                        _removeKickedUserVotesFromTallies(user.addr);
                    } else {
                        // this is a punishment
                        user.globalReputationScore -= punishment;
                        user.taskRepDelta -= int256(taskPunishment);
                        reward += punishment;

                        emit Reward(
                            user.addr,
                            user.roundReputation,
                            punishment,
                            user.globalReputationScore,
                            false
                        );

                        delete user.whitelistedForRewards;
                        delete user.weightedContribScore;
                    }
                }
            }

            // Give rewards based on positive contribution score
            for (uint i = 0; i < participants.length; i++) {
                User storage user = users[participants[i]];

                if (
                    _isEligibleForRewards(user) &&
                    contributionScore[round][user.addr] >= 0
                ) {
                    // NOTE: This refactor adds the case of !user.Disqualified, in contrast to before)
                    uint personalReward = (reward *
                        uint(user.weightedContribScore)) /
                        positiveSumOfWeightedContribScore;

                    user.globalReputationScore += personalReward;

                    uint personalRep = (rewardPerRound *
                        uint(user.weightedContribScore)) /
                        positiveSumOfWeightedContribScore;
                    user.taskRepDelta += int(personalRep);

                    emit Reward(
                        user.addr,
                        user.roundReputation,
                        personalReward,
                        user.globalReputationScore,
                        true
                    );
                }

                delete user.whitelistedForRewards;
                delete user.weightedContribScore;
            }
        }
        emit EndRound(
            round,
            votesPerRound,
            positiveSumOfWeightedContribScore,
            totalPunishment
        );

        // Reset variables
        for (uint i = 0; i < participants.length; i++) {
            User storage user = users[participants[i]];
            if (user.isRegistered && !user.isDisqualified) {
                user.nrOfVotesFromUser = 0;
                user.roundReputation = 0;
                user.nrOfRoundsParticipated += 1;
                user.isPunished = false;
                for (uint j = 0; j < participants.length; j++) {
                    delete hasVoted[user.addr][participants[j]];
                }
            }
        }

        round += 1;
        votesPerRound = 0;
        nrOfProvidedHashedWeights = 0;
        delete punishedAddresses;
    }

    // // Exit contract - Not safe, gaurds exists but will crash the contract if not met, exits should be queued?
    // function exitModel() public onlyRegisteredUsers feedbackRoundClosed {
    //     require(users[msg.sender].globalReputationScore > 0, "NEF");
    //     uint val = users[msg.sender].globalReputationScore;
    //     users[msg.sender].globalReputationScore = 0;
    //     for (uint i = 0; i < participants.length; i++) {
    //         if (participants[i] == msg.sender) {
    //             delete participants[i];
    //         }
    //     }
    //     users[msg.sender].isRegistered = false;
    //     payable(address(msg.sender)).transfer(val);
    // }

    function exitModel() public onlyRegisteredUsers feedbackRoundClosed {
        User storage user = users[msg.sender];

        uint val = user.globalReputationScore;
        require(val > 0, "NEF");

        // EFFECTS (state changes first)
        user.globalReputationScore = 0;
        user.isRegistered = false;

        for (uint i = 0; i < participants.length; i++) {
            if (participants[i] == msg.sender) {
                participants[i] = participants[participants.length - 1];
                participants.pop();
                break;
            }
        }

        // INTERACTION (external call last)
        (bool success, ) = msg.sender.call{value: val}("");
        require(success, "ETH transfer failed");
    }

    function submitFeedbackBytes(bytes calldata raw) external {
        address[] memory ads;
        int16[] memory ints;

        (ads, ints) = parseRaw(raw);

        // EXACT same for-loop as fallback
        for (uint i = 0; i < ads.length; i++) {
            if (!testing) {
                feedback(ads[i], ints[i]);
            }
        }
    }

    // function submitFeedbackBytesAndAccuraciesLosses(
    //     bytes calldata raw,
    //     uint16[] calldata accuracies,
    //     uint16[] calldata losses,
    //     uint16 prev_acc,
    //     uint16 prev_loss
    // ) external {
    //     address[] memory ads;
    //     int16[] memory ints;

    //     (ads, ints) = parseRaw(raw);

    //     require(
    //         accuracies.length == ads.length,
    //         "INVALID_LENGTH OF ACCURACY ARRAY"
    //     );
    //     require(losses.length == ads.length, "INVALID_LENGTH OF LOSS ARRAY");
    //     accuracyLossSubmissions[round][msg.sender].push(
    //         AccuracyLossSubmission({adrs: ads, acc: accuracies, loss: losses})
    //     );
    //     require(
    //         prev_acc >= 0 && prev_acc <= type(uint16).max,
    //         "PREVIOUS ACCURACY NOT BETWEEN 0 AND uint16max in submitFeedbackBytesAndAccuraciesLosses"
    //     );
    //     require(
    //         prev_loss >= 0 && prev_loss <= type(uint16).max,
    //         "PREVIOUS LOSS NOT BETWEEN 0 AND uint16max in submitFeedbackBytesAndAccuraciesLosses"
    //     );
    //     // EXACT same for-loop as fallback
    //     for (uint i = 0; i < ads.length; i++) {
    //         if (!testing) {
    //             feedback(ads[i], ints[i]);
    //         }
    //     }
    // }

    function submitFeedbackBytesAndAccuraciesLosses(
        bytes calldata raw,
        uint16[] calldata accuracies,
        uint16[] calldata losses,
        uint16 prev_acc,
        uint16 prev_loss
    ) external {
        (address[] memory ads, int16[] memory ints) = parseRaw(raw);

        _validateAccuracyLossInputs(
            ads.length,
            accuracies.length,
            losses.length,
            prev_acc,
            prev_loss
        );
        _storeAccuracyLoss(msg.sender, ads, accuracies, losses);

        _processFeedbackLoop(ads, ints);
    }

    function _validateAccuracyLossInputs(
        uint adsLength,
        uint accLength,
        uint lossLength,
        uint16 prev_acc,
        uint16 prev_loss
    ) internal pure {
        require(accLength == adsLength, "INVALID_LENGTH OF ACCURACY ARRAY");
        require(lossLength == adsLength, "INVALID_LENGTH OF LOSS ARRAY");

        require(prev_acc <= type(uint16).max, "PREVIOUS ACCURACY OUT OF RANGE");
        require(prev_loss <= type(uint16).max, "PREVIOUS LOSS OUT OF RANGE");
    }

    function _storeAccuracyLoss(
        address sender,
        address[] memory ads,
        uint16[] calldata accuracies,
        uint16[] calldata losses
    ) internal {
        accuracyLossSubmissions[round][sender].push(
            AccuracyLossSubmission({adrs: ads, acc: accuracies, loss: losses})
        );
    }

    function _processFeedbackLoop(
        address[] memory ads,
        int16[] memory ints
    ) internal {
        for (uint i = 0; i < ads.length; i++) {
            if (!testing) {
                feedback(ads[i], ints[i]);
            }
        }
    }

    function submitFeedbackBytesAndAccuracies(
        bytes calldata raw,
        uint16[] calldata accuracies,
        uint16 prev_acc
    ) external {
        address[] memory ads;
        int16[] memory ints;

        (ads, ints) = parseRaw(raw);

        require(
            accuracies.length == ads.length,
            "INVALID_LENGTH OF ACCURACY ARRAY"
        );

        accuracySubmissions[round][msg.sender].push(
            AccuracySubmission({adrs: ads, acc: accuracies})
        );
        require(
            prev_acc >= 0 && prev_acc <= 10000,
            "PREVIOUS ACCURACY NOT BETWEEN 0 AND 10000 submitFeedbackBytesAndAccuracies"
        );
        prev_accs[round][msg.sender] = prev_acc;

        // EXACT same for-loop as fallback
        for (uint i = 0; i < ads.length; i++) {
            if (!testing) {
                feedback(ads[i], ints[i]);
            }
        }
    }

    function submitFeedbackBytesAndLosses(
        bytes calldata raw,
        uint16[] calldata losses,
        uint16 prev_loss
    ) external {
        address[] memory ads;
        int16[] memory ints;

        (ads, ints) = parseRaw(raw);

        require(losses.length == ads.length, "INVALID_LENGTH OF LOSS ARRAY");
        lossSubmissions[round][msg.sender].push(
            LossSubmission({adrs: ads, loss: losses})
        );

        prev_losses[round][msg.sender] = prev_loss;

        require(
            prev_loss >= 0 && prev_loss <= 10000,
            "PREVIOUS LOSS NOT BETWEEN 0 AND 10000 in submitFeedbackBytesAndLosses"
        );

        // EXACT same for-loop as fallback
        for (uint i = 0; i < ads.length; i++) {
            if (!testing) {
                feedback(ads[i], ints[i]);
            }
        }
    }

    function parseRaw(
        bytes calldata raw
    ) internal pure returns (address[] memory ads, int16[] memory ints) {
        assembly {
            let tmp := 0
            let tmp2 := 0

            // offset inside `raw` starts at raw.offset
            let offset := raw.offset
            // adsCount = calldatasize / 0x34
            let adsCount := div(raw.length, 0x34)

            // allocate memory for addresses array
            ads := mload(0x40)
            mstore(0x40, add(ads, add(0x20, mul(adsCount, 0x20))))
            mstore(ads, adsCount)

            // load addresses (20 bytes each)
            for {
                let i := 0
            } lt(i, adsCount) {
                i := add(i, 1)
            } {
                tmp := calldataload(offset)
                tmp := shr(96, tmp)
                mstore(add(add(ads, 0x20), mul(i, 0x20)), tmp)
                offset := add(offset, 0x14)
            }

            // allocate memory for ints array
            ints := mload(0x40)
            mstore(0x40, add(ints, add(0x20, mul(adsCount, 0x20))))
            mstore(ints, adsCount)

            // load int256 values (32 bytes each)
            for {
                let i := 0
            } lt(i, adsCount) {
                i := add(i, 1)
            } {
                tmp2 := calldataload(offset)
                mstore(add(add(ints, 0x20), mul(i, 0x20)), tmp2)
                offset := add(offset, 0x20)
            }
        }
    }

    function getAllPreviousAccuraciesAndLosses()
        external
        view
        returns (
            uint16[] memory previous_accuracies,
            uint16[] memory previous_losses
        )
    {
        uint8 count_merged_participants = 0;
        for (uint i = 0; i < participants.length; i++) {
            User storage u = users[participants[i]];
            if (u.isRegistered && !u.isDisqualified && u.roundReputation >= 0) {
                count_merged_participants += 1;
            }
        }

        previous_accuracies = new uint16[](count_merged_participants);
        previous_losses = new uint16[](count_merged_participants);
        uint8 j = 0;
        for (uint i = 0; i < participants.length; i++) {
            User storage u = users[participants[i]];
            if (u.isRegistered && !u.isDisqualified && u.roundReputation >= 0) {
                previous_accuracies[j] = prev_accs[round][participants[i]];
                previous_losses[j] = prev_losses[round][participants[i]];
                j++;
            }
        }
    }

    function getAllAccuraciesLossesAbout(
        address target
    )
        external
        view
        returns (
            address[] memory voters,
            uint16[] memory accuracies,
            uint16[] memory losses
        )
    {
        uint totalCount = 0;

        // 1️. First, count total matching entries to size arrays
        for (uint i = 0; i < participants.length; i++) {
            User storage sender = users[participants[i]];
            uint subCount = accuracyLossSubmissions[round][sender.addr].length;

            for (uint j = 0; j < subCount; j++) {
                AccuracyLossSubmission storage sub = accuracyLossSubmissions[
                    round
                ][sender.addr][j];

                for (uint k = 0; k < sub.adrs.length; k++) {
                    if (sub.adrs[k] == target && _isEligibleVoter(sender)) {
                        // TODO: GØR whitelisted eller lign. ACCESSIBLE OG CLEAR DEN EFTER ROUND END!
                        totalCount++;
                    }
                }
            }
        }

        // 2. Allocate arrays
        voters = new address[](totalCount);
        accuracies = new uint16[](totalCount);
        losses = new uint16[](totalCount);

        uint idx = 0;

        // 3. Fill arrays
        for (uint i = 0; i < participants.length; i++) {
            User storage sender = users[participants[i]];
            uint subCount = accuracyLossSubmissions[round][sender.addr].length;

            for (uint j = 0; j < subCount; j++) {
                AccuracyLossSubmission storage sub = accuracyLossSubmissions[
                    round
                ][sender.addr][j];

                for (uint k = 0; k < sub.adrs.length; k++) {
                    if (sub.adrs[k] == target && _isEligibleVoter(sender)) {
                        voters[idx] = sender.addr;
                        accuracies[idx] = sub.acc[k];
                        losses[idx] = sub.loss[k];
                        idx++;
                    }
                }
            }
        }
    }

    function getAllAccuraciesAbout(
        address target
    )
        external
        view
        returns (address[] memory voters, uint16[] memory accuracies)
    {
        uint totalCount = 0;

        // 1️. First, count total matching entries to size arrays
        for (uint i = 0; i < participants.length; i++) {
            User storage sender = users[participants[i]];
            uint subCount = accuracySubmissions[round][sender.addr].length;

            for (uint j = 0; j < subCount; j++) {
                AccuracySubmission storage sub = accuracySubmissions[round][
                    sender.addr
                ][j];

                for (uint k = 0; k < sub.adrs.length; k++) {
                    if (sub.adrs[k] == target && _isEligibleVoter(sender)) {
                        totalCount++;
                    }
                }
            }
        }

        // 2. Allocate arrays
        voters = new address[](totalCount);
        accuracies = new uint16[](totalCount);

        uint idx = 0;

        // 3. Fill arrays
        for (uint i = 0; i < participants.length; i++) {
            User storage sender = users[participants[i]];
            uint subCount = accuracySubmissions[round][sender.addr].length;

            for (uint j = 0; j < subCount; j++) {
                AccuracySubmission storage sub = accuracySubmissions[round][
                    sender.addr
                ][j];

                for (uint k = 0; k < sub.adrs.length; k++) {
                    if (sub.adrs[k] == target && _isEligibleVoter(sender)) {
                        voters[idx] = sender.addr;
                        accuracies[idx] = sub.acc[k];
                        idx++;
                    }
                }
            }
        }
    }

    function getAllLossesAbout(
        address target
    ) external view returns (address[] memory voters, uint16[] memory losses) {
        uint totalCount = 0;

        // 1️. First, count total matching entries to size arrays
        for (uint i = 0; i < participants.length; i++) {
            User storage sender = users[participants[i]];
            uint subCount = lossSubmissions[round][sender.addr].length;

            for (uint j = 0; j < subCount; j++) {
                LossSubmission storage sub = lossSubmissions[round][
                    sender.addr
                ][j];

                for (uint k = 0; k < sub.adrs.length; k++) {
                    if (sub.adrs[k] == target && _isEligibleVoter(sender)) {
                        totalCount++;
                    }
                }
            }
        }

        // 2. Allocate arrays
        voters = new address[](totalCount);
        losses = new uint16[](totalCount);

        uint idx = 0;

        // 3. Fill arrays
        for (uint i = 0; i < participants.length; i++) {
            User storage sender = users[participants[i]];
            uint subCount = lossSubmissions[round][sender.addr].length;

            for (uint j = 0; j < subCount; j++) {
                LossSubmission storage sub = lossSubmissions[round][
                    sender.addr
                ][j];

                for (uint k = 0; k < sub.adrs.length; k++) {
                    if (sub.adrs[k] == target && _isEligibleVoter(sender)) {
                        voters[idx] = sender.addr;
                        losses[idx] = sub.loss[k];
                        idx++;
                    }
                }
            }
        }
    }

    function _isEligibleVoter(
        User storage sender
    ) internal view returns (bool) {
        return
            sender.isRegistered &&
            !sender.isDisqualified &&
            sender.roundReputation >= 0;
    }

    // Subtract every vote ever cast by `kickedUser` from the per-target
    // receiver tallies, and zero out the per-(voter, target) entries.
    // Called from settle() at each disqualification site so that GIR's
    // (positive/total)^2 input ignores votes from kicked participants.
    function _removeKickedUserVotesFromTallies(address kickedUser) internal {
        for (uint i = 0; i < participants.length; i++) {
            address target = participants[i];
            uint256 pos = positiveVotesGivenByTo[kickedUser][target];
            uint256 tot = totalVotesGivenByTo[kickedUser][target];
            if (tot == 0) continue;

            positiveVotesReceived[target] -= pos;
            totalVotesReceived[target] -= tot;
            positiveVotesGivenByTo[kickedUser][target] = 0;
            totalVotesGivenByTo[kickedUser][target] = 0;
        }
    }

    function _isEligibleForRewards(
        User storage user
    ) internal view returns (bool) {
        return (user.isRegistered &&
            user.whitelistedForRewards &&
            !user.isPunished &&
            !user.isDisqualified);
    }

    struct TaskRep {
        address user;
        int256 delta;
        uint globalReputationScore;
        // Cross-round vote tallies for this participant, with kicked voters
        // already excluded. JobListing feeds these into the GIR formula.
        uint256 positiveVotes;
        uint256 totalVotes;
    }

    function getTaskRepDeltaAndGRS() public view returns (TaskRep[] memory) {
        uint len = participants.length;
        TaskRep[] memory taskReps = new TaskRep[](len);

        for (uint i = 0; i < len; i++) {
            address addr = participants[i];
            User storage user = users[addr];

            taskReps[i] = TaskRep({
                user: addr,
                delta: user.taskRepDelta,
                globalReputationScore: user.globalReputationScore,
                positiveVotes: positiveVotesReceived[addr],
                totalVotes: totalVotesReceived[addr]
            });
        }

        return taskReps;
    }

    // Push this challenge's reputation deltas to the manager.
    // Called by Python after each challenge, or by users/contracts in production.
    function finalizeReputations() external {
        require(taskType != TaskType.template, "Template challenge");
        require(managerAddress != address(0), "No manager");
        IOpenFLManager(managerAddress).updateReputationsFromChallenge(
            address(this),
            taskType
        );
    }

    // Fallback function parses dynamic size feedback arrays
    // @dev This allows the contract to have an arbitrary number of participants
    fallback() external {
        address[] memory ads;
        int16[] memory ints;

        assembly {
            let tmp := 0
            let tmp2 := 0

            // Skip : function selector    : 0x4 bytes
            let offset := 0x00

            // Compute the number of addresses :
            // ((array length - 0x04) - 0x20) / 0x14
            // ((array length - sizeof(function Selector)) - sizeof(uint256)) / sizeof(address)
            let adsCount := div(calldatasize(), 0x34)

            // Allocate memory for the address array
            ads := mload(0x40)
            mstore(0x40, add(ads, add(0x20, mul(adsCount, 0x20))))

            // Set the size of the array
            mstore(ads, adsCount)

            // Get an address from calldata on each iteration :
            // loads 0x20 bytes from calldata starting at offset : calldata[offset: offset + 0x20)
            // shift value by 96 bits (12 bytes) to the right to keep only the relevant portion (first 20 bytes)
            // store that value at ads[i]
            // increments calldata offset by 0x14 (20 bytes)
            for {
                let i := 0
            } lt(i, adsCount) {
                i := add(i, 1)
            } {
                tmp := calldataload(offset)
                tmp := shr(96, tmp)
                mstore(add(add(ads, 0x20), mul(i, 0x20)), tmp)
                offset := add(offset, 0x14)
            }

            // Allocate memory for the address array
            ints := mload(0x40)
            mstore(0x40, add(ints, add(0x20, mul(adsCount, 0x20))))

            // Set the size of the array
            mstore(ints, adsCount)

            // Get an address from calldata on each iteration :
            // loads 0x20 bytes from calldata starting at offset : calldata[offset: offset + 0x20)
            // store that value at ads[i]
            // increments calldata offset by 0x20 (32 bytes)
            for {
                let i := 0
            } lt(i, adsCount) {
                i := add(i, 1)
            } {
                tmp2 := calldataload(offset)
                mstore(add(add(ints, 0x20), mul(i, 0x20)), tmp2)
                offset := add(offset, 0x20)
            }
        }

        for (uint i = 0; i < ads.length; i++) {
            if (!testing) {
                feedback(ads[i], ints[i]);
            }
        }
    }
    // Missing taskRepDelta
    // function getUser(
    //     address u
    // )
    //     external
    //     view
    //     returns (
    //         address,
    //         int256,
    //         uint,
    //         int,
    //         uint8,
    //         uint8,
    //         bool,
    //         bool,
    //         bool,
    //         bool
    //     )
    // {
    //     User storage user = users[u];
    //     return (
    //         user.addr,
    //         user.weightedContribScore,
    //         user.globalReputationScore,
    //         user.roundReputation,
    //         user.nrOfRoundsParticipated,
    //         user.nrOfVotesFromUser,
    //         user.isPunished,
    //         user.isRegistered,
    //         user.whitelistedForRewards,
    //         user.isDisqualified
    //     );
    // }

    function absUint(int x) public pure returns (uint) {
        return x >= 0 ? uint(x) : uint(-x);
    }
}
