pragma solidity ^0.8.0;

import "./OpenFLModel.sol";
import "./Types.sol";

interface Manager {
    function getUserRep(
        address,
        TaskType
    ) external view returns (uint, uint, uint);
}

contract JobListing {
    modifier onlyNotYetRegisteredUsers() {
        require(applicants[msg.sender].addr == address(0), "SAR");
        _;
    }

    modifier applicationWindowClosed() {
        require(block.timestamp >= applicationWindowCloseTime, "Too early");
        _;
    }

    event SelectionComplete(address[] participants);
    event ChallengeContractCreated(address challengeContractAddress);

    struct User {
        uint globalTaskRep; // 32
        uint globalIntegrity; // 32
        uint nrOfTasksParticipated; // 1
        address addr; // 20
        bool isSelected; // 1
    }
    mapping(address => User) public applicants;

    uint public applicationWindowCloseTime;
    TaskType taskType;
    Manager manager;
    address[] applicantAddresses;
    uint16 nrOfApplicants;
    address managerAddress;
    TrainingSpecifications trainingSpecs;

    constructor(
        bytes32 _modelHash,
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
        manager = Manager(_managerAddress);
        taskType = _taskType;
        applicationWindowCloseTime = block.timestamp + 1 seconds;

        trainingSpecs.freeriderPenalty = _freeriderPenalty;
        trainingSpecs.managerAddress = _managerAddress;
        trainingSpecs.max_collateral = _max_collateral;
        trainingSpecs.min_collateral = _min_collateral;
        trainingSpecs.min_rounds = _min_rounds;
        trainingSpecs.modelHash = _modelHash;
        trainingSpecs.punishfactor = _punishfactor;
        trainingSpecs.punishfactorContrib = _punishfactorContrib;
        trainingSpecs.reward = _reward;
        trainingSpecs.taskType = _taskType;
    }

    function register() public payable onlyNotYetRegisteredUsers {
        //require(
        //    msg.value >= min_collateral && msg.value <= max_collateral,
        //    "NWR"
        //);
        registrationProcess(msg.sender);
    }

    function registrationProcess(address userAddr) internal {
        User storage user = applicants[userAddr];

        (
            uint taskRep,
            uint globalIntegrity,
            uint nrOfTasksParticipated
        ) = manager.getUserRep(userAddr, taskType);
        user.globalTaskRep = taskRep;
        user.globalIntegrity = globalIntegrity;
        user.nrOfTasksParticipated = nrOfTasksParticipated;
        user.addr = userAddr;
        user.isSelected = false;

        nrOfApplicants += 1;
        applicantAddresses.push(userAddr);
    }

    function decideOnParticpants(
        uint8 amount
    ) public payable applicationWindowClosed {
        address[] memory selected = getTopN(amount);

        for (uint i = 0; i < selected.length; i++) {
            applicants[selected[i]].isSelected = true;
        }

        trainingSpecs.selectedParticipants = selected;

        emit SelectionComplete(selected);
    }

    function CreateNewTrainingContract() public payable {
        require(
            msg.value >= trainingSpecs._reward + trainingSpecs._min_collateral,
            "NEV"
        );

        OpenFLModel listing = new OpenFLModel{value: trainingSpecs._reward}(
            trainingSpecs
        );

        address listingAddr = address(listing);

        emit ChallengeContractCreated(listingAddr);
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
}
