pragma solidity =0.8.9;

import "./OpenFLModel.sol";

contract JobListing {
    constructor(
        bytes32 _modelHash,
        uint _min_collateral,
        uint _max_collateral,
        uint _reward,
        uint8 _min_rounds,
        uint8 _punishfactor,
        uint8 _punishfactorContrib,
        uint8 _freeriderPenalty
    ) payable {}
}
