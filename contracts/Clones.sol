// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

library Clones {
    function clone(address implementation) internal returns (address instance) {
        assembly {
            let ptr := mload(0x40)
            mstore(ptr, hex"3d602d80600a3d3981f3")
            mstore(add(ptr, 0x14), shl(0x60, implementation))
            mstore(add(ptr, 0x28), hex"5af43d82803e903d91602b57fd5bf3")
            instance := create(0, ptr, 0x37)
        }
        require(instance != address(0), "ERC1167: create failed");
    }
}
