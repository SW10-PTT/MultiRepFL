// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

import "forge-std/Test.sol";
import "../../contracts/OpenFLChallenge.sol";
import "../../contracts/Types.sol";

// Harness exposing helpers needed to test the cross-round vote tally lifecycle
// without driving the full registration / round-settlement flow. Constructed
// with TaskType.template so the OpenFLChallenge constructor skips its
// JobListing.getSelectedParticipants() lookup.
contract OpenFLChallengeHarness is OpenFLChallenge {
    constructor()
        OpenFLChallenge(
            ChallengeSpecifications({
                modelHash: bytes32("h"),
                min_collateral: 1e18,
                max_collateral: 2e18,
                managerAddress: address(0),
                reward: 1e18,
                min_rounds: 3,
                punishfactor: 3,
                punishfactorContrib: 3,
                freeriderPenalty: 50,
                taskType: TaskType.template,
                jobListingAddress: address(0)
            })
        )
    {}

    function tForceRegister(address u, uint256 balance) external {
        User storage user = users[u];
        user.isRegistered = true;
        user.isSelected = true;
        user.addr = u;
        user.nrOfRoundsParticipated = 1;
        user.globalReputationScore = balance;
        participants.push(u);
        nrOfActiveParticipants += 1;
    }

    function tCastVote(address voter, address target, int256 score) external {
        // Direct mutation mirroring feedback()'s tally bookkeeping. Bypasses
        // the per-round modifiers so individual scenarios can be staged
        // independently of round timing.
        totalVotesGivenByTo[voter][target] += 1;
        totalVotesReceived[target] += 1;
        if (score == 1) {
            positiveVotesGivenByTo[voter][target] += 1;
            positiveVotesReceived[target] += 1;
        }
    }

    function tRemoveKickedUserVotes(address kicked) external {
        _removeKickedUserVotesFromTallies(kicked);
    }
}

contract GirVotesTest is Test {
    OpenFLChallengeHarness h;

    address constant A = address(0xA1);
    address constant B = address(0xB2);
    address constant C = address(0xC3);

    function setUp() public {
        h = new OpenFLChallengeHarness();
        h.tForceRegister(A, 1e18);
        h.tForceRegister(B, 1e18);
        h.tForceRegister(C, 1e18);
    }

    // -----------------------------------------------------------
    // Vote tally accumulation
    // -----------------------------------------------------------

    function testTally_singlePositive() public {
        h.tCastVote(A, B, 1);

        assertEq(h.positiveVotesReceived(B), 1);
        assertEq(h.totalVotesReceived(B), 1);
        assertEq(h.positiveVotesGivenByTo(A, B), 1);
        assertEq(h.totalVotesGivenByTo(A, B), 1);
    }

    function testTally_neutralCountsAsTotal() public {
        h.tCastVote(A, B, 0);

        assertEq(h.positiveVotesReceived(B), 0);
        assertEq(h.totalVotesReceived(B), 1);
    }

    function testTally_negativeCountsAsTotal() public {
        h.tCastVote(A, B, -1);

        assertEq(h.positiveVotesReceived(B), 0);
        assertEq(h.totalVotesReceived(B), 1);
    }

    function testTally_accumulatesAcrossRounds() public {
        // Same voter votes for same target across many rounds.
        h.tCastVote(A, B, 1);
        h.tCastVote(A, B, 1);
        h.tCastVote(A, B, -1);
        h.tCastVote(A, B, 0);

        assertEq(h.positiveVotesReceived(B), 2);
        assertEq(h.totalVotesReceived(B), 4);
    }

    function testTally_multipleVoters() public {
        h.tCastVote(A, B, 1);
        h.tCastVote(C, B, 1);
        h.tCastVote(C, B, -1);

        assertEq(h.positiveVotesReceived(B), 2);
        assertEq(h.totalVotesReceived(B), 3);
    }

    // -----------------------------------------------------------
    // Kicked voters' contributions are subtracted
    // -----------------------------------------------------------

    function testKick_subtractsAllVotesFromTarget() public {
        h.tCastVote(A, B, 1);
        h.tCastVote(A, B, 1);
        h.tCastVote(A, B, -1);

        h.tRemoveKickedUserVotes(A);

        assertEq(h.positiveVotesReceived(B), 0);
        assertEq(h.totalVotesReceived(B), 0);
        assertEq(h.positiveVotesGivenByTo(A, B), 0);
        assertEq(h.totalVotesGivenByTo(A, B), 0);
    }

    function testKick_leavesOtherVotersUntouched() public {
        h.tCastVote(A, B, 1);
        h.tCastVote(A, B, 1);
        h.tCastVote(C, B, 1);
        h.tCastVote(C, B, -1);

        h.tRemoveKickedUserVotes(A);

        // C's votes still count.
        assertEq(h.positiveVotesReceived(B), 1);
        assertEq(h.totalVotesReceived(B), 2);
    }

    function testKick_spansAllTargets() public {
        // Voter A votes for both B and C; B for A.
        h.tCastVote(A, B, 1);
        h.tCastVote(A, C, 1);
        h.tCastVote(A, C, 0);
        h.tCastVote(B, A, 1);

        h.tRemoveKickedUserVotes(A);

        assertEq(h.positiveVotesReceived(B), 0);
        assertEq(h.totalVotesReceived(B), 0);
        assertEq(h.positiveVotesReceived(C), 0);
        assertEq(h.totalVotesReceived(C), 0);
        // A's own incoming tally (from B) is unaffected — only A's outgoing
        // votes get removed.
        assertEq(h.positiveVotesReceived(A), 1);
        assertEq(h.totalVotesReceived(A), 1);
    }

    function testKick_isIdempotent() public {
        h.tCastVote(A, B, 1);

        h.tRemoveKickedUserVotes(A);
        h.tRemoveKickedUserVotes(A); // no-op second call must not underflow

        assertEq(h.positiveVotesReceived(B), 0);
        assertEq(h.totalVotesReceived(B), 0);
    }
}
