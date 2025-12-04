from uno import Uno
from typing import Optional, Tuple

class GameController:
    """
    Controller for managing games between two players.

    Handles:
    - Game initialization
    - Turn management
    - Belief updates for AI players
    - Game statistics
    """

    def __init__(self, player1, player2, verbose: bool = True):
        """
        Initialize game controller.

        Args:
            player1: Player 1 (Uno_AI or Uno_Heuristic)
            player2: Player 2 (Uno_AI or Uno_Heuristic)
            verbose: Whether to print game progress
        """
        self.player1 = player1
        self.player2 = player2
        self.verbose = verbose
        self.game = None
        self.current_player = 1
        self.turn_count = 0

        # Initialize/Clear log file
        if self.verbose:
            with open("game_verbose_log.txt", "w", encoding='utf-8') as f:
                f.write("=== UNO GAME LOG ===\n")

        # Configure players
        self.player1.player_id = 1
        self.player2.player_id = 2

    def init_game(self, seed: Optional[int] = None):
        """Initialize new game."""
        self.game = Uno()
        self.game.new_game(seed=seed)

        # Set game reference for players
        if hasattr(self.player1, 'init_belief'):
            self.player1.init_belief(self.game)
        if hasattr(self.player1, 'set_game'):
            self.player1.set_game(self.game)

        if hasattr(self.player2, 'init_belief'):
            self.player2.init_belief(self.game)
        if hasattr(self.player2, 'set_game'):
            self.player2.set_game(self.game)

        self.current_player = 1
        self.turn_count = 0

        if self.verbose:
            print("=" * 60)
            print("NEW GAME STARTED")
            print("=" * 60)
            print(f"Player 1 hand: {self.game.H_1}")
            print(f"Player 2 hand size: {len(self.game.H_2)}")
            print(f"Top card: {self.game.P_t}")
            print("=" * 60)

    def play_turn(self) -> bool:
        """
        Play one turn with detailed logging of MDP and POMDP states.
        Logs the full "God" state, the AI's belief state, and the specific action or inaction taken.
        """
        # 0. Check Game Over immediately
        if self.game.G_o == 'GameOver':
            return False

        self.turn_count += 1

        # Identify current player
        player = self.player1 if self.current_player == 1 else self.player2
        player_name = "Player 1" if self.current_player == 1 else "Player 2"

        # --- LOGGING START: PRE-ACTION STATE ---
        if self.verbose:
            # 1. Prepare "God" State (MDP)
            self.game.create_S()  # Refresh state to ensure accuracy
            s = self.game.State   # (H_1, H_2, D_g, P, P_t, G_o)

            # Build the log string
            log_str = f"\n{'='*40}\n"
            log_str += f"TURN {self.turn_count}: {player_name}'s Turn\n"
            log_str += f"{'='*40}\n"

            # Log MDP Variables (God View)
            log_str += f"[MDP GOD STATE]\n"
            log_str += f"  H_1 (P1 Hand): {s[0]}\n"
            log_str += f"  H_2 (P2 Hand): {s[1]}\n"
            log_str += f"  D_g (Deck):    {len(s[2])} cards {s[2]}\n" # detailed list + count
            log_str += f"  P   (Pile):    {len(s[3])} cards (Top: {s[4]})\n"
            log_str += f"  P_t (Target):  {s[4]}\n"
            log_str += f"  G_o (Status):  {s[5]}\n"

            # 2. Log AI's POMDP State (Belief) if available
            # We typically only track belief for the AI (Player 1), but we check whoever has it.
            if hasattr(self.player1, 'belief') and self.player1.belief:
                b = self.player1.belief
                log_str += f"\n[AI POMDP STATE (Player 1 Perspective)]\n"
                log_str += f"  Known H_1:      {b.H_1}\n"
                log_str += f"  Belief |H_2|:   {b.h2_size}\n"
                log_str += f"  Belief |D_g|:   {b.dg_size}\n"
                log_str += f"  Unknown L:      {len(b.L)} cards\n" # List can be huge, maybe just count or partial
                # Uncomment next line if you want the full list of unknown cards printed:
                # log_str += f"  L (Full):       {b.L}\n"
                log_str += f"  Legal N(P_t):   {b.N_Pt}\n"
                log_str += f"  Posterior Mode: {b.posterior_mode}\n"
                log_str += f"  P(No Legal):    {b._prob_no_legal():.4f}\n"
                log_str += f"  Entropy:        {b.entropy():.4f}\n"

            log_str += "-"*40 + "\n"

            # Print and Save
            print(log_str, end='')
            with open("game_verbose_log.txt", "a", encoding='utf-8') as f:
                f.write(log_str)
        # --- LOGGING END ---

        # 1. Choose Action
        action = player.choose_action()

        # Handle Case: No Action Chosen (Inaction)
        if action is None:
            msg = f"> INACTION: {player_name} could not find a valid action (returned None).\n"
            if self.verbose:
                print(msg)
                with open("game_verbose_log.txt", "a", encoding='utf-8') as f:
                    f.write(msg)
            return False

        # 2. Execute Action
        success = self.game.execute_action(action, self.current_player)

        # Handle Case: Invalid Action (Inaction/Failure)
        if not success:
            msg = f"> INACTION: {player_name} attempted invalid action {action} and failed.\n"
            if self.verbose:
                print(msg)
                with open("game_verbose_log.txt", "a", encoding='utf-8') as f:
                    f.write(msg)
            return False

        # 3. Log Successful Action
        if self.verbose:
            if action.is_play():
                action_msg = f"> ACTION: {player_name} played {action.X_1}\n"
            else:
                action_msg = f"> ACTION: {player_name} drew {action.n} card(s)\n"

            print(action_msg)
            with open("game_verbose_log.txt", "a", encoding='utf-8') as f:
                f.write(action_msg)

        # 4. Update Beliefs
        # If the opponent is an AI, they need to observe this action to update their belief
        opponent = self.player2 if self.current_player == 1 else self.player1
        if hasattr(opponent, 'update_belief'):
            opponent.update_belief(action)

        # 5. Check Win Condition
        if self.game.G_o == 'GameOver':
            winner = 1 if len(self.game.H_1) == 0 else 2
            if self.verbose:
                win_msg = f"\n{'='*40}\n🎉 GAME OVER: Player {winner} wins in {self.turn_count} turns!\n{'='*40}\n"
                print(win_msg)
                with open("game_verbose_log.txt", "a", encoding='utf-8') as f:
                    f.write(win_msg)
            return False

        # 6. Switch Player
        self.current_player = 3 - self.current_player
        return True

    def play_game(self, seed: Optional[int] = None, max_turns: int = 500) -> Tuple[int, int]:
        """
        Play complete game.

        Args:
            seed: Random seed for game initialization
            max_turns: Maximum turns before declaring draw

        Returns:
            Tuple of (winner, num_turns) where winner is 1, 2, or 0 for draw
        """
        self.init_game(seed=seed)

        while self.turn_count < max_turns:
            continue_game = self.play_turn()
            if not continue_game:
                break

        # Determine winner
        if self.game.G_o == 'GameOver':
            if len(self.game.H_1) == 0:
                return 1, self.turn_count
            elif len(self.game.H_2) == 0:
                return 2, self.turn_count

        # Draw (max turns reached)
        if self.verbose:
            print(f"\nGame ended in draw after {max_turns} turns")
        return 0, self.turn_count