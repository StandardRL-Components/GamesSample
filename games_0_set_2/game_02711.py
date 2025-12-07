
# Generated: 2025-08-28T05:42:22.124293
# Source Brief: brief_02711.md
# Brief Index: 2711

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to select a piece and then a move. Space to confirm. Shift to deselect."
    )

    game_description = (
        "A visual checkers game. Strategically move your pieces (blue circles) to capture all of the opponent's pieces (red squares). Pieces reaching the far end are 'kinged' and can move backwards."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.BOARD_SIZE = 8
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_BOARD_DARK = (40, 40, 60)
        self.COLOR_BOARD_LIGHT = (60, 60, 90)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_OPPONENT = (255, 80, 80)
        self.COLOR_KING_ACCENT = (255, 215, 0)
        self.COLOR_SELECTION = (255, 255, 0, 180)
        self.COLOR_SELECTOR = (100, 255, 100, 150)
        self.COLOR_VALID_MOVE = (100, 255, 100, 80)
        self.COLOR_VALID_JUMP = (255, 100, 100, 80)
        self.COLOR_UI_TEXT = (220, 220, 220)

        # Board rendering setup
        self.CELL_SIZE = 40
        self.BOARD_DIM = self.BOARD_SIZE * self.CELL_SIZE
        self.BOARD_OFFSET_X = (self.WIDTH - self.BOARD_DIM) // 2
        self.BOARD_OFFSET_Y = (self.HEIGHT - self.BOARD_DIM) // 2

        # Piece IDs
        self.EMPTY = 0
        self.PLAYER_PIECE = 1
        self.OPPONENT_PIECE = 2
        self.PLAYER_KING = 3
        self.OPPONENT_KING = 4

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans", 18)
        self.font_large = pygame.font.SysFont("sans", 24)
        
        # Internal state variables are initialized in reset()
        self.board = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_turn = True
        self.selected_piece = None
        self.selector_pos = None
        self.forced_jump_pos = None
        self.particles = []
        self.player_pieces = 0
        self.opponent_pieces = 0
        self.last_player_captures = 0
        self.last_player_kinged = False
        self.last_opponent_captures = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.uint8)
        self._setup_board()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_turn = True
        self.selected_piece = None
        self.selector_pos = None
        self.forced_jump_pos = None
        self.particles = []
        self.player_pieces = 12
        self.opponent_pieces = 12
        self.last_player_captures = 0
        self.last_player_kinged = False
        self.last_opponent_captures = 0

        return self._get_observation(), self._get_info()

    def _setup_board(self):
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if (r + c) % 2 == 1:
                    if r < 3:
                        self.board[r, c] = self.OPPONENT_PIECE
                    elif r > 4:
                        self.board[r, c] = self.PLAYER_PIECE

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.last_player_captures = 0
        self.last_player_kinged = False
        self.last_opponent_captures = 0
        
        turn_ended = self._handle_player_action(movement, space_held, shift_held)

        if turn_ended:
            self.player_turn = False
            self._check_and_promote_pieces()
            if self._check_win_condition():
                self.game_over = True
            else:
                # Opponent takes its turn immediately
                self._opponent_turn()
                self._check_and_promote_pieces()
                if self._check_loss_condition():
                    self.game_over = True
            self.player_turn = True

        self.steps += 1
        reward = self._calculate_reward()
        self.score += reward
        terminated = self._check_termination()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_action(self, movement, space_held, shift_held):
        turn_ended = False

        if shift_held:
            self.selected_piece = None
            self.selector_pos = None
            if self.forced_jump_pos is None: # Can't cancel a forced jump
                return False

        # If forced to jump, only allow moves with that piece
        if self.forced_jump_pos:
            self.selected_piece = self.forced_jump_pos
        
        # --- SELECTION LOGIC ---
        if movement != 0 and self.selected_piece is None:
            player_pieces = np.argwhere((self.board == self.PLAYER_PIECE) | (self.board == self.PLAYER_KING))
            if len(player_pieces) > 0:
                if movement == 1: # Up -> Top-most
                    self.selected_piece = tuple(player_pieces[player_pieces[:, 0].argmin()])
                elif movement == 2: # Down -> Bottom-most
                    self.selected_piece = tuple(player_pieces[player_pieces[:, 0].argmax()])
                elif movement == 3: # Left -> Left-most
                    self.selected_piece = tuple(player_pieces[player_pieces[:, 1].argmin()])
                elif movement == 4: # Right -> Right-most
                    self.selected_piece = tuple(player_pieces[player_pieces[:, 1].argmax()])
                self.selector_pos = None

        # --- MOVE SELECTION LOGIC ---
        elif movement != 0 and self.selected_piece is not None:
            r, c = self.selected_piece
            if movement == 1: self.selector_pos = (r - 1, c - 1)
            elif movement == 2: self.selector_pos = (r - 1, c + 1)
            elif movement == 3: self.selector_pos = (r + 1, c - 1)
            elif movement == 4: self.selector_pos = (r + 1, c + 1)

        # --- CONFIRMATION LOGIC ---
        if space_held and self.selected_piece and self.selector_pos:
            valid_moves = self._get_valid_moves(self.selected_piece[0], self.selected_piece[1])
            is_jump = self.selector_pos in valid_moves['jumps']
            is_move = self.selector_pos in valid_moves['moves']
            
            # Prioritize jumps if available
            can_jump = len(valid_moves['jumps']) > 0
            if self.forced_jump_pos:
                can_jump = True

            if is_jump:
                captures = self._execute_move(self.selected_piece, self.selector_pos)
                self.last_player_captures += captures
                # Check for multi-jump
                new_jumps = self._get_valid_moves(self.selector_pos[0], self.selector_pos[1])['jumps']
                if len(new_jumps) > 0:
                    self.forced_jump_pos = self.selector_pos # Must continue jumping
                    self.selected_piece = self.selector_pos
                    self.selector_pos = None
                else:
                    self.forced_jump_pos = None
                    turn_ended = True
            elif not can_jump and is_move:
                self._execute_move(self.selected_piece, self.selector_pos)
                turn_ended = True
        
        if turn_ended:
            self.selected_piece = None
            self.selector_pos = None

        return turn_ended

    def _opponent_turn(self):
        # Simple AI: 1. Prioritize jumps, 2. Make a random move
        all_jumps = []
        all_moves = []
        opponent_pieces = np.argwhere((self.board == self.OPPONENT_PIECE) | (self.board == self.OPPONENT_KING))
        self.np_random.shuffle(opponent_pieces) # Randomize piece order

        for r, c in opponent_pieces:
            valid = self._get_valid_moves(r, c)
            for jump in valid['jumps']:
                all_jumps.append(((r, c), jump))
            for move in valid['moves']:
                all_moves.append(((r, c), move))

        if all_jumps:
            start_pos, end_pos = self.np_random.choice(all_jumps)
            # Handle multi-jump for AI
            while True:
                captures = self._execute_move(start_pos, end_pos)
                self.last_opponent_captures += captures
                # sfx: opponent_move
                new_jumps = self._get_valid_moves(end_pos[0], end_pos[1])['jumps']
                if new_jumps:
                    start_pos = end_pos
                    end_pos = self.np_random.choice(new_jumps)
                else:
                    break
        elif all_moves:
            start_pos, end_pos = self.np_random.choice(all_moves)
            self._execute_move(start_pos, end_pos)
            # sfx: opponent_move

    def _get_valid_moves(self, r, c):
        moves = {'moves': [], 'jumps': []}
        piece = self.board[r, c]
        if piece == self.EMPTY: return moves

        is_player = piece in [self.PLAYER_PIECE, self.PLAYER_KING]
        is_king = piece in [self.PLAYER_KING, self.OPPONENT_KING]
        
        move_dirs = [(-1, -1), (-1, 1)] if is_player else [(1, -1), (1, 1)]
        if is_king:
            move_dirs += [(1, -1), (1, 1)] if is_player else [(-1, -1), (-1, 1)]

        for dr, dc in move_dirs:
            nr, nc = r + dr, c + dc
            # Simple move
            if 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr, nc] == self.EMPTY:
                moves['moves'].append((nr, nc))
            # Jump
            elif 0 <= nr < 8 and 0 <= nc < 8:
                opponent_piece_ids = [self.OPPONENT_PIECE, self.OPPONENT_KING] if is_player else [self.PLAYER_PIECE, self.PLAYER_KING]
                if self.board[nr, nc] in opponent_piece_ids:
                    jr, jc = r + 2*dr, c + 2*dc
                    if 0 <= jr < 8 and 0 <= jc < 8 and self.board[jr, jc] == self.EMPTY:
                        moves['jumps'].append((jr, jc))
        return moves

    def _execute_move(self, start_pos, end_pos):
        r1, c1 = start_pos
        r2, c2 = end_pos
        
        piece = self.board[r1, c1]
        self.board[r2, c2] = piece
        self.board[r1, c1] = self.EMPTY
        
        captures = 0
        # If it's a jump, remove the captured piece
        if abs(r1 - r2) == 2:
            mr, mc = (r1 + r2) // 2, (c1 + c2) // 2
            is_player_move = piece in [self.PLAYER_PIECE, self.PLAYER_KING]
            if is_player_move:
                self.opponent_pieces -= 1
            else:
                self.player_pieces -= 1
            self.board[mr, mc] = self.EMPTY
            captures = 1
            # sfx: capture
            self._spawn_particles(mc, mr)
        return captures

    def _check_and_promote_pieces(self):
        for c in range(self.BOARD_SIZE):
            # Player piece reaches opponent's side
            if self.board[0, c] == self.PLAYER_PIECE:
                self.board[0, c] = self.PLAYER_KING
                if self.player_turn: self.last_player_kinged = True # sfx: promote
            # Opponent piece reaches player's side
            if self.board[self.BOARD_SIZE - 1, c] == self.OPPONENT_PIECE:
                self.board[self.BOARD_SIZE - 1, c] = self.OPPONENT_KING

    def _calculate_reward(self):
        reward = 0
        if self.game_over:
            if self._check_win_condition(): return 100.0
            if self._check_loss_condition(): return -100.0

        reward += self.last_player_captures * 1.0
        reward += 2.0 if self.last_player_kinged else 0.0
        reward -= self.last_opponent_captures * 1.0

        # Threat calculation
        player_threats, opponent_threats = self._calculate_threats()
        reward += player_threats * 0.1
        reward -= opponent_threats * 0.1
        
        return reward

    def _calculate_threats(self):
        player_threats, opponent_threats = 0, 0
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                piece = self.board[r, c]
                if piece != self.EMPTY:
                    jumps = self._get_valid_moves(r, c)['jumps']
                    if piece in [self.PLAYER_PIECE, self.PLAYER_KING]:
                        player_threats += len(jumps)
                    else:
                        opponent_threats += len(jumps)
        return player_threats, opponent_threats

    def _check_win_condition(self):
        return self.opponent_pieces <= 0
    
    def _check_loss_condition(self):
        return self.player_pieces <= 0

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        
        # Check for no legal moves (stalemate loss)
        if self.player_turn:
            player_pieces = np.argwhere((self.board == self.PLAYER_PIECE) | (self.board == self.PLAYER_KING))
            has_moves = False
            for r, c in player_pieces:
                moves = self._get_valid_moves(r, c)
                if moves['moves'] or moves['jumps']:
                    has_moves = True
                    break
            if not has_moves:
                self.game_over = True
                return True

        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_board()
        self._render_highlights()
        self._render_pieces()
        self._update_and_render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pieces": self.player_pieces,
            "opponent_pieces": self.opponent_pieces
        }

    def _render_board(self):
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                color = self.COLOR_BOARD_LIGHT if (r + c) % 2 == 0 else self.COLOR_BOARD_DARK
                rect = (self.BOARD_OFFSET_X + c * self.CELL_SIZE, self.BOARD_OFFSET_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, color, rect)
        
        # Turn indicator border
        border_color = self.COLOR_PLAYER if self.player_turn else self.COLOR_OPPONENT
        if self.game_over:
            if self._check_win_condition(): border_color = self.COLOR_KING_ACCENT
            elif self._check_loss_condition(): border_color = self.COLOR_OPPONENT
        pygame.draw.rect(self.screen, border_color, (self.BOARD_OFFSET_X-2, self.BOARD_OFFSET_Y-2, self.BOARD_DIM+4, self.BOARD_DIM+4), 4, 3)

    def _render_highlights(self):
        # Highlight selected piece
        if self.selected_piece:
            r, c = self.selected_piece
            x = self.BOARD_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
            y = self.BOARD_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(s, self.COLOR_SELECTION, (self.CELL_SIZE//2, self.CELL_SIZE//2), self.CELL_SIZE // 2 - 2)
            self.screen.blit(s, (x - self.CELL_SIZE//2, y - self.CELL_SIZE//2))

            # Highlight valid moves for selected piece
            valid_moves = self._get_valid_moves(r, c)
            forced = len(valid_moves['jumps']) > 0 or self.forced_jump_pos is not None
            if not forced:
                for mr, mc in valid_moves['moves']:
                    self._draw_highlight_square(mc, mr, self.COLOR_VALID_MOVE)
            for jr, jc in valid_moves['jumps']:
                self._draw_highlight_square(jc, jr, self.COLOR_VALID_JUMP)

        # Highlight selector
        if self.selector_pos:
            r, c = self.selector_pos
            self._draw_highlight_square(c, r, self.COLOR_SELECTOR)

    def _draw_highlight_square(self, c, r, color):
        if 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE:
            rect = (self.BOARD_OFFSET_X + c * self.CELL_SIZE, self.BOARD_OFFSET_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, rect)

    def _render_pieces(self):
        radius = self.CELL_SIZE // 2 - 6
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                piece = self.board[r, c]
                if piece == self.EMPTY: continue
                
                x = self.BOARD_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
                y = self.BOARD_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2

                if piece == self.PLAYER_PIECE or piece == self.PLAYER_KING:
                    pygame.gfxdraw.aacircle(self.screen, x, y, radius, self.COLOR_PLAYER)
                    pygame.gfxdraw.filled_circle(self.screen, x, y, radius, self.COLOR_PLAYER)
                elif piece == self.OPPONENT_PIECE or piece == self.OPPONENT_KING:
                    rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
                    pygame.draw.rect(self.screen, self.COLOR_OPPONENT, rect, border_radius=3)
                
                if piece == self.PLAYER_KING or piece == self.OPPONENT_KING:
                    pygame.gfxdraw.aacircle(self.screen, x, y, radius // 2, self.COLOR_KING_ACCENT)
                    pygame.gfxdraw.filled_circle(self.screen, x, y, radius // 2, self.COLOR_KING_ACCENT)

    def _render_ui(self):
        # Player pieces count
        player_text = self.font_large.render(f"{self.player_pieces}", True, self.COLOR_PLAYER)
        self.screen.blit(player_text, (15, 10))
        
        # Opponent pieces count
        opponent_text = self.font_large.render(f"{self.opponent_pieces}", True, self.COLOR_OPPONENT)
        self.screen.blit(opponent_text, (self.WIDTH - 15 - opponent_text.get_width(), 10))
        
        # Turn/Game Over message
        msg = ""
        if self.game_over:
            if self._check_win_condition(): msg = "YOU WIN!"
            elif self._check_loss_condition(): msg = "YOU LOSE"
            else: msg = "GAME OVER"
        elif self.forced_jump_pos:
            msg = "Forced Jump!"
        
        if msg:
            status_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            self.screen.blit(status_text, (self.WIDTH // 2 - status_text.get_width() // 2, self.HEIGHT - 30))

    def _spawn_particles(self, c, r):
        x = self.BOARD_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.BOARD_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': [x, y], 'vel': vel, 'life': life, 'max_life': life, 'color': self.COLOR_OPPONENT})

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / p['max_life']))
                color = (*p['color'], alpha)
                size = int(8 * (p['life'] / p['max_life']))
                if size > 0:
                    s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                    pygame.draw.circle(s, color, (size, size), size)
                    self.screen.blit(s, (int(p['pos'][0]-size), int(p['pos'][1]-size)))

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Checkers RL Environment")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)

        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            # Wait for a moment before reset
            pygame.time.wait(2000)
            obs, info = env.reset()
        
        clock.tick(10) # Run at 10 FPS for human playability

    env.close()