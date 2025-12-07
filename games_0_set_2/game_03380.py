
# Generated: 2025-08-27T23:10:53.425249
# Source Brief: brief_03380.md
# Brief Index: 3380

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: Use arrow keys to move the selector. Press space to select a gem, then "
        "move the selector to an adjacent gem and press space again to swap. Press shift to cancel a selection."
    )

    game_description = (
        "Swap adjacent gems to create matches of 3 or more in this fast-paced isometric puzzle game. "
        "Chain reactions create score multipliers. Reach 1000 points before you run out of 20 moves!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.NUM_GEM_TYPES = 5
        self.WIN_SCORE = 1000
        self.MAX_MOVES = 20
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 60, 80)
        self.GEM_COLORS = [
            (255, 80, 80),    # Red
            (80, 255, 80),    # Green
            (80, 150, 255),   # Blue
            (255, 255, 80),   # Yellow
            (200, 80, 255),   # Purple
        ]
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_HIGHLIGHT = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 230)

        # --- Isometric Projection ---
        self.TILE_W, self.TILE_H = 48, 24
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80

        # --- Animation Timings ---
        self.ANIM_TIME_SWAP = 6
        self.ANIM_TIME_CLEAR = 8
        self.ANIM_TIME_FALL = 5

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_score = pygame.font.SysFont("Consolas", 32, bold=True)

        # --- Initialize State ---
        self.board = None
        self.cursor_pos = None
        self.selected_gem = None
        self.game_state = None
        self.animation_timer = 0
        self.animation_data = {}
        self.particles = []
        self.steps = 0
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.last_match_size = 0
        self.chain_multiplier = 1
        self.current_reward = 0

        self.reset()
        self.validate_implementation()

    def _grid_to_iso(self, r, c):
        x = self.ORIGIN_X + (c - r) * (self.TILE_W / 2)
        y = self.ORIGIN_Y + (c + r) * (self.TILE_H / 2)
        return int(x), int(y)

    def _generate_board(self):
        """Generates a board with no initial matches and at least one possible move."""
        while True:
            board = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
            if not self._find_all_matches(board) and self._find_possible_moves(board):
                return board

    def _find_all_matches(self, board):
        """Finds all horizontal and vertical matches of 3 or more."""
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if board[r, c] == 0: continue
                # Horizontal
                if c < self.GRID_SIZE - 2 and board[r, c] == board[r, c + 1] == board[r, c + 2]:
                    matches.update([(r, c), (r, c + 1), (r, c + 2)])
                # Vertical
                if r < self.GRID_SIZE - 2 and board[r, c] == board[r + 1, c] == board[r + 2, c]:
                    matches.update([(r, c), (r + 1, c), (r + 2, c)])
        return list(matches)

    def _find_possible_moves(self, board):
        """Checks if any valid moves exist on the board."""
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                # Try swapping right
                if c < self.GRID_SIZE - 1:
                    board[r, c], board[r, c + 1] = board[r, c + 1], board[r, c]
                    if self._find_all_matches(board):
                        board[r, c], board[r, c + 1] = board[r, c + 1], board[r, c]
                        return True
                    board[r, c], board[r, c + 1] = board[r, c + 1], board[r, c]
                # Try swapping down
                if r < self.GRID_SIZE - 1:
                    board[r, c], board[r + 1, c] = board[r + 1, c], board[r, c]
                    if self._find_all_matches(board):
                        board[r, c], board[r + 1, c] = board[r + 1, c], board[r, c]
                        return True
                    board[r, c], board[r + 1, c] = board[r + 1, c], board[r, c]
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = self._generate_board()
        self.cursor_pos = (0, 0)
        self.selected_gem = None
        self.game_state = 'IDLE'
        self.animation_timer = 0
        self.animation_data = {}
        self.particles = []
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.chain_multiplier = 1
        self.current_reward = 0
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        self.current_reward = 0
        
        # --- Handle game states and animations ---
        if self.game_state != 'IDLE':
            self._update_animation()
        else:
            # Only process actions when idle
            self._handle_action(action)

        # --- Termination conditions ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            self.current_reward += 100
            terminated = True
        elif self.moves_left <= 0 and self.game_state == 'IDLE':
            self.current_reward -= 10
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        # If no moves are possible on an idle board, end the game
        if self.game_state == 'IDLE' and not self._find_possible_moves(self.board):
             self.current_reward -= 10 # No more moves is a loss
             terminated = True

        self.game_over = terminated
        reward = self.current_reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        r, c = self.cursor_pos

        if shift_pressed:
            self.selected_gem = None
            # sound: cancel_select.wav
            return

        # Move cursor
        if movement == 1: self.cursor_pos = (max(0, r - 1), c)  # Up
        elif movement == 2: self.cursor_pos = (min(self.GRID_SIZE - 1, r + 1), c)  # Down
        elif movement == 3: self.cursor_pos = (r, max(0, c - 1))  # Left
        elif movement == 4: self.cursor_pos = (r, min(self.GRID_SIZE - 1, c + 1))  # Right
        
        if space_pressed:
            if self.selected_gem is None:
                self.selected_gem = self.cursor_pos
                # sound: select_gem.wav
            else:
                # Attempt swap if adjacent
                sr, sc = self.selected_gem
                cr, cc = self.cursor_pos
                if abs(sr - cr) + abs(sc - cc) == 1:
                    self._initiate_swap(self.selected_gem, self.cursor_pos)
                else: # Not adjacent, so just re-select
                    self.selected_gem = self.cursor_pos
                    # sound: invalid_selection.wav

    def _initiate_swap(self, pos1, pos2):
        self.moves_left -= 1
        self.game_state = 'SWAPPING'
        self.animation_timer = self.ANIM_TIME_SWAP
        self.animation_data = {'pos1': pos1, 'pos2': pos2, 'is_valid': False}
        self.selected_gem = None
        # sound: swap_start.wav

    def _update_animation(self):
        self.animation_timer -= 1
        if self.animation_timer > 0:
            return

        # --- SWAPPING complete ---
        if self.game_state == 'SWAPPING':
            pos1, pos2 = self.animation_data['pos1'], self.animation_data['pos2']
            r1, c1 = pos1
            r2, c2 = pos2
            
            # Perform swap on board
            self.board[r1, c1], self.board[r2, c2] = self.board[r2, c2], self.board[r1, c1]
            
            matches = self._find_all_matches(self.board)
            if matches:
                self.animation_data['matches'] = matches
                self.game_state = 'CLEARING'
                self.animation_timer = self.ANIM_TIME_CLEAR
                # sound: match_found.wav
            else: # Invalid swap, swap back
                self.game_state = 'SWAP_BACK'
                self.animation_timer = self.ANIM_TIME_SWAP
                self.current_reward -= 0.1
                # sound: invalid_swap.wav

        # --- SWAP_BACK complete ---
        elif self.game_state == 'SWAP_BACK':
            pos1, pos2 = self.animation_data['pos1'], self.animation_data['pos2']
            r1, c1 = pos1
            r2, c2 = pos2
            self.board[r1, c1], self.board[r2, c2] = self.board[r2, c2], self.board[r1, c1] # Swap back
            self.game_state = 'IDLE'
            self.chain_multiplier = 1

        # --- CLEARING complete ---
        elif self.game_state == 'CLEARING':
            matches = self.animation_data['matches']
            
            # Calculate score and reward
            num_cleared = len(matches)
            base_score = num_cleared * self.chain_multiplier
            self.current_reward += num_cleared
            if num_cleared == 4:
                base_score += 5 * self.chain_multiplier
                self.current_reward += 5
            elif num_cleared >= 5:
                base_score += 10 * self.chain_multiplier
                self.current_reward += 10
            self.score += base_score

            # Spawn particles and clear board
            for r, c in matches:
                self._spawn_particles(r, c, self.board[r, c])
                self.board[r, c] = 0
            
            # sound: gems_cleared.wav
            
            self.game_state = 'FALLING'
            self._apply_gravity()
            self.animation_timer = self.ANIM_TIME_FALL

        # --- FALLING complete ---
        elif self.game_state == 'FALLING':
            self._refill_board()
            self.chain_multiplier += 1
            matches = self._find_all_matches(self.board)
            if matches: # Cascade!
                self.animation_data['matches'] = matches
                self.game_state = 'CLEARING'
                self.animation_timer = self.ANIM_TIME_CLEAR
                # sound: cascade_trigger.wav
            else:
                self.game_state = 'IDLE'
                self.chain_multiplier = 1

    def _apply_gravity(self):
        """Shifts gems down to fill empty spaces in the data grid."""
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.board[r, c] != 0:
                    self.board[empty_row, c], self.board[r, c] = self.board[r, c], self.board[empty_row, c]
                    empty_row -= 1

    def _refill_board(self):
        """Fills empty spaces at the top with new gems."""
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.board[r, c] == 0:
                    self.board[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
    
    def _spawn_particles(self, r, c, gem_type):
        """Spawns particles for a cleared gem."""
        px, py = self._grid_to_iso(r, c)
        color = self.GEM_COLORS[gem_type - 1]
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = self.np_random.integers(10, 20)
            self.particles.append([px, py, vx, vy, life, color])

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for r in range(self.GRID_SIZE + 1):
            p1 = self._grid_to_iso(r, 0)
            p2 = self._grid_to_iso(r, self.GRID_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
        for c in range(self.GRID_SIZE + 1):
            p1 = self._grid_to_iso(0, c)
            p2 = self._grid_to_iso(self.GRID_SIZE, c)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)

        # Draw gems
        swapping_gems = []
        if self.game_state in ['SWAPPING', 'SWAP_BACK']:
            swapping_gems.extend(self.animation_data.values())

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if (r, c) in swapping_gems: continue
                gem_type = self.board[r, c]
                if gem_type > 0:
                    self._draw_gem(r, c, gem_type)

        # Draw animated gems
        self._render_animations()
        
        # Update and draw particles
        self._render_particles()

        # Draw cursor
        cr, cc = self.cursor_pos
        self._draw_selector(cr, cc, self.COLOR_HIGHLIGHT, 3)

        # Draw selection
        if self.selected_gem:
            sr, sc = self.selected_gem
            self._draw_selector(sr, sc, (255, 255, 0), 2)

    def _draw_gem(self, r, c, gem_type, offset_x=0, offset_y=0, scale=1.0):
        if gem_type == 0: return
        
        center_x, center_y = self._grid_to_iso(r, c)
        center_x += offset_x
        center_y += offset_y

        w = self.TILE_W * 0.8 * scale
        h = self.TILE_H * 1.6 * scale
        
        points = [
            (center_x, center_y - h / 2),
            (center_x + w / 2, center_y),
            (center_x, center_y + h / 2),
            (center_x - w / 2, center_y)
        ]
        points = [(int(p[0]), int(p[1])) for p in points]
        
        color = self.GEM_COLORS[gem_type - 1]
        
        # Draw filled polygon and antialiased outline
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _draw_selector(self, r, c, color, width):
        center_x, center_y = self._grid_to_iso(r, c)
        w, h = self.TILE_W, self.TILE_H
        points = [
            (center_x, center_y - h / 2),
            (center_x + w / 2, center_y),
            (center_x, center_y + h / 2),
            (center_x - w / 2, center_y)
        ]
        pygame.draw.lines(self.screen, color, True, [(int(p[0]), int(p[1])) for p in points], width)

    def _render_animations(self):
        # SWAPPING / SWAP_BACK
        if self.game_state in ['SWAPPING', 'SWAP_BACK']:
            progress = (self.ANIM_TIME_SWAP - self.animation_timer) / self.ANIM_TIME_SWAP
            
            r1, c1 = self.animation_data['pos1']
            r2, c2 = self.animation_data['pos2']

            x1, y1 = self._grid_to_iso(r1, c1)
            x2, y2 = self._grid_to_iso(r2, c2)

            gem_type1 = self.board[r2, c2] if self.game_state == 'SWAPPING' else self.board[r1, c1]
            gem_type2 = self.board[r1, c1] if self.game_state == 'SWAPPING' else self.board[r2, c2]

            self._draw_gem(r1, c1, gem_type1, offset_x=(x2 - x1) * progress, offset_y=(y2 - y1) * progress)
            self._draw_gem(r2, c2, gem_type2, offset_x=(x1 - x2) * progress, offset_y=(y1 - y2) * progress)

        # CLEARING
        elif self.game_state == 'CLEARING':
            progress = (self.ANIM_TIME_CLEAR - self.animation_timer) / self.ANIM_TIME_CLEAR
            scale = 1.0 - progress
            for r, c in self.animation_data['matches']:
                gem_type = self.board[r, c]
                self._draw_gem(r, c, gem_type, scale=max(0, scale))

    def _render_particles(self):
        for p in self.particles[:]:
            p[0] += p[2]  # x += vx
            p[1] += p[3]  # y += vy
            p[4] -= 1     # life -= 1
            if p[4] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p[4] / 20))
                color = p[5]
                temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color + (alpha,), (2, 2), 2)
                self.screen.blit(temp_surf, (int(p[0]) - 2, int(p[1]) - 2))

    def _render_ui(self):
        # Score
        score_text = self.font_score.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Moves
        moves_text = self.font_score.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 10, 10))

        # Chain Multiplier
        if self.chain_multiplier > 1:
            chain_text = self.font_main.render(f"x{self.chain_multiplier} Chain!", True, self.GEM_COLORS[self.chain_multiplier % self.NUM_GEM_TYPES])
            text_rect = chain_text.get_rect(center=(self.WIDTH / 2, 40))
            self.screen.blit(chain_text, text_rect)

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font_score.render(msg, True, self.COLOR_WHITE)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

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
        
        # Test survival of random agent
        self.reset()
        for _ in range(50):
            action = self.action_space.sample()
            _, _, terminated, _, _ = self.step(action)
            if terminated:
                self.reset()
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # no-op, release, release
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    done = False
                    continue
        
        # Only step if an action is taken or an animation is in progress
        if any(action) or env.game_state != 'IDLE':
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if reward != 0:
                print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")

        # Render the environment to a Pygame window
        render_surface = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_surface.blit(draw_surface, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit frame rate

    pygame.quit()