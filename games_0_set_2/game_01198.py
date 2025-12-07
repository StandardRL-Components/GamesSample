
# Generated: 2025-08-27T16:21:05.689427
# Source Brief: brief_01198.md
# Brief Index: 1198

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move the selector. Press space to swap the selected monster with the one "
        "in the direction of your last move. Hold shift to restart the level."
    )

    game_description = (
        "Match colorful monsters in a grid to clear the background jellies and progress through "
        "increasingly challenging stages with limited moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 8, 8
        self.NUM_MONSTER_TYPES = 5
        self.MAX_PARTICLES = 200
        self.MAX_STEPS = 1000

        self.GRID_WIDTH = 280
        self.GRID_HEIGHT = 280
        self.CELL_SIZE = self.GRID_WIDTH // self.GRID_COLS
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2 + 20

        # --- Colors ---
        self.COLOR_BG = (15, 25, 40)
        self.COLOR_GRID = (30, 45, 65)
        self.COLOR_JELLY = (70, 130, 180, 100) # RGBA for alpha
        self.COLOR_JELLY_CLEARED = (40, 60, 85)
        self.COLOR_SELECTOR = (255, 255, 0)
        self.COLOR_TEXT = (240, 240, 255)
        self.MONSTER_COLORS = [
            (220, 50, 50),   # Red
            (50, 220, 50),   # Green
            (50, 120, 220),  # Blue
            (230, 230, 50),  # Yellow
            (180, 50, 220),  # Purple
            (240, 140, 40),  # Orange
        ]

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        self.font_huge = pygame.font.Font(None, 80)
        
        # --- State Variables ---
        self.grid = None
        self.jelly_grid = None
        self.selector_pos = None
        self.last_move_dir = None
        self.total_score = 0
        self.stage = 1
        self.moves_left = 0
        self.steps = 0
        self.game_over = False
        self.game_won_stage = False
        self.particles = deque()
        self.show_message_timer = 0
        self.message_text = ""
        self.message_color = (0,0,0)

        self.np_random = None

        # --- Final Initialization ---
        self.reset()
        # self.validate_implementation() # Uncomment for self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            # Create a new generator if one doesn't exist or if a new seed is not provided
            if self.np_random is None:
                 self.np_random = np.random.default_rng()

        if not self.game_won_stage:
            self.stage = 1
            self.total_score = 0

        self.moves_left = max(10, 32 - self.stage * 2)
        self.steps = 0
        self.game_over = False
        self.game_won_stage = False
        self.selector_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.last_move_dir = [0, 0]
        self.particles.clear()

        self._generate_board()
        self.show_message(f"Stage {self.stage}", self.COLOR_TEXT, 60)

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        
        self.steps += 1
        if self.show_message_timer > 0:
            self.show_message_timer -= 1

        if shift_held:
            # Restart action: ends episode with a penalty
            reward = -100
            terminated = True
            self.game_over = True
            self.show_message("Restarting", (255,100,100), 30)
            return self._get_observation(), reward, terminated, False, self._get_info()

        # --- Handle Movement ---
        move_map = {1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1]}
        if movement in move_map:
            move_dir = move_map[movement]
            self.selector_pos[0] = (self.selector_pos[0] + move_dir[0]) % self.GRID_ROWS
            self.selector_pos[1] = (self.selector_pos[1] + move_dir[1]) % self.GRID_COLS
            self.last_move_dir = move_dir
            # Small penalty for just moving the cursor
            reward -= 0.01

        # --- Handle Swap Action ---
        if space_held and self.last_move_dir != [0, 0]:
            reward += self._process_swap()
            self.moves_left -= 1
            # sfx: swap_attempt.wav

        # --- Check Termination Conditions ---
        if np.sum(self.jelly_grid) == 0:
            # Stage cleared
            reward += 100  # Large reward for clearing stage
            self.stage += 1
            self.game_won_stage = True
            terminated = True # End episode, reset will handle stage progression
            self.show_message("Stage Clear!", (100,255,100), 60)
            # sfx: stage_clear.wav
        elif self.moves_left <= 0:
            # Game over
            reward += -100 # Large penalty for running out of moves
            terminated = True
            self.game_over = True
            self.game_won_stage = False
            self.show_message("Game Over", (255,100,100), 60)
            # sfx: game_over.wav
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _process_swap(self):
        r1, c1 = self.selector_pos
        r2 = (r1 + self.last_move_dir[0]) % self.GRID_ROWS
        c2 = (c1 + self.last_move_dir[1]) % self.GRID_COLS
        
        # Perform swap
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

        total_cleared = 0
        chain_multiplier = 1.0
        
        while True:
            matches = self._find_matches()
            if not matches:
                # If no match on the first attempt, swap back
                if total_cleared == 0:
                    self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
                    # sfx: invalid_swap.wav
                    return -0.1  # Penalty for invalid swap
                break
            
            # sfx: match_found.wav
            num_cleared = len(matches)
            total_cleared += num_cleared
            self.total_score += int(num_cleared * chain_multiplier)
            
            # Create particles and clear jelly
            for r, c in matches:
                if self.jelly_grid[r, c] == 1:
                    self.jelly_grid[r, c] = 0
                    # sfx: jelly_clear.wav
                self._create_particles(r, c, self.grid[r, c])

            self._clear_and_refill(matches)
            chain_multiplier += 0.5 # Increase multiplier for cascades
        
        # Reward is 1 per monster cleared
        return total_cleared

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                monster_type = self.grid[r, c]
                if monster_type == -1: continue

                # Horizontal match
                if c < self.GRID_COLS - 2 and self.grid[r, c+1] == monster_type and self.grid[r, c+2] == monster_type:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                
                # Vertical match
                if r < self.GRID_ROWS - 2 and self.grid[r+1, c] == monster_type and self.grid[r+2, c] == monster_type:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _clear_and_refill(self, matches):
        # Set matched cells to -1 (empty)
        for r, c in matches:
            self.grid[r, c] = -1

        # Gravity: drop monsters down
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != -1:
                    self.grid[empty_row, c], self.grid[r, c] = self.grid[r, c], self.grid[empty_row, c]
                    empty_row -= 1
        
        # Refill top rows with new monsters
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_MONSTER_TYPES)

    def _generate_board(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_MONSTER_TYPES, size=(self.GRID_ROWS, self.GRID_COLS))
            if not self._find_matches() and self._is_board_valid():
                break
        self.jelly_grid = np.ones((self.GRID_ROWS, self.GRID_COLS), dtype=np.int8)

    def _is_board_valid(self):
        # Check for any possible moves
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Test swap right
                if c < self.GRID_COLS - 1:
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                    if self._find_matches():
                        self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                # Test swap down
                if r < self.GRID_ROWS - 1:
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                    if self._find_matches():
                        self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
        return False

    def _create_particles(self, r, c, monster_type):
        px = self.GRID_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        color = self.MONSTER_COLORS[monster_type]
        for _ in range(10):
            if len(self.particles) > self.MAX_PARTICLES: break
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(15, 30)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'color': color, 'radius': random.uniform(2, 5)})

    def show_message(self, text, color, duration):
        self.message_text = text
        self.message_color = color
        self.show_message_timer = duration

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid and jellies
        jelly_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                rect = pygame.Rect(self.GRID_X + c * self.CELL_SIZE, self.GRID_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                if self.jelly_grid[r, c]:
                    jelly_surface.fill(self.COLOR_JELLY)
                    self.screen.blit(jelly_surface, rect.topleft)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_JELLY_CLEARED, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Render monsters
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                monster_type = self.grid[r, c]
                if monster_type != -1:
                    rect = pygame.Rect(self.GRID_X + c * self.CELL_SIZE, self.GRID_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                    self._draw_monster(self.screen, monster_type, rect)
        
        self._render_particles()
        self._render_selector()

    def _draw_monster(self, surface, monster_type, rect):
        padding = 5
        center_x, center_y = rect.center
        radius = (self.CELL_SIZE - padding * 2) // 2
        color = self.MONSTER_COLORS[monster_type]

        # Body
        if monster_type == 0: # Circle
            pygame.gfxdraw.filled_circle(surface, center_x, center_y, radius, color)
            pygame.gfxdraw.aacircle(surface, center_x, center_y, radius, (0,0,0,50))
        elif monster_type == 1: # Square
            inner_rect = pygame.Rect(rect.left + padding, rect.top + padding, self.CELL_SIZE - padding * 2, self.CELL_SIZE - padding * 2)
            pygame.draw.rect(surface, color, inner_rect, border_radius=4)
        elif monster_type == 2: # Triangle
            points = [
                (center_x, rect.top + padding),
                (rect.left + padding, rect.bottom - padding),
                (rect.right - padding, rect.bottom - padding)
            ]
            pygame.gfxdraw.filled_polygon(surface, points, color)
            pygame.gfxdraw.aapolygon(surface, points, (0,0,0,50))
        elif monster_type == 3: # Hexagon
            points = []
            for i in range(6):
                angle = math.pi / 3 * i
                points.append((center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)))
            pygame.gfxdraw.filled_polygon(surface, points, color)
            pygame.gfxdraw.aapolygon(surface, points, (0,0,0,50))
        elif monster_type == 4: # Star
            points = []
            for i in range(10):
                r = radius if i % 2 == 0 else radius / 2
                angle = math.pi / 5 * i - math.pi / 2
                points.append((center_x + r * math.cos(angle), center_y + r * math.sin(angle)))
            pygame.gfxdraw.filled_polygon(surface, points, color)
            pygame.gfxdraw.aapolygon(surface, points, (0,0,0,50))
        
        # Eyes
        eye_radius = max(2, int(radius * 0.15))
        eye_offset = int(radius * 0.4)
        pygame.draw.circle(surface, (255, 255, 255), (center_x - eye_offset, center_y - eye_offset//2), eye_radius + 1)
        pygame.draw.circle(surface, (255, 255, 255), (center_x + eye_offset, center_y - eye_offset//2), eye_radius + 1)
        pygame.draw.circle(surface, (0, 0, 0), (center_x - eye_offset, center_y - eye_offset//2), eye_radius-1)
        pygame.draw.circle(surface, (0, 0, 0), (center_x + eye_offset, center_y - eye_offset//2), eye_radius-1)

    def _render_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= 0.1
            if p['life'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p['life'] / 20))))
                color = (*p['color'], alpha)
                temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
                self.screen.blit(temp_surf, (p['pos'][0] - p['radius'], p['pos'][1] - p['radius']), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_selector(self):
        if self.game_over: return
        r, c = self.selector_pos
        rect = pygame.Rect(self.GRID_X + c * self.CELL_SIZE, self.GRID_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        
        # Pulsating glow effect
        glow_alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.005)
        glow_color = (*self.COLOR_SELECTOR, glow_alpha)
        
        # Draw multiple lines for a thicker, softer look
        for i in range(3):
            pygame.draw.rect(self.screen, glow_color, rect.inflate(i*2, i*2), 1, border_radius=3)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {self.total_score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Moves
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 20, 20))

        # Stage
        stage_text = self.font_small.render(f"Stage: {self.stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH // 2 - stage_text.get_width() // 2, self.HEIGHT - 30))

        # Center message
        if self.show_message_timer > 0:
            alpha = min(255, self.show_message_timer * 10)
            msg_surf = self.font_huge.render(self.message_text, True, self.message_color)
            msg_surf.set_alpha(alpha)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.total_score,
            "steps": self.steps,
            "stage": self.stage,
            "moves_left": self.moves_left,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# Example usage:
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame window setup for human play ---
    pygame.display.set_caption("Monster Matcher")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    running = True
    while running:
        action = [0, 0, 0] # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Stage: {info['stage']}")
            obs, info = env.reset()

        # --- Draw the observation to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we control the step rate
        env.clock.tick(10) # Limit to 10 actions per second for human play
        
    env.close()