
# Generated: 2025-08-28T03:28:06.569056
# Source Brief: brief_02031.md
# Brief Index: 2031

        
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

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your avatar on the grid. "
        "Collect all yellow gems to advance. Avoid the red traps!"
    )

    # User-facing description of the game
    game_description = (
        "Navigate a procedurally generated grid, collecting gems while avoiding traps "
        "to achieve the highest score. Complete 3 stages to win."
    )

    # Frames advance only on action
    auto_advance = False

    # --- Constants ---
    # Game world
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    CELL_SIZE = 40
    SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE  # 640
    SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE # 400
    MAX_STEPS = 1000
    
    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 60)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_PLAYER_GLOW = (50, 150, 255, 50)
    COLOR_GEM = (255, 220, 50)
    COLOR_TRAP = (255, 50, 50)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_TRAP_FLASH = (255, 0, 0, 100)

    # Game mechanics
    INITIAL_GEMS = 15
    INITIAL_TRAPS = 3
    MAX_STAGES = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.player_pos = [0, 0]
        self.gems = []
        self.traps = []
        self.particles = []
        self.flash_timer = 0
        
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def _setup_stage(self):
        """Generates a new layout for gems and traps for the current stage."""
        self.particles = []
        
        num_traps = self.INITIAL_TRAPS + self.stage - 1
        num_gems = self.INITIAL_GEMS

        # Generate all possible grid cells
        all_cells = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_cells)

        # Place player in a relatively central position
        self.player_pos = list(all_cells.pop(len(all_cells) // 2))

        # Place traps and gems, ensuring no overlap
        self.traps = [all_cells.pop() for _ in range(num_traps)]
        self.gems = [all_cells.pop() for _ in range(num_gems)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.flash_timer = 0
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        terminated = False
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        self.steps += 1
        
        # Calculate distance to nearest gem before moving
        dist_before = self._get_dist_to_nearest_gem()

        # 1. Handle player movement
        prev_pos = list(self.player_pos)
        if movement == 1:  # Up
            self.player_pos[1] -= 1
        elif movement == 2:  # Down
            self.player_pos[1] += 1
        elif movement == 3:  # Left
            self.player_pos[0] -= 1
        elif movement == 4:  # Right
            self.player_pos[0] += 1

        # Clamp position to grid boundaries
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_WIDTH - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_HEIGHT - 1)
        
        moved = prev_pos != self.player_pos

        # 2. Check for game events
        player_pos_tuple = tuple(self.player_pos)

        # Check for gem collection
        if player_pos_tuple in self.gems:
            self.gems.remove(player_pos_tuple)
            reward += 10
            self.score += 10
            self._create_particles(self.player_pos, self.COLOR_GEM)
            # Sound: // sfx_gem_collect.wav

        # Check for trap collision
        elif player_pos_tuple in self.traps:
            reward -= 100
            self.score -= 100
            terminated = True
            self.game_over = True
            self.flash_timer = 5 # Flash for 5 frames
            # Sound: // sfx_trap_hit.wav

        # 3. Calculate continuous reward
        else:
            dist_after = self._get_dist_to_nearest_gem()
            if moved:
                if dist_after < dist_before:
                    reward += 1  # Moved closer to a gem
                else:
                    reward -= 0.1 # Moved away or parallel
            else: # No-op or bumped into wall
                reward -= 0.1 

        # 4. Check for stage/game completion
        if not self.gems: # All gems collected
            if self.stage < self.MAX_STAGES:
                self.stage += 1
                self._setup_stage()
                # Sound: // sfx_stage_clear.wav
            else: # All stages completed
                reward += 100
                self.score += 100
                terminated = True
                self.game_over = True
                # Sound: // sfx_game_win.wav

        # 5. Check for max steps
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        # Update particle animations
        self._update_particles()
        if self.flash_timer > 0:
            self.flash_timer -= 1

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_dist_to_nearest_gem(self):
        if not self.gems:
            return 0
        
        player_x, player_y = self.player_pos
        min_dist = float('inf')
        for gem_x, gem_y in self.gems:
            dist = abs(player_x - gem_x) + abs(player_y - gem_y) # Manhattan distance
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _create_particles(self, grid_pos, color):
        center_x = (grid_pos[0] + 0.5) * self.CELL_SIZE
        center_y = (grid_pos[1] + 0.5) * self.CELL_SIZE
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(10, 20)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['vel'][1] += 0.1 # Gravity

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py))

        # Draw traps
        for tx, ty in self.traps:
            rect = pygame.Rect(tx * self.CELL_SIZE, ty * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_TRAP, rect.inflate(-8, -8))

        # Draw gems
        for gx, gy in self.gems:
            center_x = int((gx + 0.5) * self.CELL_SIZE)
            center_y = int((gy + 0.5) * self.CELL_SIZE)
            radius = int(self.CELL_SIZE * 0.3)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_GEM)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_GEM)

        # Draw player
        player_rect = pygame.Rect(
            self.player_pos[0] * self.CELL_SIZE,
            self.player_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        # Glow effect
        glow_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, player_rect.topleft)
        # Main player rect
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-12, -12), border_radius=4)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20.0))))
            color = (*p['color'], alpha)
            pygame.draw.circle(self.screen, color, (int(p['pos'][0]), int(p['pos'][1])), 2)

    def _render_ui(self):
        # Score display
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Stage display
        stage_text = self.font_ui.render(f"Stage: {self.stage} / {self.MAX_STAGES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (self.SCREEN_WIDTH - stage_text.get_width() - 10, 10))
        
        # Trap hit flash
        if self.flash_timer > 0:
            flash_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            flash_surface.fill(self.COLOR_TRAP_FLASH)
            self.screen.blit(flash_surface, (0, 0))

        # Game Over message
        if self.game_over:
            msg = "YOU WIN!" if not self.gems else "GAME OVER"
            color = self.COLOR_GEM if not self.gems else self.COLOR_TRAP
            
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_game_over.render(msg, True, color)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "gems_left": len(self.gems),
            "player_pos": self.player_pos,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        assert info['gems_left'] == self.INITIAL_GEMS
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Example ---
    # This part requires a display and is for human testing.
    # To run, you need to change render_mode to "human" and add display setup.
    # For this headless implementation, we just verify it runs.
    
    print("Starting environment test run...")
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    
    for i in range(200):
        action = env.action_space.sample() # Take a random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if (i + 1) % 50 == 0:
            print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Info={info}")

        if terminated:
            print(f"Episode terminated after {i+1} steps. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

    env.close()
    print("Environment test run completed.")