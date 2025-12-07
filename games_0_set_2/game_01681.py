
# Generated: 2025-08-28T02:21:06.781848
# Source Brief: brief_01681.md
# Brief Index: 1681

        
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
    """
    Gymnasium environment for a fast-paced arcade puzzle game.
    The player must navigate a grid to collect all targets before time runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your square around the grid. "
        "Collect all the green targets before the timer runs out to win."
    )

    # Short, user-facing description of the game
    game_description = (
        "A fast-paced arcade puzzle game. Navigate a grid to collect all targets before time runs out. "
        "Each move costs time, so find the most efficient path to victory!"
    )

    # Frames only advance when an action is received.
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 16, 10
        self.CELL_SIZE = 40
        self.MAX_TIME = 3600  # 60 seconds at 60 steps/sec
        self.NUM_TARGETS = 25

        # EXACT spaces as per specification
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.Font(None, 32)
        self.end_font = pygame.font.Font(None, 72)

        # Visual style
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 255, 150, 100)
        self.COLOR_TARGET = (100, 255, 100)
        self.COLOR_TARGET_OUTLINE = (180, 255, 180)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TIMER_LOW = (255, 80, 80)
        
        # Game state variables
        self.player_pos = None
        self.targets = None
        self.score = 0
        self.time_left = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        
        # Initialize state
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_TIME
        self.particles.clear()
        
        # Generate all possible grid positions and shuffle them
        all_pos = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_pos)
        
        # Place player and targets from the shuffled list
        self.player_pos = all_pos.pop()
        self.targets = {all_pos[i] for i in range(self.NUM_TARGETS)}
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # If the game is over, do not process any more actions
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Update game logic
        self.steps += 1
        self.time_left -= 1
        reward = -0.01  # Small penalty for each step to encourage efficiency

        # --- Player Movement ---
        px, py = self.player_pos
        if movement == 1: py -= 1  # Up
        elif movement == 2: py += 1  # Down
        elif movement == 3: px -= 1  # Left
        elif movement == 4: px += 1  # Right
        
        # Clamp player position to grid boundaries
        px = max(0, min(self.GRID_WIDTH - 1, px))
        py = max(0, min(self.GRID_HEIGHT - 1, py))
        self.player_pos = (px, py)

        # --- Target Collection ---
        if self.player_pos in self.targets:
            self.targets.remove(self.player_pos)
            self.score += 1
            reward += 1.0
            self._create_particles(self.player_pos)
            # SFX: play_collect_sound()

        # --- Termination Check ---
        terminated = False
        if not self.targets:  # Win condition: all targets collected
            terminated = True
            self.game_over = True
            reward = 100.0  # Large positive reward for winning
        elif self.time_left <= 0:  # Lose condition: time runs out
            terminated = True
            self.game_over = True
            reward = -100.0 # Large negative reward for losing
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background color
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (W, H, C) -> (H, W, C)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT), 1)
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py), 1)
            
        # Draw targets
        for pos in self.targets:
            cx = int((pos[0] + 0.5) * self.CELL_SIZE)
            cy = int((pos[1] + 0.5) * self.CELL_SIZE)
            radius = int(self.CELL_SIZE * 0.3)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, self.COLOR_TARGET)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, self.COLOR_TARGET_OUTLINE)

        # Update and draw particles
        self._update_and_draw_particles()
        
        # Draw player
        px, py = self.player_pos
        player_rect = pygame.Rect(
            px * self.CELL_SIZE, py * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE
        )
        
        # Player glow effect
        glow_surface = pygame.Surface((self.CELL_SIZE * 2, self.CELL_SIZE * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.COLOR_PLAYER_GLOW, (self.CELL_SIZE, self.CELL_SIZE), self.CELL_SIZE * 0.7)
        self.screen.blit(glow_surface, (player_rect.centerx - self.CELL_SIZE, player_rect.centery - self.CELL_SIZE), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Player main body and border
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        pygame.draw.rect(self.screen, (255, 255, 200), player_rect, 2)

    def _render_ui(self):
        # Render score
        score_text = self.ui_font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))
        
        # Render timer
        time_sec = self.time_left / 60.0
        timer_color = self.COLOR_TEXT
        if time_sec < 10:
            # Flashing effect for low time
            if int(time_sec * 2) % 2 == 0:
                timer_color = self.COLOR_TIMER_LOW
        
        timer_text = self.ui_font.render(f"TIME: {time_sec:.1f}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 10))
        self.screen.blit(timer_text, timer_rect)

        # Render game over/win message
        if self.game_over:
            msg = "YOU WIN!" if not self.targets else "TIME UP!"
            color = self.COLOR_TARGET if not self.targets else self.COLOR_TIMER_LOW
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg_surf = self.end_font.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "targets_left": len(self.targets)
        }
        
    def _create_particles(self, grid_pos):
        cx = (grid_pos[0] + 0.5) * self.CELL_SIZE
        cy = (grid_pos[1] + 0.5) * self.CELL_SIZE
        for _ in range(25):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(20, 40)
            color = random.choice([self.COLOR_TARGET, (255, 255, 255), (150, 255, 150)])
            self.particles.append([cx, cy, vx, vy, lifetime, color])

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p[0] += p[2]  # x += vx
            p[1] += p[3]  # y += vy
            p[4] -= 1     # lifetime -= 1
            if p[4] > 0:
                alpha = max(0, (p[4] / 40.0) * 255)
                radius = 2
                pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), radius, (*p[5], int(alpha)))
                active_particles.append(p)
        self.particles = active_particles

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")