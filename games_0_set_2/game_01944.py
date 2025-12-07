import os
import os
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the falling block. Press Space to drop it instantly."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Stack falling blocks as high as you can. A block must be fully supported by the one below it, or the stack will topple!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_HEIGHT = 20
    MAX_STEPS = 1000

    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 45, 50)
    COLOR_TARGET_LINE = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_GAMEOVER_TEXT = (255, 80, 80)
    COLOR_WIN_TEXT = (80, 255, 80)
    
    BLOCK_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]

    BLOCK_HEIGHT = 15
    MIN_BLOCK_WIDTH = 50
    MAX_BLOCK_WIDTH = 120
    BASE_BLOCK_WIDTH = 200
    
    INITIAL_FALL_SPEED = 2.0
    FALL_SPEED_INCREMENT = 0.05
    PLAYER_MOVE_SPEED = 8

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_gameover = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.stacked_blocks = []
        self.falling_block = None
        self.falling_block_color = None
        self.fall_speed = self.INITIAL_FALL_SPEED
        
        self.particles = []
        self.last_space_held = False
        
        # self.reset() is called by the environment runner, no need to call it here.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.particles = []
        self.last_space_held = False

        # Create the base block
        base_block_rect = pygame.Rect(
            (self.SCREEN_WIDTH - self.BASE_BLOCK_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.BLOCK_HEIGHT,
            self.BASE_BLOCK_WIDTH,
            self.BLOCK_HEIGHT
        )
        self.stacked_blocks = [(base_block_rect, (100, 100, 110))]
        
        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        reward = -0.02 # Small penalty for time passing
        
        # --- Handle Input ---
        if self.falling_block:
            if movement == 3:  # Left
                self.falling_block.x -= self.PLAYER_MOVE_SPEED
            elif movement == 4: # Right
                self.falling_block.x += self.PLAYER_MOVE_SPEED
            
            # Clamp block to screen bounds
            self.falling_block.x = max(0, min(self.SCREEN_WIDTH - self.falling_block.width, self.falling_block.x))

        # Check for drop action (on rising edge of space press)
        drop_block = space_held and not self.last_space_held
        self.last_space_held = space_held

        # --- Update Game Logic ---
        self._update_particles()

        if self.falling_block:
            if drop_block:
                # Move block down until it collides or goes off-screen
                while not self._check_collision() and self.falling_block.top < self.SCREEN_HEIGHT:
                    self.falling_block.y += 1
            else:
                self.falling_block.y += self.fall_speed

        # --- Collision and Stacking ---
        if self.falling_block and self._check_collision():
            support_block_rect, _ = self.stacked_blocks[-1]
            
            # Check for stability
            is_stable = (self.falling_block.left >= support_block_rect.left and 
                         self.falling_block.right <= support_block_rect.right)
            
            if is_stable:
                # sfx: place_block_good
                reward = 0.1
                self.stacked_blocks.append((self.falling_block, self.falling_block_color))
                self.score = len(self.stacked_blocks) - 1 # -1 for the base

                self._create_particles(self.falling_block.midbottom, self.falling_block_color, 20, is_stable=True)

                if self.score >= self.TARGET_HEIGHT:
                    self.win = True
                    self.game_over = True
                    reward = 10.0
                else:
                    self._spawn_new_block()
                    if self.score > 0 and self.score % 5 == 0:
                        self.fall_speed += self.FALL_SPEED_INCREMENT
            else:
                # sfx: place_block_bad
                reward = -10.0
                self.game_over = True
                self._create_particles(self.falling_block.midbottom, (150, 150, 150), 40, is_stable=False)
                self.falling_block = None

        # Check if block fell off the bottom
        if self.falling_block and self.falling_block.top > self.SCREEN_HEIGHT:
            reward = -10.0
            self.game_over = True
            self.falling_block = None

        self.steps += 1
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _spawn_new_block(self):
        width = self.np_random.integers(self.MIN_BLOCK_WIDTH, self.MAX_BLOCK_WIDTH + 1)
        x_pos = self.np_random.integers(0, self.SCREEN_WIDTH - width + 1)
        
        self.falling_block = pygame.Rect(x_pos, -self.BLOCK_HEIGHT, width, self.BLOCK_HEIGHT)
        color_idx = self.np_random.integers(0, len(self.BLOCK_COLORS))
        self.falling_block_color = self.BLOCK_COLORS[color_idx]

    def _check_collision(self):
        if not self.stacked_blocks or not self.falling_block:
            return False
        top_block_rect, _ = self.stacked_blocks[-1]
        return self.falling_block.colliderect(top_block_rect)

    def _create_particles(self, pos, color, count, is_stable):
        for _ in range(count):
            if is_stable:
                vel = [self.np_random.uniform(-1, 1), self.np_random.uniform(-2, 0)]
                life = self.np_random.integers(15, 30)
            else: # Fall effect
                vel = [self.np_random.uniform(-3, 3), self.np_random.uniform(1, 4)]
                life = self.np_random.integers(30, 60)
            
            particle = {
                "pos": list(pos),
                "vel": vel,
                "life": life,
                "color": color,
                "size": self.np_random.uniform(2, 5)
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            p["size"] -= 0.1
        self.particles = [p for p in self.particles if p["life"] > 0 and p["size"] > 0]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": self.score,
            "fall_speed": self.fall_speed,
        }

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw target height line
        target_y = self.SCREEN_HEIGHT - (self.TARGET_HEIGHT + 1) * self.BLOCK_HEIGHT
        if target_y > 0:
            pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (0, target_y), (self.SCREEN_WIDTH, target_y), 1)

        # Draw stacked blocks
        for block_rect, color in self.stacked_blocks:
            self._draw_block(block_rect, color)
            
        # Draw falling block and ghost
        if self.falling_block and not self.game_over:
            # Ghost block
            ghost_block = self.falling_block.copy()
            support_block, _ = self.stacked_blocks[-1]
            ghost_block.top = support_block.top - self.BLOCK_HEIGHT
            
            ghost_color = self.falling_block_color[:3] + (50,) # Add alpha
            s = pygame.Surface(ghost_block.size, pygame.SRCALPHA)
            s.fill(ghost_color)
            self.screen.blit(s, ghost_block.topleft)

            # Falling block
            self._draw_block(self.falling_block, self.falling_block_color)

        # Draw particles
        for p in self.particles:
            pygame.draw.rect(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1]), int(p["size"]), int(p["size"])))

    def _draw_block(self, rect, color):
        darker_color = tuple(max(0, c - 40) for c in color)
        pygame.draw.rect(self.screen, darker_color, rect)
        inner_rect = rect.inflate(-4, -4)
        pygame.draw.rect(self.screen, color, inner_rect)

    def _render_ui(self):
        # Render score
        score_text = self.font_ui.render(f"Height: {self.score}/{self.TARGET_HEIGHT}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Render game over/win message
        if self.game_over:
            if self.win:
                message = "YOU WIN!"
                color = self.COLOR_WIN_TEXT
            else:
                message = "GAME OVER"
                color = self.COLOR_GAMEOVER_TEXT
            
            gameover_surf = self.font_gameover.render(message, True, color)
            gameover_rect = gameover_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(gameover_surf, gameover_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It is NOT part of the Gym environment
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display for direct play
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a separate pygame screen for rendering the observations
    pygame.display.set_caption("Arcade Stacker")
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    done = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default no-op action
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Keyboard input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        # Step the environment
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        # Render the observation to the display screen
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if done:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            done = False
            total_reward = 0

        # Control the frame rate
        env.clock.tick(30)

    env.close()