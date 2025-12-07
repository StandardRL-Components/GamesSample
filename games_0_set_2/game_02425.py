
# Generated: 2025-08-28T04:47:37.104857
# Source Brief: brief_02425.md
# Brief Index: 2425

        
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


# Helper classes for game objects
class Block:
    """Represents a single block, either falling or stacked."""
    def __init__(self, x, y, width, height, color, block_type, vel_y=0):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.type = block_type
        self.vel_y = vel_y

class Particle:
    """Represents a single particle for visual effects."""
    def __init__(self, x, y, color, np_random):
        self.pos = [x, y]
        # Use np_random for reproducible randomness
        self.vel = [np_random.uniform(-1.5, 1.5), np_random.uniform(-3.5, -1.0)]
        self.life = np_random.integers(15, 30)
        self.radius = np_random.integers(2, 5)
        self.color = color
        self.gravity = 0.1

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[1] += self.gravity
        self.life -= 1
        return self.life > 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move the block. Press space to drop it faster."
    )

    game_description = (
        "Stack falling blocks as high as possible. Red blocks are heavier and require more careful placement to be stable."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (30, 40, 50)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TARGET_LINE = (255, 200, 0, 150)
    
    COLOR_BLOCK_RED = (230, 50, 50)
    COLOR_BLOCK_BLUE = (50, 150, 230)
    COLOR_BLOCK_RED_STACKED = (180, 40, 40)
    COLOR_BLOCK_BLUE_STACKED = (40, 120, 180)
    COLOR_BLOCK_OUTLINE = (10, 10, 10)
    COLOR_BASE = (80, 80, 90)

    # Game parameters
    BLOCK_WIDTH, BLOCK_HEIGHT = 50, 20
    BASE_WIDTH, BASE_HEIGHT = 200, 20
    BLOCK_FALL_SPEED = 2.5
    BLOCK_DROP_SPEED = 20
    BLOCK_MOVE_SPEED = 6
    MAX_STEPS = 1500
    WIN_HEIGHT = 50

    # Stability thresholds (percentage of block width)
    STABILITY_RED = 0.50  # 50% overlap required
    STABILITY_BLUE = 0.25 # 25% overlap required

    # Rewards
    REWARD_STACK = 0.1
    REWARD_PLACE_RED = 1.0
    REWARD_PLACE_BLUE = 0.5
    REWARD_WIN = 100.0
    PENALTY_FAIL = -10.0
    PENALTY_MOVE = -0.01

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        
        self.player_block = None
        self.stacked_blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.stacked_blocks = []
        self.particles = []

        # Create the ground base
        base_x = (self.WIDTH - self.BASE_WIDTH) / 2
        base_y = self.HEIGHT - self.BASE_HEIGHT
        base_block = Block(base_x, base_y, self.BASE_WIDTH, self.BASE_HEIGHT, self.COLOR_BASE, "base")
        self.stacked_blocks.append(base_block)
        
        self._generate_new_block()
        
        return self._get_observation(), self._get_info()

    def _generate_new_block(self):
        block_type = self.np_random.choice(["red", "blue"])
        color = self.COLOR_BLOCK_RED if block_type == "red" else self.COLOR_BLOCK_BLUE
        
        # Start block at a random horizontal position
        start_x = self.np_random.integers(
            self.BLOCK_WIDTH, self.WIDTH - 2 * self.BLOCK_WIDTH
        )
        
        self.player_block = Block(
            start_x, -self.BLOCK_HEIGHT, 
            self.BLOCK_WIDTH, self.BLOCK_HEIGHT, 
            color, block_type, vel_y=self.BLOCK_FALL_SPEED
        )

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle player input ---
        if movement == 3:  # Left
            self.player_block.rect.x -= self.BLOCK_MOVE_SPEED
            reward += self.PENALTY_MOVE
        elif movement == 4: # Right
            self.player_block.rect.x += self.BLOCK_MOVE_SPEED
            reward += self.PENALTY_MOVE

        # Clamp player block to screen bounds
        self.player_block.rect.x = np.clip(
            self.player_block.rect.x, 0, self.WIDTH - self.BLOCK_WIDTH
        )

        if space_held:
            self.player_block.vel_y = self.BLOCK_DROP_SPEED
        
        # --- Update game state ---
        self.player_block.rect.y += self.player_block.vel_y
        
        # --- Collision and stacking logic ---
        landing_surface = None
        for block in self.stacked_blocks:
            if self.player_block.rect.colliderect(block.rect) and self.player_block.rect.bottom > block.rect.top:
                landing_surface = block
                break
        
        if landing_surface:
            # Snap block to the top of the landing surface
            self.player_block.rect.bottom = landing_surface.rect.top

            # Calculate horizontal overlap
            overlap = max(0, min(self.player_block.rect.right, landing_surface.rect.right) - max(self.player_block.rect.left, landing_surface.rect.left))
            
            # Check stability
            threshold = self.STABILITY_RED if self.player_block.type == 'red' else self.STABILITY_BLUE
            is_stable = (overlap / self.BLOCK_WIDTH) >= threshold

            if is_stable:
                # --- Successful placement ---
                # Sound: place_block.wav
                self.score += 1
                reward += self.REWARD_STACK
                if self.player_block.type == 'red':
                    reward += self.REWARD_PLACE_RED
                    self.player_block.color = self.COLOR_BLOCK_RED_STACKED
                else:
                    reward += self.REWARD_PLACE_BLUE
                    self.player_block.color = self.COLOR_BLOCK_BLUE_STACKED
                
                self.stacked_blocks.append(self.player_block)
                self._create_particles(self.player_block.rect.midbottom, self.player_block.color)
                self._generate_new_block()
            else:
                # --- Failed placement ---
                # Sound: fail.wav
                self.game_over = True
                reward += self.PENALTY_FAIL
                self._create_particles(self.player_block.rect.midbottom, (255, 255, 255), count=30)
        
        # Update particles
        self.particles = [p for p in self.particles if p.update()]

        # --- Check termination conditions ---
        terminated = self.game_over
        if self.score >= self.WIN_HEIGHT:
            reward += self.REWARD_WIN
            terminated = True
            # Sound: win_game.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_particles(self, pos, color, count=15):
        for _ in range(count):
            self.particles.append(Particle(pos[0], pos[1], color, self.np_random))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Draw target height line (visual guide)
        target_y = self.HEIGHT - self.BASE_HEIGHT - 10 * self.BLOCK_HEIGHT
        pygame.gfxdraw.hline(self.screen, 0, self.WIDTH, target_y, self.COLOR_TARGET_LINE)
        
        # Draw stacked blocks
        for block in self.stacked_blocks:
            pygame.draw.rect(self.screen, block.color, block.rect)
            pygame.draw.rect(self.screen, self.COLOR_BLOCK_OUTLINE, block.rect, 1)

        # Draw player block
        if self.player_block and not self.game_over:
            # Add a glow effect to the falling block
            glow_rect = self.player_block.rect.inflate(6, 6)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            glow_color = (*self.player_block.color, 50)
            pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=5)
            self.screen.blit(glow_surf, glow_rect.topleft)

            pygame.draw.rect(self.screen, self.player_block.color, self.player_block.rect)
            pygame.draw.rect(self.screen, self.COLOR_BLOCK_OUTLINE, self.player_block.rect, 1)

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p.life / 30))
            color = (*p.color, alpha)
            temp_surf = pygame.Surface((p.radius * 2, p.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p.radius, p.radius), p.radius)
            self.screen.blit(temp_surf, (int(p.pos[0] - p.radius), int(p.pos[1] - p.radius)))

    def _render_ui(self):
        height_text = self.font.render(f"Height: {self.score} / {self.WIN_HEIGHT}", True, self.COLOR_TEXT)
        self.screen.blit(height_text, (10, 10))

        if self.game_over:
            status_text = "GAME OVER" if self.score < self.WIN_HEIGHT else "YOU WIN!"
            status_font = pygame.font.SysFont("monospace", 50, bold=True)
            text_surf = status_font.render(status_text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "win": self.score >= self.WIN_HEIGHT,
        }

    def close(self):
        pygame.font.quit()
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
    # For a more robust human player, see gymnasium.utils.play
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Block Stacker")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    while running:
        # --- Create action from keyboard input ---
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                # print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")
                pass
            if terminated:
                print(f"Episode finished! Final Score: {info['score']}, Steps: {info['steps']}")
        
        # --- Render the observation to the display window ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
    env.close()