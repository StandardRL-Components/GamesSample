import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:55:09.830574
# Source Brief: brief_01361.md
# Brief Index: 1361
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game objects
class Block:
    """Represents a player-controlled falling block."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vy = 1.0
        self.color = color
        self.width = 40
        self.height = 40
        self.is_landed = False
        self.glow_size = 0
        self.glow_alpha = 0

    def get_rect(self):
        return pygame.Rect(self.x - self.width / 2, self.y - self.height / 2, self.width, self.height)

class Platform:
    """Represents a moving target platform."""
    def __init__(self, x, y, level):
        self.base_x = x
        self.y = y
        self.width = 100
        self.height = 15
        self.amplitude = 40 + level * 20
        self.frequency = 0.015 + level * 0.003
        self.offset = random.uniform(0, 2 * math.pi)
        self.x = self.base_x

    def update(self, steps):
        self.x = self.base_x + math.sin(steps * self.frequency + self.offset) * self.amplitude

    def get_rect(self):
        return pygame.Rect(self.x - self.width / 2, self.y - self.height / 2, self.width, self.height)

class Particle:
    """Represents a single particle for visual effects."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-4, -1)
        self.life = 60 # 2 seconds at 30fps
        self.color = color
        self.radius = random.uniform(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1 # Gravity on particles
        self.life -= 1


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Drop three blocks and land them simultaneously on their moving platforms to score points and advance."
    )
    user_guide = (
        "Controls: Use ↑/↓ to adjust the speed of the first block, ←/→ for the second, and Shift/Space for the third. "
        "Land all blocks at the same time to win."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    NUM_LEVELS = 5
    MAX_EPISODE_STEPS = 1000

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 50)
    COLOR_PLATFORM = (220, 220, 240)
    COLOR_PLATFORM_FLASH = (255, 255, 100)
    BLOCK_COLORS = [(255, 80, 80), (80, 255, 80), (80, 120, 255)]
    COLOR_TEXT = (240, 240, 240)

    # Physics
    GRAVITY = 0.15
    BLOCK_SPEED_ADJUST = 0.4
    MAX_BLOCK_SPEED = 8.0
    MIN_BLOCK_SPEED = 0.5

    # Rewards
    REWARD_PER_STEP = -0.01
    REWARD_LANDING = 1.0
    REWARD_SYNC = 5.0
    REWARD_LEVEL_COMPLETE = 50.0
    REWARD_GAME_WIN = 100.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('Consolas', 20, bold=True)
        self.font_title = pygame.font.SysFont('Consolas', 48, bold=True)

        # State variables initialized in reset()
        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False
        self.game_won = False
        self.blocks = []
        self.platforms = []
        self.particles = []
        self.sync_flash_timer = 0

        # self.reset() is called by the environment wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False
        self.game_won = False
        self.particles = []
        self.sync_flash_timer = 0

        self._setup_level()

        return self._get_observation(), self._get_info()

    def _setup_level(self):
        self.blocks.clear()
        self.platforms.clear()

        num_entities = 3
        spacing = self.SCREEN_WIDTH / (num_entities + 1)

        for i in range(num_entities):
            x_pos = spacing * (i + 1)
            self.blocks.append(Block(x_pos, 50, self.BLOCK_COLORS[i]))
            self.platforms.append(Platform(x_pos, self.SCREEN_HEIGHT - 40, self.level))

    def step(self, action):
        if self.game_over:
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        reward = self.REWARD_PER_STEP
        self.steps += 1

        self._handle_input(action)
        self._update_game_state()

        land_reward, sync_reward, level_reward, terminated, win = self._check_events_and_rewards()
        reward += land_reward + sync_reward + level_reward

        # We only add non-step rewards to the score for clarity in the UI
        self.score += land_reward + sync_reward + level_reward

        self.game_over = terminated or self.steps >= self.MAX_EPISODE_STEPS
        if win:
            self.game_won = True
            reward += self.REWARD_GAME_WIN
            self.score += self.REWARD_GAME_WIN

        truncated = self.steps >= self.MAX_EPISODE_STEPS

        return (
            self._get_observation(),
            reward,
            self.game_over,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Block 1 Control (Up/Down)
        if movement == 1: # Up
            self.blocks[0].vy -= self.BLOCK_SPEED_ADJUST
        elif movement == 2: # Down
            self.blocks[0].vy += self.BLOCK_SPEED_ADJUST

        # Block 2 Control (Left/Right)
        if movement == 3: # Left
            self.blocks[1].vy -= self.BLOCK_SPEED_ADJUST
        elif movement == 4: # Right
            self.blocks[1].vy += self.BLOCK_SPEED_ADJUST

        # Block 3 Control (Space/Shift)
        if space_held and not shift_held:
            self.blocks[2].vy += self.BLOCK_SPEED_ADJUST
        elif shift_held and not space_held:
            self.blocks[2].vy -= self.BLOCK_SPEED_ADJUST

        # Clamp speeds
        for block in self.blocks:
            block.vy = max(self.MIN_BLOCK_SPEED, min(self.MAX_BLOCK_SPEED, block.vy))

    def _update_game_state(self):
        # Update platforms
        for p in self.platforms:
            p.update(self.steps)

        # Update blocks
        for block in self.blocks:
            if not block.is_landed:
                block.vy += self.GRAVITY
                block.y += block.vy
            # Animate glow effect based on speed
            block.glow_size = int(5 + abs(block.vy) * 2)
            block.glow_alpha = int(50 + abs(block.vy) * 10)

        # Update particles
        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)

        # Update sync flash
        if self.sync_flash_timer > 0:
            self.sync_flash_timer -= 1

    def _check_events_and_rewards(self):
        land_reward = 0
        sync_reward = 0
        level_reward = 0
        terminated = False
        win = False

        landed_this_step_count = 0

        for i, block in enumerate(self.blocks):
            if block.is_landed:
                continue

            platform = self.platforms[i]
            block_rect = block.get_rect()
            platform_rect = platform.get_rect()

            # Check for landing
            if block_rect.colliderect(platform_rect) and block.vy > 0:
                if block.y + block.height / 2 < platform.y:
                    block.is_landed = True
                    block.y = platform.y - block.height / 2
                    block.vy = 0
                    landed_this_step_count += 1
                    land_reward += self.REWARD_LANDING
                    # sfx: landing_sound()

            # Check for out of bounds
            if block.y > self.SCREEN_HEIGHT + block.height:
                terminated = True
                # sfx: fail_sound()
                return 0, 0, 0, True, False

        num_landed_blocks = sum(1 for b in self.blocks if b.is_landed)

        if num_landed_blocks == len(self.blocks):
            if landed_this_step_count == len(self.blocks): # All landed in the same step
                # SYNC SUCCESS
                sync_reward = self.REWARD_SYNC
                level_reward = self.REWARD_LEVEL_COMPLETE
                self.level += 1
                self.sync_flash_timer = 15 # 0.5s flash

                # sfx: level_complete_sound()
                for p in self.platforms:
                    self._create_particles(p.x, p.y, p.get_rect().w, p.height, self.COLOR_PLATFORM_FLASH)

                if self.level > self.NUM_LEVELS:
                    win = True
                    terminated = True
                    # sfx: game_win_sound()
                else:
                    self._setup_level()
            else: # Landed, but not in sync
                terminated = True
                # sfx: fail_sound()

        return land_reward, sync_reward, level_reward, terminated, win

    def _create_particles(self, x, y, width, height, color):
        for _ in range(30):
            px = x + random.uniform(-width / 2, width / 2)
            py = y - height / 2
            self.particles.append(Particle(px, py, color))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p.life / 60))))
            color = (p.color[0], p.color[1], p.color[2], alpha)
            temp_surf = pygame.Surface((p.radius*2, p.radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (int(p.radius), int(p.radius)), int(p.radius))
            self.screen.blit(temp_surf, (int(p.x - p.radius), int(p.y - p.radius)))

        # Draw platforms
        platform_color = self.COLOR_PLATFORM_FLASH if self.sync_flash_timer > 0 else self.COLOR_PLATFORM
        for p in self.platforms:
            rect = p.get_rect()
            pygame.gfxdraw.box(self.screen, rect, platform_color)

        # Draw blocks
        for block in self.blocks:
            # Draw glow
            if not block.is_landed:
                glow_surf = pygame.Surface((block.width + block.glow_size*2, block.height + block.glow_size*2), pygame.SRCALPHA)
                glow_color = (*block.color, block.glow_alpha)
                pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=8)
                self.screen.blit(glow_surf, (int(block.x - block.width/2 - block.glow_size), int(block.y - block.height/2 - block.glow_size)))

            # Draw block
            rect = block.get_rect()
            pygame.gfxdraw.box(self.screen, rect, block.color)
            pygame.gfxdraw.rectangle(self.screen, rect, (255,255,255, 50)) # Highlight border

    def _render_ui(self):
        # Render score and level
        level_text = self.font_ui.render(f"Level: {min(self.level, self.NUM_LEVELS)} / {self.NUM_LEVELS}", True, self.COLOR_TEXT)
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 10))
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.game_won:
                end_text = self.font_title.render("YOU WIN!", True, self.COLOR_PLATFORM_FLASH)
            else:
                end_text = self.font_title.render("GAME OVER", True, self.BLOCK_COLORS[0])

            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block is for human play and is not part of the environment definition.
    # It requires a display.
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "mac", etc.
    
    env = GameEnv(render_mode="rgb_array")

    # Manual play loop
    obs, info = env.reset()
    done = False

    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Trio Drop")
    clock = pygame.time.Clock()
    running = True

    while running:
        # Action defaults
        movement = 0 # none
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if done and running:
            # Allow reset on key press after game over
            if keys[pygame.K_r]:
                obs, info = env.reset()
                done = False

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(env.FPS)

    env.close()