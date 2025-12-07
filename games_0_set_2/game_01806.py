
# Generated: 2025-08-27T18:21:22.990865
# Source Brief: brief_01806.md
# Brief Index: 1806

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
    """
    A Gymnasium environment for a top-down arcade survival game.
    The player must survive a chaotic rain of colorful blocks for 60 seconds
    across 3 escalating stages.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your white square and dodge the falling blocks."
    )
    game_description = (
        "Survive a chaotic rain of colorful blocks for 60 seconds. Each stage gets faster. "
        "Red blocks are slow (1pt), green are medium (3pt), and blue are fast (5pt). "
        "Get points for close calls!"
    )

    # Frame advance setting
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255)
    COLOR_TEXT = (220, 220, 220)
    BLOCK_COLORS = {
        "red": (255, 50, 50),
        "green": (50, 255, 50),
        "blue": (50, 100, 255)
    }

    # Player settings
    PLAYER_SIZE = 20
    PLAYER_SPEED = 5.0

    # Game settings
    MAX_STAGES = 3
    STAGE_DURATION_SECONDS = 60
    
    # Block settings
    INITIAL_BLOCK_SPEEDS = {"red": 2.0, "green": 4.0, "blue": 6.0}
    INITIAL_SPAWN_RATE_PER_SEC = 0.5
    SPAWN_RATE_INCREASE_PER_SEC = 0.05
    SPEED_INCREASE_PER_10_SEC = 0.1
    NEAR_MISS_DISTANCE = 10
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Initialize state variables
        self.player_pos = None
        self.blocks = None
        self.particles = None
        self.score = None
        self.stage = None
        self.stage_timer = None
        self.total_steps = None
        self.game_over = None
        self.block_spawn_rate_per_frame = None
        self.current_block_speeds = None
        self.speed_increase_timer = None
        self.block_spawn_cooldown = None
        self.rng = None
        
        # This call will initialize the state for the first time
        # self.reset()

        # Self-validation
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        else:
            self.rng = np.random.default_rng()

        # Initialize player state
        self.player_pos = pygame.math.Vector2(
            self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - self.PLAYER_SIZE * 2
        )

        # Initialize game state
        self.score = 0
        self.stage = 1
        self.total_steps = 0
        self.game_over = False
        
        self.blocks = []
        self.particles = []

        self._reset_stage_parameters()

        return self._get_observation(), self._get_info()

    def _reset_stage_parameters(self):
        """Resets parameters that change with each stage."""
        self.stage_timer = self.STAGE_DURATION_SECONDS * self.FPS
        self.block_spawn_rate_per_frame = self.INITIAL_SPAWN_RATE_PER_SEC / self.FPS
        self.current_block_speeds = self.INITIAL_BLOCK_SPEEDS.copy()
        self.speed_increase_timer = 10 * self.FPS
        self.block_spawn_cooldown = 0
        self.blocks.clear() # Clear existing blocks on stage transition
        self._create_particles(100, self.player_pos, (0, 255, 100), 5) # Stage clear effect

    def step(self, action):
        movement = action[0]
        
        reward = 0.0
        terminated = False

        # --- Update Game Logic ---
        self.total_steps += 1
        self.stage_timer -= 1
        self.block_spawn_cooldown -= 1

        self._handle_player_movement(movement)
        self._update_difficulty()
        self._update_blocks()
        self._update_particles()
        
        # --- Calculate Rewards and Check Termination ---
        player_rect = self._get_player_rect()

        # 1. Survival reward and safe zone penalty
        reward += 0.1
        safe_zone = pygame.Rect(
            self.SCREEN_WIDTH / 4, self.SCREEN_HEIGHT / 4,
            self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2
        )
        if safe_zone.colliderect(player_rect):
            reward -= 0.2
        
        # 2. Block interaction rewards/penalties
        for block in self.blocks:
            # Collision check
            if player_rect.colliderect(block["rect"]):
                # sfx: player_explosion
                self.game_over = True
                terminated = True
                reward = -100.0
                self._create_particles(200, self.player_pos, self.COLOR_PLAYER, 8)
                break

            # Near miss check
            if not block["reward_given"]:
                near_miss_rect = player_rect.inflate(
                    self.NEAR_MISS_DISTANCE * 2, self.NEAR_MISS_DISTANCE * 2
                )
                if near_miss_rect.colliderect(block["rect"]):
                    # sfx: near_miss_ding
                    reward += block["points"]
                    block["reward_given"] = True
                    self._create_particles(10, block["rect"].center, block["color"], 2)

        # 3. Stage completion
        if not terminated and self.stage_timer <= 0:
            self.stage += 1
            if self.stage > self.MAX_STAGES:
                # sfx: game_win_fanfare
                terminated = True # Game won
            else:
                # sfx: stage_clear_sound
                self._reset_stage_parameters()
            reward += 100.0

        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _handle_player_movement(self, movement):
        if self.game_over:
            return
            
        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1:  # Up
            move_vec.y = -1
        elif movement == 2:  # Down
            move_vec.y = 1
        elif movement == 3:  # Left
            move_vec.x = -1
        elif movement == 4:  # Right
            move_vec.x = 1

        if move_vec.length_squared() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_SPEED

        # Clamp player position to screen bounds
        half_size = self.PLAYER_SIZE / 2
        self.player_pos.x = np.clip(self.player_pos.x, half_size, self.SCREEN_WIDTH - half_size)
        self.player_pos.y = np.clip(self.player_pos.y, half_size, self.SCREEN_HEIGHT - half_size)

    def _update_difficulty(self):
        # Increase spawn rate over time
        self.block_spawn_rate_per_frame += self.SPAWN_RATE_INCREASE_PER_SEC / self.FPS / self.FPS

        # Increase block speeds every 10 seconds
        self.speed_increase_timer -= 1
        if self.speed_increase_timer <= 0:
            self.speed_increase_timer = 10 * self.FPS
            for color in self.current_block_speeds:
                self.current_block_speeds[color] += self.SPEED_INCREASE_PER_10_SEC

    def _update_blocks(self):
        # Spawn new blocks
        if self.block_spawn_cooldown <= 0:
            # sfx: block_spawn_whoosh
            self._spawn_block()
            self.block_spawn_cooldown = int(1.0 / self.block_spawn_rate_per_frame)

        # Move and remove old blocks
        for block in self.blocks[:]:
            block["pos"].y += block["speed"]
            block["rect"].topleft = block["pos"]
            
            # Add to trail
            block["trail"].append(block["pos"].copy())

            if block["rect"].top > self.SCREEN_HEIGHT:
                self.blocks.remove(block)
                # Successful dodge adds to score
                if not block["reward_given"]:
                     self.score += block["points"]


    def _spawn_block(self):
        block_type = self.rng.choice(list(self.BLOCK_COLORS.keys()))
        color = self.BLOCK_COLORS[block_type]
        speed = self.current_block_speeds[block_type]
        
        size = self.rng.integers(15, 35)
        pos = pygame.math.Vector2(self.rng.integers(0, self.SCREEN_WIDTH - size), -size)
        
        points = {"red": 1, "green": 3, "blue": 5}[block_type]

        self.blocks.append({
            "pos": pos,
            "rect": pygame.Rect(int(pos.x), int(pos.y), size, size),
            "color": color,
            "speed": speed,
            "points": points,
            "reward_given": False,
            "trail": deque(maxlen=10)
        })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95  # friction
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _create_particles(self, count, position, color, max_speed):
        for _ in range(count):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * max_speed
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": position.copy(),
                "vel": vel,
                "life": self.rng.integers(15, 30),
                "color": color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render block trails
        for block in self.blocks:
            for i, pos in enumerate(block["trail"]):
                alpha = int(255 * (i / len(block["trail"])) * 0.5)
                trail_color = block["color"] + (alpha,)
                trail_surf = pygame.Surface(block["rect"].size, pygame.SRCALPHA)
                pygame.draw.rect(trail_surf, trail_color, (0, 0, block["rect"].width, block["rect"].height))
                self.screen.blit(trail_surf, (int(pos.x), int(pos.y)))

        # Render blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in block["color"]), block["rect"], 1)

        # Render particles
        for p in self.particles:
            size = max(1, p["life"] // 5)
            pygame.draw.circle(self.screen, p["color"], (int(p["pos"].x), int(p["pos"].y)), size)

        # Render player
        if not self.game_over:
            player_rect = self._get_player_rect()
            
            # Glow effect
            glow_radius = int(self.PLAYER_SIZE * 1.2)
            glow_center = player_rect.center
            
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, self.COLOR_PLAYER_GLOW + (50,), (glow_radius, glow_radius), glow_radius)
            pygame.draw.circle(s, self.COLOR_PLAYER_GLOW + (80,), (glow_radius, glow_radius), int(glow_radius*0.8))
            self.screen.blit(s, (glow_center[0] - glow_radius, glow_center[1] - glow_radius))

            # Player square
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
            pygame.draw.rect(self.screen, self.COLOR_BG, player_rect, 2)


    def _render_ui(self):
        # Timer
        timer_seconds = max(0, self.stage_timer // self.FPS)
        timer_text = self.font_medium.render(f"TIME: {timer_seconds}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (10, 10))

        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)
        
        # Stage
        stage_str = f"STAGE {self.stage} / {self.MAX_STAGES}"
        if self.game_over and self.stage > self.MAX_STAGES:
            stage_str = "VICTORY!"
        elif self.game_over:
            stage_str = "GAME OVER"
            
        stage_text = self.font_large.render(stage_str, True, self.COLOR_TEXT)
        stage_rect = stage_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 30))
        self.screen.blit(stage_text, stage_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.total_steps,
            "stage": self.stage,
        }
        
    def _get_player_rect(self):
        return pygame.Rect(
            int(self.player_pos.x - self.PLAYER_SIZE / 2),
            int(self.player_pos.y - self.PLAYER_SIZE / 2),
            self.PLAYER_SIZE,
            self.PLAYER_SIZE
        )

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Validating implementation...")
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    env.reset()

    # Create a window to display the game
    pygame.display.set_caption("Block Rain Survival")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # 0=none
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                env.reset()
                total_reward = 0

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
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting, or wait for 'R' key
            pygame.time.wait(2000)
            env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)

    env.close()