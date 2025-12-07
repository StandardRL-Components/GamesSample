
# Generated: 2025-08-28T02:16:18.700509
# Source Brief: brief_04396.md
# Brief Index: 4396

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to move, and ↑ or Space to jump. "
        "Collect coins and reach the green goal!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist platformer. Navigate a procedurally generated level, "
        "collecting coins and racing against the clock to reach the goal."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # Class-level variable for persistent difficulty
    successful_completions = 0
    difficulty_gap_mod = 0.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # Use 30 FPS for smoother auto-advance
        self.TIME_LIMIT = 30  # seconds

        # Physics constants
        self.GRAVITY = 0.6
        self.PLAYER_ACCEL = 0.8
        self.PLAYER_FRICTION = -0.15
        self.PLAYER_JUMP_STRENGTH = -12
        self.MAX_VEL_X = 6
        
        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLATFORM = (100, 110, 120)
        self.COLOR_COIN = (255, 220, 0)
        self.COLOR_GOAL = (0, 255, 120)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_TEXT_WARN = (255, 80, 80)
        self.COLOR_TRAIL = (200, 200, 255)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        
        # Initialize state variables
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_size = 20
        self.on_ground = False
        self.coyote_time = 0
        self.jump_buffer = 0
        
        self.platforms = []
        self.coins = []
        self.particles = []
        self.goal_rect = pygame.Rect(0,0,0,0)
        
        self.camera_x = 0
        
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.game_over = False
        
        self.reset()
        self.validate_implementation()
    
    def _generate_level(self):
        self.platforms.clear()
        self.coins.clear()
        
        # Start platform
        start_platform = pygame.Rect(50, self.HEIGHT - 50, 200, 50)
        self.platforms.append(start_platform)
        
        current_x = start_platform.right
        current_y = start_platform.top
        
        level_length = 5000
        
        while current_x < level_length:
            # Determine gap and next platform properties
            gap = self.np_random.integers(80, 120) + self.difficulty_gap_mod
            width = self.np_random.integers(150, 400)
            height_change = self.np_random.integers(-80, 80)
            
            next_x = current_x + gap
            next_y = np.clip(current_y + height_change, 150, self.HEIGHT - 50)
            
            new_platform = pygame.Rect(next_x, next_y, width, self.HEIGHT - next_y)
            self.platforms.append(new_platform)
            
            # Place coins above the new platform
            num_coins = self.np_random.integers(1, 4)
            for i in range(num_coins):
                coin_x = new_platform.left + (i + 1) * (new_platform.width / (num_coins + 1))
                coin_y = new_platform.top - 40
                self.coins.append(pygame.Rect(coin_x, coin_y, 12, 12))
                
            current_x = new_platform.right
            current_y = new_platform.top
        
        # Goal
        last_platform = self.platforms[-1]
        self.goal_rect = pygame.Rect(last_platform.centerx - 25, last_platform.top - 50, 50, 50)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_level()
        
        start_platform = self.platforms[0]
        self.player_pos = pygame.math.Vector2(start_platform.centerx, start_platform.top - self.player_size)
        self.player_vel = pygame.math.Vector2(0, 0)
        
        self.on_ground = False
        self.coyote_time = 0
        self.jump_buffer = 0
        self.player_trail = []
        
        self.particles.clear()
        
        self.steps = 0
        self.score = 0
        self.timer = self.TIME_LIMIT
        self.game_over = False
        
        self.camera_x = self.player_pos.x - self.WIDTH / 4

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        terminated = False
        
        # --- Handle Input ---
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel.x -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel.x += self.PLAYER_ACCEL
        
        # Jump input buffering
        if movement == 1 or space_held:
            self.jump_buffer = 5 # buffer for 5 frames
        else:
            self.jump_buffer = max(0, self.jump_buffer - 1)

        # --- Physics and Gameplay Update ---
        prev_dist_to_goal = abs(self.player_pos.x - self.goal_rect.centerx)

        # Apply friction
        self.player_vel.x += self.player_vel.x * self.PLAYER_FRICTION
        if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0
        self.player_vel.x = np.clip(self.player_vel.x, -self.MAX_VEL_X, self.MAX_VEL_X)
        
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        
        # Jumping logic with coyote time
        if self.jump_buffer > 0 and self.coyote_time > 0:
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self.jump_buffer = 0
            self.coyote_time = 0
            # SFX: Jump sound

        # Update position
        self.player_pos.x += self.player_vel.x
        self.player_pos.y += self.player_vel.y
        
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.player_size, self.player_size)
        
        # --- Collision Detection ---
        self.on_ground = False
        for plat in self.platforms:
            if player_rect.colliderect(plat):
                # Check vertical collision (landing on top)
                if self.player_vel.y > 0 and player_rect.bottom - self.player_vel.y <= plat.top:
                    self.player_pos.y = plat.top - self.player_size
                    self.player_vel.y = 0
                    self.on_ground = True
                # Check horizontal collision
                elif player_rect.right - self.player_vel.x <= plat.left or player_rect.left - self.player_vel.x >= plat.right:
                     if self.player_vel.x > 0:
                         self.player_pos.x = plat.left - self.player_size
                     elif self.player_vel.x < 0:
                         self.player_pos.x = plat.right
                     self.player_vel.x = 0
                # Check vertical collision (hitting bottom)
                elif self.player_vel.y < 0 and player_rect.top - self.player_vel.y >= plat.bottom:
                    self.player_pos.y = plat.bottom
                    self.player_vel.y = 0
        
        if self.on_ground:
            self.coyote_time = 7 # 7 frames of coyote time
        else:
            self.coyote_time = max(0, self.coyote_time - 1)
            
        # Coin collection
        for coin in self.coins[:]:
            if player_rect.colliderect(coin):
                self.coins.remove(coin)
                self.score += 1
                reward += 1
                # SFX: Coin collect sound
                # Particle burst
                for _ in range(10):
                    self.particles.append({
                        "pos": pygame.math.Vector2(coin.center),
                        "vel": pygame.math.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)),
                        "life": 15,
                        "color": self.COLOR_COIN
                    })
        
        # --- Update Game State ---
        self.steps += 1
        self.timer -= 1 / self.FPS
        
        # Update player trail
        if self.steps % 2 == 0:
            self.player_trail.append(player_rect.copy())
            if len(self.player_trail) > 5:
                self.player_trail.pop(0)

        # --- Termination Checks ---
        # 1. Reached goal
        if player_rect.colliderect(self.goal_rect):
            reward += 100
            terminated = True
            self.game_over = True
            # SFX: Level complete fanfare
            # Update persistent difficulty
            GameEnv.successful_completions += 1
            if GameEnv.successful_completions % 500 == 0:
                GameEnv.difficulty_gap_mod += 0.5
        
        # 2. Fell off world
        if self.player_pos.y > self.HEIGHT + 100:
            reward -= 100
            terminated = True
            self.game_over = True
            # SFX: Falling sound
        
        # 3. Ran out of time
        if self.timer <= 0:
            self.timer = 0
            if not terminated: # Only apply penalty if not already terminated
                reward -= 50
                terminated = True
                self.game_over = True
                # SFX: Timeout buzzer

        # --- Reward Shaping ---
        current_dist_to_goal = abs(self.player_pos.x - self.goal_rect.centerx)
        dist_delta = prev_dist_to_goal - current_dist_to_goal
        reward += dist_delta * 0.1 # Reward for getting closer to goal
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _update_camera(self):
        # Smoothly follow player with a lead in the direction of movement
        lead = self.WIDTH / 4 * np.sign(self.player_vel.x) if abs(self.player_vel.x) > 1 else 0
        target_x = self.player_pos.x - self.WIDTH / 2 + lead
        # Smooth interpolation
        self.camera_x += (target_x - self.camera_x) * 0.1

    def _render_game(self):
        self._update_camera()

        # Parallax background
        for i in range(5):
            # Darker, slower shapes in the background
            offset = (self.camera_x * (0.1 * (i + 1))) % self.WIDTH
            pygame.draw.rect(self.screen, (25 + i*5, 35 + i*5, 45 + i*5), (0 - offset, 100 + i*50, 50, 50))
            pygame.draw.rect(self.screen, (25 + i*5, 35 + i*5, 45 + i*5), (300 - offset, 50 + i*40, 30, 30))
            pygame.draw.rect(self.screen, (25 + i*5, 35 + i*5, 45 + i*5), (600 - offset, 150 + i*60, 40, 40))

        # Render platforms
        for plat in self.platforms:
            screen_rect = plat.move(-self.camera_x, 0)
            if screen_rect.colliderect(self.screen.get_rect()):
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, screen_rect)

        # Render coins
        for coin in self.coins:
            screen_rect = coin.move(-self.camera_x, 0)
            if screen_rect.colliderect(self.screen.get_rect()):
                pygame.gfxdraw.filled_circle(self.screen, int(screen_rect.centerx), int(screen_rect.centery), int(coin.width/2), self.COLOR_COIN)
                pygame.gfxdraw.aacircle(self.screen, int(screen_rect.centerx), int(screen_rect.centery), int(coin.width/2), self.COLOR_COIN)

        # Render goal
        screen_goal_rect = self.goal_rect.move(-self.camera_x, 0)
        pygame.draw.rect(self.screen, self.COLOR_GOAL, screen_goal_rect)
        pygame.draw.rect(self.screen, (255,255,255), screen_goal_rect, 2) # White outline

        # Render particles
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p["life"] / 15))
                color = (*p["color"], alpha)
                temp_surf = pygame.Surface((4,4), pygame.SRCALPHA)
                pygame.draw.rect(temp_surf, color, (0,0,4,4))
                self.screen.blit(temp_surf, (p["pos"].x - self.camera_x - 2, p["pos"].y - 2))
                
        # Render player trail
        for i, trail_rect in enumerate(self.player_trail):
            alpha = int(100 * (i + 1) / len(self.player_trail))
            trail_surf = pygame.Surface(trail_rect.size, pygame.SRCALPHA)
            trail_surf.fill((*self.COLOR_TRAIL, alpha))
            screen_trail_rect = trail_rect.move(-self.camera_x, 0)
            self.screen.blit(trail_surf, screen_trail_rect.topleft)
            
        # Render player
        player_rect = pygame.Rect(self.player_pos.x - self.camera_x, self.player_pos.y, self.player_size, self.player_size)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
    
    def _render_ui(self):
        # Timer
        time_text = f"TIME: {self.timer:.1f}"
        time_color = self.COLOR_TEXT_WARN if self.timer < 5 else self.COLOR_TEXT
        time_surface = self.font_ui.render(time_text, True, time_color)
        self.screen.blit(time_surface, (10, 10))

        # Score (Coins)
        score_text = f"COINS: {self.score}"
        score_surface = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surface, (self.WIDTH - score_surface.get_width() - 10, 10))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "player_pos": (self.player_pos.x, self.player_pos.y),
            "goal_pos": (self.goal_rect.centerx, self.goal_rect.centery)
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "dummy" to run headlessly
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually ---
    # Use Pygame to display the environment and capture inputs
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Minimalist Platformer")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    while not terminated:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        # No down action for player
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # If auto_advance is True, the environment's internal clock is not used for display.
        # We control the display loop speed here.
        clock.tick(env.FPS)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            # Optional: reset and play again
            # obs, info = env.reset()
            # terminated = False
            
    env.close()