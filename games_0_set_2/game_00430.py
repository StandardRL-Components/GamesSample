
# Generated: 2025-08-27T13:37:15.247996
# Source Brief: brief_00430.md
# Brief Index: 430

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↑ or Space to jump. Collect coins and reach the green flag!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, side-scrolling platformer. Jump across perilous pits, "
        "collect coins for score, and reach the flag before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and physics constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.GRAVITY = 0.8
        self.PLAYER_SPEED = 5
        self.JUMP_STRENGTH = -14
        self.MAX_STEPS = 1800  # 60 seconds * 30 FPS

        # Define spaces
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
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_PLAYER_GLOW = (255, 150, 150)
        self.COLOR_PLATFORM = (100, 110, 130)
        self.COLOR_COIN = (255, 220, 100)
        self.COLOR_FLAG_POLE = (200, 200, 200)
        self.COLOR_FLAG = (80, 220, 80)
        self.COLOR_PIT = (0, 0, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_SPARKLE = (255, 255, 200)
        self.COLOR_DUST = (150, 140, 130)
        
        # Game state variables
        self.player_rect = None
        self.player_vel = None
        self.on_ground = False
        self.platforms = []
        self.coins = []
        self.pits = []
        self.flag_rect = None
        self.flag_pole_rect = None
        self.particles = []
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over = False
        self.level = 1
        self.pit_freq_modifier = 0.0
        self.last_dist_to_flag = 0

        self.reset()
        self.validate_implementation()

    def _generate_level(self):
        self.platforms.clear()
        self.coins.clear()
        self.pits.clear()

        # Start platform
        start_plat = pygame.Rect(0, 350, 120, 50)
        self.platforms.append(start_plat)
        
        x = start_plat.width
        last_y = start_plat.y

        # Procedurally generate platforms, pits, and coins
        while x < self.SCREEN_WIDTH - 100:
            is_gap = self.np_random.random() < (0.2 + self.pit_freq_modifier)
            
            if is_gap:
                gap_width = self.np_random.integers(60, 100)
                self.pits.append(pygame.Rect(x, self.SCREEN_HEIGHT - 10, gap_width, 10))
                x += gap_width
            else:
                plat_width = self.np_random.integers(80, 200)
                plat_y = np.clip(last_y + self.np_random.integers(-90, 90), 200, 350)
                
                new_platform = pygame.Rect(x, plat_y, plat_width, self.SCREEN_HEIGHT - plat_y)
                self.platforms.append(new_platform)

                # Add coins
                num_coins = self.np_random.integers(1, 4)
                for i in range(num_coins):
                    coin_x = x + (i + 1) * (plat_width / (num_coins + 1))
                    coin_y = plat_y - 30 - self.np_random.integers(0, 20)
                    self.coins.append(pygame.Rect(int(coin_x), int(coin_y), 12, 12))
                
                last_y = plat_y
                x += plat_width
        
        # End platform and flag
        last_plat = self.platforms[-1]
        self.flag_pole_rect = pygame.Rect(last_plat.right - 30, last_plat.top - 60, 5, 60)
        self.flag_rect = pygame.Rect(self.flag_pole_rect.left - 30, self.flag_pole_rect.top, 30, 20)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.pit_freq_modifier = min(0.3, 0.05 * (self.level - 1))
        self._generate_level()

        self.player_rect = pygame.Rect(50, 250, 20, 20)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False

        self.steps = 0
        self.score = 0
        self.time_remaining = self.MAX_STEPS
        self.game_over = False
        self.particles.clear()
        
        self.last_dist_to_flag = self.flag_rect.centerx - self.player_rect.centerx

        return self._get_observation(), self._get_info()

    def _create_particles(self, pos, count, color, life_range, speed_range, gravity=0.2):
        for _ in range(count):
            vel = pygame.Vector2(
                self.np_random.uniform(-speed_range, speed_range),
                self.np_random.uniform(-speed_range, speed_range)
            )
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "life": self.np_random.integers(life_range[0], life_range[1]),
                "max_life": life_range[1],
                "color": color,
                "gravity": gravity
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Handle Input ---
        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_SPEED
        else:
            self.player_vel.x = 0

        # Jump (Up arrow or Space)
        if (movement == 1 or space_held) and self.on_ground:
            self.player_vel.y = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump
            self._create_particles(self.player_rect.midbottom, 15, self.COLOR_DUST, (10, 20), 2.5)

        # --- Physics Update ---
        self.player_vel.y += self.GRAVITY
        self.player_rect.x += int(self.player_vel.x)
        self.player_rect.y += int(self.player_vel.y)
        self.on_ground = False

        # --- Collision Detection ---
        # Screen boundaries
        self.player_rect.left = max(0, self.player_rect.left)
        self.player_rect.right = min(self.SCREEN_WIDTH, self.player_rect.right)

        # Platforms
        for plat in self.platforms:
            if self.player_rect.colliderect(plat) and self.player_vel.y > 0:
                # Check if player was above the platform in the previous frame
                if self.player_rect.bottom - self.player_vel.y <= plat.top:
                    self.player_rect.bottom = plat.top
                    self.player_vel.y = 0
                    self.on_ground = True
                    break

        # --- Game Logic ---
        # Coin collection
        collected_coins = []
        for coin in self.coins:
            if self.player_rect.colliderect(coin):
                collected_coins.append(coin)
                self.score += 10
                reward += 10
                # sfx: coin collect
                self._create_particles(coin.center, 20, self.COLOR_SPARKLE, (15, 25), 3, 0.1)
        self.coins = [c for c in self.coins if c not in collected_coins]
        
        # Reward for getting closer to the flag
        current_dist_to_flag = self.flag_rect.centerx - self.player_rect.centerx
        dist_delta = self.last_dist_to_flag - current_dist_to_flag
        reward += dist_delta * 0.1 # Reward for progress
        self.last_dist_to_flag = current_dist_to_flag

        # --- Termination Conditions ---
        terminated = False
        
        # Pitfall
        if self.player_rect.top > self.SCREEN_HEIGHT:
            reward = -100
            terminated = True
            self.game_over = True
            # sfx: fall
        
        # Reached flag
        if self.player_rect.colliderect(self.flag_rect):
            reward = 100 + self.time_remaining * 0.1 # Bonus for speed
            self.score += 1000
            terminated = True
            self.game_over = True
            self.level += 1
            # sfx: win
            self._create_particles(self.flag_rect.center, 50, self.COLOR_FLAG, (30, 60), 4, 0.05)
        
        # Time and step limits
        self.steps += 1
        self.time_remaining -= 1
        reward -= 0.01 # Small penalty for time passing
        if self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            if not terminated: # Don't overwrite win/loss reward
                reward = -50 
            terminated = True
            self.game_over = True

        # --- Particle Update ---
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["life"] -= 1
            p["vel"].y += p["gravity"]
            p["pos"] += p["vel"]

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_game(self):
        # Background elements
        for i in range(30):
            x = (hash(i * 10) % self.SCREEN_WIDTH + self.steps * 0.1) % self.SCREEN_WIDTH
            y = hash(i * 20) % self.SCREEN_HEIGHT
            pygame.gfxdraw.pixel(self.screen, int(x), y, (50, 50, 80))

        # Platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat)
        
        # Pits (visual only)
        for pit in self.pits:
            pygame.draw.rect(self.screen, self.COLOR_PIT, (pit.x, pit.y-10, pit.width, 20))

        # Coins
        for coin in self.coins:
            y_offset = math.sin(self.steps * 0.1 + coin.x) * 3
            pygame.draw.circle(self.screen, self.COLOR_COIN, coin.center, coin.width // 2)
            pygame.draw.circle(self.screen, (255,255,255), coin.center, coin.width // 2 - 3, 1)

        # Flag
        pygame.draw.rect(self.screen, self.COLOR_FLAG_POLE, self.flag_pole_rect)
        flag_points = [
            self.flag_rect.topleft,
            self.flag_rect.topright,
            (self.flag_rect.centerx, self.flag_rect.centery)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_points)

        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p["life"] / p["max_life"]))
            size = int(5 * (p["life"] / p["max_life"]))
            if size > 0:
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*p["color"], alpha), (size, size), size)
                self.screen.blit(temp_surf, (p["pos"].x - size, p["pos"].y - size), special_flags=pygame.BLEND_RGBA_ADD)

        # Player
        glow_size = self.player_rect.width + 10
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 50), (glow_size // 2, glow_size // 2), glow_size // 2)
        self.screen.blit(glow_surf, (self.player_rect.centerx - glow_size // 2, self.player_rect.centery - glow_size // 2), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect)

    def _render_ui(self):
        # Timer
        time_text = f"TIME: {max(0, self.time_remaining // self.FPS):02d}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Score
        score_text = f"SCORE: {self.score:04d}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_COIN)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "LEVEL COMPLETE" if self.player_rect.colliderect(self.flag_rect) else "GAME OVER"
            text_surf = self.font_game_over.render(msg, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)


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
            "time_remaining": self.time_remaining,
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    print(env.user_guide)

    while not done:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        mov = 0
        if keys[pygame.K_UP]: mov = 1
        # K_DOWN (2) is no-op
        if keys[pygame.K_LEFT]: mov = 3
        if keys[pygame.K_RIGHT]: mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render to the human-visible screen ---
        # Pygame uses (width, height) but our observation is (height, width, 3)
        # So we need to transpose it back for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        clock.tick(env.FPS)
        
    print(f"Game Over! Final Info: {info}")
    pygame.quit()