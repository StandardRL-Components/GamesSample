
# Generated: 2025-08-27T21:17:46.762498
# Source Brief: brief_02742.md
# Brief Index: 2742

        
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
        "Controls: Use arrow keys to jump. Space for a high jump, Shift for a short hop. "
        "Land on platforms and collect coins to score points."
    )

    game_description = (
        "Leap across procedurally generated platforms, collecting coins and managing risk to reach the end of each stage in this side-scrolling arcade hopper."
    )

    auto_advance = True

    # --- Constants ---
    # Game Feel
    GRAVITY = 0.5
    JUMP_POWER_HIGH = -12
    JUMP_POWER_LOW = -7
    JUMP_POWER_STD_V = -9.5
    JUMP_POWER_STD_H = 7
    JUMP_POWER_DOWN_HOP = 3
    AIR_DRAG = 0.98
    PLAYER_SIZE = 20
    FPS = 30 # For auto_advance=True

    # Colors
    COLOR_BG_TOP = (20, 30, 50)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_PLAYER = (100, 255, 100)
    COLOR_PLAYER_OUTLINE = (200, 255, 200)
    COLOR_PLATFORM = (150, 150, 160)
    COLOR_COIN = (255, 223, 0)
    COLOR_END_PLATFORM = (0, 150, 255)
    COLOR_TEXT = (255, 255, 255)
    COLOR_PARTICLE_LAND = (200, 200, 200)
    COLOR_PARTICLE_COIN = (255, 223, 0)

    # World
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    STAGE_HEIGHT_GOAL = -3000 # Vertical distance to clear a stage
    MAX_STAGES = 3
    MAX_TIME_PER_STAGE_SECONDS = 60

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_timer = pygame.font.SysFont("monospace", 24, bold=True)

        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.platforms = None
        self.coins = None
        self.particles = None
        self.score = None
        self.stage = None
        self.stage_timer = None
        self.game_over = None
        self.camera_y = None
        self.max_y_reached = None
        self.last_landed_platform_idx = None
        self.end_platform = None
        self.rng = None
        self.bg_surface = self._create_gradient_background()

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.player_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT - 50.0])
        self.player_vel = np.array([0.0, 0.0])
        self.on_ground = True
        self.last_landed_platform_idx = 0

        self.platforms = []
        self.coins = []
        self.particles = []

        self.score = 0
        self.stage = 1
        self.steps = 0
        self.game_over = False

        self.camera_y = 0.0
        self.max_y_reached = self.player_pos[1]
        
        self._start_stage()

        return self._get_observation(), self._get_info()

    def _start_stage(self):
        self.stage_timer = self.MAX_TIME_PER_STAGE_SECONDS * self.FPS
        self.platforms.clear()
        self.coins.clear()

        # Initial platform
        initial_platform = pygame.Rect(self.SCREEN_WIDTH / 2 - 50, self.SCREEN_HEIGHT - 30, 100, 20)
        self.platforms.append(initial_platform)

        # Procedurally generate platforms for the stage
        y_pos = self.SCREEN_HEIGHT - 100
        while y_pos > self.STAGE_HEIGHT_GOAL - self.SCREEN_HEIGHT:
            self._generate_platform_row(y_pos)
            y_pos -= self.rng.integers(80, 150)

        # Place end platform
        self.end_platform = pygame.Rect(self.SCREEN_WIDTH / 2 - 60, self.STAGE_HEIGHT_GOAL, 120, 30)

    def _generate_platform_row(self, y_pos):
        platform_width = self.rng.integers(
            max(40, 100 - (self.stage - 1) * 15),
            max(60, 150 - (self.stage - 1) * 20)
        )
        platform_x = self.rng.integers(0, self.SCREEN_WIDTH - platform_width)
        new_platform = pygame.Rect(platform_x, y_pos, platform_width, 20)
        self.platforms.append(new_platform)

        # Add coins to the platform
        num_coins = self.rng.choice([0, 1, 1, 2], p=[0.3, 0.4, 0.2, 0.1])
        for _ in range(num_coins):
            coin_x = self.rng.integers(platform_x + 10, platform_x + platform_width - 10)
            coin_y = y_pos - 15
            self.coins.append(np.array([coin_x, coin_y]))

    def step(self, action):
        reward = 0
        
        # --- 1. Handle Action ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if self.on_ground:
            jump_initiated = False
            # Priority: Space > Shift > Arrows
            if space_held:
                self.player_vel = np.array([0.0, self.JUMP_POWER_HIGH])
                jump_initiated = True
            elif shift_held:
                self.player_vel = np.array([0.0, self.JUMP_POWER_LOW])
                jump_initiated = True
            elif movement == 1: # Up
                self.player_vel = np.array([0.0, self.JUMP_POWER_STD_V])
                jump_initiated = True
            elif movement == 2: # Down
                self.player_vel = np.array([0.0, self.JUMP_POWER_DOWN_HOP])
                jump_initiated = True
            elif movement == 3: # Left
                self.player_vel = np.array([-self.JUMP_POWER_STD_H, self.JUMP_POWER_STD_V * 0.8])
                jump_initiated = True
            elif movement == 4: # Right
                self.player_vel = np.array([self.JUMP_POWER_STD_H, self.JUMP_POWER_STD_V * 0.8])
                jump_initiated = True
            
            if jump_initiated:
                self.on_ground = False
                # sfx: jump sound

        # --- 2. Apply Physics ---
        if not self.on_ground:
            self.player_vel[1] += self.GRAVITY
        
        self.player_vel[0] *= self.AIR_DRAG
        self.player_pos += self.player_vel

        # Screen bounds
        if self.player_pos[0] < 0:
            self.player_pos[0] = 0
            self.player_vel[0] = 0
        if self.player_pos[0] > self.SCREEN_WIDTH - self.PLAYER_SIZE:
            self.player_pos[0] = self.SCREEN_WIDTH - self.PLAYER_SIZE
            self.player_vel[0] = 0

        # --- 3. Collision Detection ---
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)

        # Platforms
        if self.player_vel[1] > 0: # Only check for landing if moving down
            self.on_ground = False
            for i, plat in enumerate(self.platforms):
                if player_rect.colliderect(plat) and player_rect.bottom < plat.bottom:
                    self.player_pos[1] = plat.top - self.PLAYER_SIZE
                    self.player_vel = np.array([0.0, 0.0])
                    self.on_ground = True
                    self._create_particles(self.player_pos + np.array([self.PLAYER_SIZE/2, self.PLAYER_SIZE]), 10, self.COLOR_PARTICLE_LAND)
                    # sfx: land sound

                    # Reward for new height
                    if self.player_pos[1] < self.max_y_reached:
                        reward += 5
                        self.max_y_reached = self.player_pos[1]
                    
                    # Penalty for staying on adjacent platforms
                    if abs(i - self.last_landed_platform_idx) <= 1 and i != self.last_landed_platform_idx:
                        reward -= 0.2

                    self.last_landed_platform_idx = i
                    break
        
        # End Platform
        if not self.game_over and self.player_vel[1] > 0 and player_rect.colliderect(self.end_platform):
            reward += 100
            self.score += 100
            self.stage += 1
            if self.stage > self.MAX_STAGES:
                self.game_over = True
            else:
                self._start_stage()
                self.player_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT - 50.0])
                self.on_ground = True
                self.last_landed_platform_idx = 0
                self.max_y_reached = self.player_pos[1]
                # sfx: stage complete fanfare

        # Coins
        collected_coins = []
        for i, coin_pos in enumerate(self.coins):
            coin_rect = pygame.Rect(coin_pos[0] - 5, coin_pos[1] - 5, 10, 10)
            if player_rect.colliderect(coin_rect):
                collected_coins.append(i)
                reward += 1
                self.score += 1
                self._create_particles(coin_pos, 8, self.COLOR_PARTICLE_COIN)
                # sfx: coin collect sound
        
        # Remove collected coins (in reverse to not mess up indices)
        for i in sorted(collected_coins, reverse=True):
            self.coins.pop(i)

        # --- 4. Update Game State ---
        self.steps += 1
        self.stage_timer -= 1
        self._update_particles()
        
        # Smooth camera follow
        target_cam_y = self.player_pos[1] - self.SCREEN_HEIGHT * 0.5
        self.camera_y += (target_cam_y - self.camera_y) * 0.08

        # --- 5. Check Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.player_pos[1] > self.camera_y + self.SCREEN_HEIGHT + 50:
                reward = -100 # Fell off screen
                # sfx: fall/fail sound
            elif self.stage_timer <= 0:
                reward = -50 # Timed out
                # sfx: timeout sound
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        # Fall off bottom of screen
        if self.player_pos[1] > self.camera_y + self.SCREEN_HEIGHT + 50:
            return True
        # Timer runs out
        if self.stage_timer <= 0:
            return True
        # Completed all stages
        if self.game_over:
            return True
        return False
    
    def _get_info(self):
        return {
            "score": self.score,
            "stage": self.stage,
            "steps": self.steps,
            "time_left": max(0, self.stage_timer // self.FPS)
        }

    def _get_observation(self):
        self.screen.blit(self.bg_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _create_gradient_background(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio),
            )
            pygame.draw.line(bg, color, (0, y), (self.SCREEN_WIDTH, y))
        return bg

    def _render_game(self):
        # All rendering is offset by camera_y
        cam_offset = int(self.camera_y)

        # Render End Platform
        ep_rect = self.end_platform.move(0, -cam_offset)
        for i in range(10, 0, -2):
            glow_color = (*self.COLOR_END_PLATFORM, i * 10)
            glow_rect = ep_rect.inflate(i, i)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, glow_color, s.get_rect(), border_radius=8)
            self.screen.blit(s, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_END_PLATFORM, ep_rect, border_radius=8)

        # Render Platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat.move(0, -cam_offset), border_radius=4)
        
        # Render Coins
        for coin_pos in self.coins:
            x, y = int(coin_pos[0]), int(coin_pos[1] - cam_offset)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 7, self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, x, y, 7, self.COLOR_COIN)

        # Render Particles
        for p in self.particles:
            pos = p["pos"]
            color = (*p["color"], p["alpha"])
            size = int(p["size"])
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size, size), size)
                self.screen.blit(s, (int(pos[0] - size - 0), int(pos[1] - size - cam_offset)))

        # Render Player
        player_screen_pos = (int(self.player_pos[0]), int(self.player_pos[1] - cam_offset))
        player_rect = pygame.Rect(*player_screen_pos, self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect.inflate(4, 4), border_radius=6)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=5)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"COINS: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Stage
        stage_text = self.font_ui.render(f"STAGE: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.SCREEN_WIDTH - stage_text.get_width() - 10, 10))
        
        # Timer
        time_left = max(0, self.stage_timer // self.FPS)
        timer_color = (255, 100, 100) if time_left < 10 else self.COLOR_TEXT
        timer_text = self.font_timer.render(f"{time_left}", True, timer_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH / 2 - timer_text.get_width() / 2, 8))

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 2 + 1
            self.particles.append({
                "pos": pos.copy(),
                "vel": np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                "lifetime": self.rng.integers(15, 25),
                "size": self.rng.random() * 3 + 2,
                "color": color,
                "alpha": 255
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"][1] += 0.1 # particle gravity
            p["lifetime"] -= 1
            p["alpha"] = max(0, int(255 * (p["lifetime"] / 25)))
            if p["lifetime"] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Arcade Hopper")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    while running:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        move_action = 0 # None
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move_action, space_action, shift_action]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}. Press 'R' to restart.")

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it.
        # Pygame uses (width, height), but our obs is (height, width, 3).
        # We also need to convert from the numpy array format back to a Pygame Surface.
        # The original `env.screen` is what we want to display.
        surf = pygame.transform.rotate(pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2))), -90)
        surf = pygame.transform.flip(surf, True, False)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()