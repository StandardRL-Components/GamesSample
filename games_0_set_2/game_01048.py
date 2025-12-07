import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Collect coins and reach the flag at the end!"
    )

    game_description = (
        "A fast-paced, procedurally generated platformer. Jump between platforms, "
        "collect coins, and race to the end before the timer runs out."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2000
        self.LEVEL_WIDTH_PIXELS = 10000

        # Colors
        self.COLOR_BG_TOP = (50, 50, 80)
        self.COLOR_BG_BOTTOM = (20, 20, 40)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_PLAYER_GLOW = (255, 150, 150)
        self.COLOR_PLATFORM = (139, 69, 19)
        self.COLOR_PLATFORM_TOP = (160, 82, 45)
        self.COLOR_COIN = (255, 223, 0)
        self.COLOR_BONUS_COIN = (255, 120, 0)
        self.COLOR_DANGER = (255, 0, 0)
        self.COLOR_FLAG = (255, 255, 255)
        self.COLOR_FLAGPOLE = (200, 200, 200)
        self.COLOR_TEXT = (255, 255, 255)

        # Physics constants
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -15
        self.PLAYER_ACCEL = 1.2
        self.PLAYER_FRICTION = 0.85
        self.PLAYER_MAX_SPEED_BASE = 8.0
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 36)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.player_size = None
        self.on_ground = None
        self.camera_x = None
        self.platforms = None
        self.coins = None
        self.danger_zones = None
        self.particles = None
        self.steps = None
        self.score = None
        self.lives = None
        self.game_over = None
        self.level_end_x = None
        self.player_max_speed = None
        self.platform_gap = None
        self.np_random = None

        # This will call reset, which needs np_random to be initialized by super().reset()
        # We call reset separately after __init__ is complete or handle it carefully.
        # For validation purposes during init, we ensure a seed is used.
        self.reset(seed=0)
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([100.0, 100.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_size = np.array([24, 24])
        self.on_ground = False
        
        self.camera_x = 0.0
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.level_end_reached = False
        
        self.player_max_speed = self.PLAYER_MAX_SPEED_BASE
        self.platform_gap = 20

        self.particles = deque()
        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms = []
        self.coins = []
        self.danger_zones = []

        # Start platform
        self.platforms.append(pygame.Rect(50, 250, 200, 20))
        
        current_x = 250
        min_y, max_y = 150, 350
        last_y = 250

        while current_x < self.LEVEL_WIDTH_PIXELS - 500:
            gap = self.np_random.integers(20, 80)
            current_x += gap
            
            width = self.np_random.integers(80, 250)
            height = 20
            
            # Ensure next platform is reachable
            y_delta = self.np_random.integers(-80, 80)
            y = np.clip(last_y + y_delta, min_y, max_y)
            
            platform_rect = pygame.Rect(current_x, y, width, height)
            self.platforms.append(platform_rect)
            
            # Add coins
            num_coins = self.np_random.integers(1, int(width / 50) + 2)
            for i in range(num_coins):
                cx = current_x + (i + 1) * (width / (num_coins + 1))
                cy = y - 30
                is_bonus = self.np_random.random() < 0.1 # 10% chance for a bonus coin
                self.coins.append({'rect': pygame.Rect(cx, cy, 12, 12), 'type': 'bonus' if is_bonus else 'regular'})

            # Add danger zones
            if self.np_random.random() < 0.2: # 20% chance for a danger zone
                dz_width = self.np_random.integers(30, int(width * 0.8))
                dz_x = current_x + self.np_random.integers(0, width - dz_width)
                self.danger_zones.append(pygame.Rect(dz_x, y - 5, dz_width, 5))

            last_y = y
            current_x += width

        # End flag
        self.level_end_x = current_x + 100
        self.platforms.append(pygame.Rect(self.level_end_x - 50, last_y, 100, 20))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        self.steps += 1
        
        # --- Player Control ---
        if movement == 3:  # Left
            self.player_vel[0] -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel[0] += self.PLAYER_ACCEL
        
        if movement == 1 and self.on_ground:  # Jump
            self.player_vel[1] = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump_sound

        # --- Physics and Collisions ---
        # Apply friction and cap speed
        self.player_vel[0] *= self.PLAYER_FRICTION
        self.player_vel[0] = np.clip(self.player_vel[0], -self.player_max_speed, self.player_max_speed)
        if abs(self.player_vel[0]) < 0.1: self.player_vel[0] = 0

        # Apply gravity
        self.player_vel[1] += self.GRAVITY
        
        # Store old position for reward calculation
        old_player_x = self.player_pos[0]
        
        # Move player
        self.player_pos += self.player_vel
        
        # Reward for horizontal progress
        reward += (self.player_pos[0] - old_player_x) * 0.01

        # Prevent moving off left edge of world
        self.player_pos[0] = max(0, self.player_pos[0])
        
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.player_size[0], self.player_size[1])

        # Platform collisions
        self.on_ground = False
        for plat in self.platforms:
            if player_rect.colliderect(plat):
                # Check if player was above the platform in the previous step
                if self.player_vel[1] > 0 and (player_rect.bottom - self.player_vel[1]) <= plat.top:
                    self.player_pos[1] = plat.top - self.player_size[1]
                    self.player_vel[1] = 0
                    self.on_ground = True
                    # sfx: land_sound
                    break
        
        # --- Collectibles and Hazards ---
        for coin in self.coins[:]:
            if player_rect.colliderect(coin['rect']):
                if coin['type'] == 'bonus':
                    self.score += 5
                    reward += 5
                else:
                    self.score += 1
                    reward += 1
                self.coins.remove(coin)
                # sfx: coin_collect_sound
                self._create_particles(coin['rect'].center, self.COLOR_COIN, 10)
        
        for danger in self.danger_zones:
            if player_rect.colliderect(danger):
                reward -= 0.1
                # sfx: damage_sound
                
        # --- Update World ---
        # Camera follows player smoothly
        self.camera_x += (self.player_pos[0] - self.camera_x - self.WIDTH / 3) * 0.1
        self.camera_x = max(0, self.camera_x)

        # Update particles
        self._update_particles()
        
        # --- Difficulty Scaling ---
        if self.steps % 200 == 0:
            self.player_max_speed += 0.05
        if self.steps % 500 == 0:
            self.platform_gap = min(50, self.platform_gap + 1)
            
        # --- Termination Conditions ---
        terminated = False
        if self.player_pos[1] > self.HEIGHT: # Fell off screen
            self.lives -= 1
            reward = -50
            # sfx: fall_sound
            if self.lives <= 0:
                self.game_over = True
                terminated = True
            else:
                self._reset_player_position()

        if player_rect.right > self.level_end_x: # Reached end
            self.score += 50
            reward += 50
            self.game_over = True
            terminated = True
            self.level_end_reached = True
            # sfx: level_complete_sound
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _reset_player_position(self):
        self.player_pos = np.array([self.camera_x + 50, 100.0])
        self.player_vel = np.array([0.0, 0.0])
        self._create_particles(self.player_pos, self.COLOR_PLAYER, 20)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            vel = [self.np_random.uniform(-3, 3), self.np_random.uniform(-5, 1)]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        for p in list(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.2 # Particle gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Draw background gradient
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw platforms
        for plat in self.platforms:
            screen_rect = plat.move(-self.camera_x, 0)
            if screen_rect.right < 0 or screen_rect.left > self.WIDTH:
                continue
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, screen_rect)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_TOP, (screen_rect.x, screen_rect.y, screen_rect.width, 4))
        
        # Draw danger zones (flashing)
        flash_alpha = 128 + 127 * math.sin(self.steps * 0.3)
        danger_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        for danger in self.danger_zones:
            screen_rect = danger.move(-self.camera_x, 0)
            if screen_rect.right < 0 or screen_rect.left > self.WIDTH:
                continue
            color = (*self.COLOR_DANGER, flash_alpha)
            pygame.draw.rect(danger_surface, color, screen_rect)
        self.screen.blit(danger_surface, (0,0))
        
        # Draw coins (rotating/bobbing)
        for coin in self.coins:
            screen_rect = coin['rect'].move(-self.camera_x, 0)
            if screen_rect.right < 0 or screen_rect.left > self.WIDTH:
                continue
            
            bob_offset = math.sin(self.steps * 0.1 + screen_rect.x * 0.1) * 3
            scale_x = abs(math.cos(self.steps * 0.15 + screen_rect.x * 0.1))
            
            color = self.COLOR_BONUS_COIN if coin['type'] == 'bonus' else self.COLOR_COIN
            width = int(coin['rect'].width * scale_x)
            
            pygame.draw.ellipse(self.screen, color, 
                (screen_rect.centerx - width // 2, screen_rect.y + bob_offset, width, coin['rect'].height))
        
        # Draw end flag
        pole_x = self.level_end_x - self.camera_x
        flag_y = self.platforms[-1].y
        if 0 < pole_x < self.WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FLAGPOLE, (pole_x, flag_y), (pole_x, flag_y - 60), 3)
            flag_points = [(pole_x, flag_y - 60), (pole_x + 40, flag_y - 50), (pole_x, flag_y - 40)]
            pygame.gfxdraw.aapolygon(self.screen, flag_points, self.COLOR_FLAG)
            pygame.gfxdraw.filled_polygon(self.screen, flag_points, self.COLOR_FLAG)

        # Draw particles
        for p in self.particles:
            screen_pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1]))
            size = max(1, int(p['life'] / 6))
            pygame.draw.circle(self.screen, p['color'], screen_pos, size)
            
        # Draw player
        if self.lives > 0:
            player_screen_pos = (self.player_pos[0] - self.camera_x, self.player_pos[1])
            player_screen_rect = pygame.Rect(player_screen_pos, self.player_size)
            
            # Glow effect
            glow_rect = player_screen_rect.inflate(8, 8)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*self.COLOR_PLAYER_GLOW, 50), glow_surf.get_rect(), border_radius=8)
            # FIX: Create a pygame.Rect object before calling inflate. The most likely intent
            # was to create a smaller, centered rectangle inside the glow surface.
            inner_glow_rect = glow_surf.get_rect().inflate(-4, -4)
            pygame.draw.rect(glow_surf, (*self.COLOR_PLAYER_GLOW, 80), inner_glow_rect, border_radius=6)
            self.screen.blit(glow_surf, glow_rect.topleft)
            
            # Main body
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_screen_rect, border_radius=4)
            
    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

        if self.game_over:
            if self.level_end_reached:
                msg = "LEVEL COMPLETE!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "player_pos_x": self.player_pos[0],
            "camera_x": self.camera_x,
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # Unset the dummy video driver if we want to play manually
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Pygame setup for human play
    pygame.display.set_caption("Procedural Platformer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    # Game loop
    while not terminated:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        # No down action in this game
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Event handling
        quit_game = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_game = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False # ensure loop continues after reset

        if quit_game:
            break

        clock.tick(env.FPS)
        
        if terminated:
            print(f"Game Over. Final Info: {info}")
            # Wait a bit before closing or restarting
            pygame.time.wait(2000)
            # For continuous play, you could reset here
            obs, info = env.reset()
            terminated = False

    env.close()