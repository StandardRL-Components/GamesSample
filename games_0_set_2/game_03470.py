import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to jump sideways. Space for a long jump. Shift for a high jump. Combine for a long, high jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop between procedurally generated platforms, collect coins, and reach the end flag in this side-scrolling arcade hopper."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 23, 42)
    COLOR_PLAYER = (5, 255, 150)
    COLOR_PLAYER_SHADOW = (2, 130, 76)
    COLOR_PLATFORM = (148, 163, 184)
    COLOR_PLATFORM_SHADOW = (71, 85, 105)
    COLOR_COIN = (250, 204, 21)
    COLOR_COIN_SHADOW = (161, 98, 7)
    COLOR_FLAG = (239, 68, 68)
    COLOR_FLAGPOLE = (203, 213, 225)
    COLOR_TEXT = (226, 232, 240)
    
    # Physics
    GRAVITY = 0.4
    PLAYER_SIZE = pygame.Vector2(20, 20)
    
    # Jumping
    JUMP_BASE_POWER = 3.0  # Base hop when on ground
    JUMP_SIDE_POWER = 3.0
    JUMP_UP_POWER = 10.0
    JUMP_HIGH_BONUS = 3.0
    JUMP_LONG_MULTIPLIER = 1.8
    MAX_VELOCITY_Y = 15
    
    # Level Generation
    NUM_STAGES = 3
    PLATFORMS_PER_STAGE = 50
    LEVEL_END_PADDING = 1000
    
    # Rewards
    REWARD_SURVIVAL = 0.01
    REWARD_COIN = 1.0
    REWARD_WIN = 100.0
    REWARD_FALL = -10.0
    REWARD_HIGH_JUMP = 0.5
    REWARD_LOW_JUMP = -0.2
    
    # Game State
    MAX_STEPS = 5000

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
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Game state variables
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_on_ground = False
        self.player_squash = 1.0 # For animation
        
        self.platforms = []
        self.coins = []
        self.particles = []
        self.stars = []
        
        self.flag_pos = pygame.Vector2(0, 0)
        
        self.camera_offset_x = 0.0
        self.last_platform_y = 0

        self.steps = 0
        self.score = 0
        self.stage = 0
        self.game_over = False
        
        self.np_random = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize RNG
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        # Reset state
        self.steps = 0
        self.score = 0
        self.stage = 1
        self.game_over = False
        self.camera_offset_x = 0.0
        
        # Clear dynamic objects
        self.platforms.clear()
        self.coins.clear()
        self.particles.clear()
        
        # Generate level
        self._generate_stars()
        self._generate_level()
        
        # Reset player
        start_platform = self.platforms[0]
        self.player_pos = pygame.Vector2(start_platform.centerx, start_platform.top - self.PLAYER_SIZE.y)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_on_ground = True
        self.last_platform_y = start_platform.y
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = self.REWARD_SURVIVAL
        self.steps += 1
        
        # --- Update Game Logic ---
        reward += self._update_player(action)
        reward += self._update_coins()
        self._update_particles()
        self._update_camera()

        # --- Check Termination ---
        terminated = False
        truncated = False
        if self.player_pos.y > self.SCREEN_HEIGHT + 50: # Fell off screen
            self.game_over = True
            terminated = True
            reward = self.REWARD_FALL
        
        player_rect = self._get_player_rect()
        flag_rect = pygame.Rect(self.flag_pos.x, self.flag_pos.y - 60, 10, 60)
        if player_rect.colliderect(flag_rect): # Reached flag
            self.game_over = True
            terminated = True
            reward = self.REWARD_WIN
            self.score += 50 # Bonus points for winning
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            truncated = True

        # In auto-advance mode, we can tick the clock here.
        if self.auto_advance:
            self.clock.tick(30)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self, action):
        landing_reward = 0
        
        # 1. Apply jump action if on ground
        if self.player_on_ground:
            self.player_on_ground = False
            self.player_squash = 1.5 # Stretch for jump
            # sound: jump.wav
            self._create_jump_particles(self.player_pos + pygame.Vector2(self.PLAYER_SIZE.x / 2, self.PLAYER_SIZE.y))

            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            jump_y = self.JUMP_BASE_POWER
            jump_x = 0

            if shift_held:
                jump_y += self.JUMP_HIGH_BONUS

            if movement == 1: # Up
                jump_y = self.JUMP_UP_POWER
                if shift_held: jump_y += self.JUMP_HIGH_BONUS / 2
            elif movement == 3: # Left
                jump_x = -self.JUMP_SIDE_POWER
            elif movement == 4: # Right
                jump_x = self.JUMP_SIDE_POWER
            
            if space_held:
                jump_x *= self.JUMP_LONG_MULTIPLIER

            self.player_vel.y = -jump_y
            self.player_vel.x = jump_x
            
        # 2. Apply physics
        self.player_vel.y += self.GRAVITY
        self.player_vel.y = min(self.player_vel.y, self.MAX_VELOCITY_Y)
        self.player_pos += self.player_vel
        
        # 3. Check for platform collisions
        player_rect = self._get_player_rect()
        for platform in self.platforms:
            # Check if player is roughly above the platform and falling
            if player_rect.colliderect(platform) and self.player_vel.y > 0:
                # Check if player's bottom was above platform top in previous frame
                if (player_rect.bottom - self.player_vel.y) <= platform.top:
                    self.player_pos.y = platform.top - self.PLAYER_SIZE.y
                    self.player_vel.y = 0
                    self.player_vel.x *= 0.8 # Friction
                    self.player_on_ground = True
                    self.player_squash = 0.7 # Squash on land
                    # sound: land.wav
                    self._create_jump_particles(self.player_pos + pygame.Vector2(self.PLAYER_SIZE.x / 2, self.PLAYER_SIZE.y))
                    
                    # Calculate landing reward
                    if platform.y < self.last_platform_y - 20:
                        landing_reward = self.REWARD_HIGH_JUMP
                    elif platform.y > self.last_platform_y + 20:
                        landing_reward = self.REWARD_LOW_JUMP
                    self.last_platform_y = platform.y
                    break
        
        # Update player animation
        self.player_squash += (1.0 - self.player_squash) * 0.2

        return landing_reward

    def _update_coins(self):
        collected_reward = 0
        player_rect = self._get_player_rect()
        for coin in self.coins[:]:
            coin_rect = pygame.Rect(coin[0] - 5, coin[1] - 5, 10, 10)
            if player_rect.colliderect(coin_rect):
                self.coins.remove(coin)
                self.score += 1
                collected_reward += self.REWARD_COIN
                # sound: coin.wav
                for _ in range(10):
                    self.particles.append(self._create_particle(pygame.Vector2(coin), self.COLOR_COIN))
        return collected_reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _update_camera(self):
        target_camera_x = self.player_pos.x - self.SCREEN_WIDTH / 3
        self.camera_offset_x += (target_camera_x - self.camera_offset_x) * 0.1

    def _generate_level(self):
        current_x = 100
        last_y = self.SCREEN_HEIGHT - 50
        
        # Start platform
        self.platforms.append(pygame.Rect(50, last_y, 150, 40))

        for stage in range(1, self.NUM_STAGES + 1):
            # Difficulty scaling
            min_gap = 40 - stage * 5
            max_gap = 100 - stage * 10
            min_height_diff = - (30 + stage * 15)
            max_height_diff = (20 + stage * 10)
            
            for _ in range(self.PLATFORMS_PER_STAGE):
                gap = self.np_random.integers(min_gap, max_gap)
                width = self.np_random.integers(60, 120)
                height_diff = self.np_random.integers(min_height_diff, max_height_diff)
                
                current_x += gap + width
                current_y = np.clip(last_y + height_diff, self.SCREEN_HEIGHT * 0.3, self.SCREEN_HEIGHT - 40)
                
                platform_rect = pygame.Rect(current_x, current_y, width, self.SCREEN_HEIGHT - current_y)
                self.platforms.append(platform_rect)
                
                # Add a coin?
                if self.np_random.random() < 0.5:
                    self.coins.append(pygame.Vector2(platform_rect.centerx, platform_rect.top - 20))
                
                last_y = current_y

        # End flag
        self.flag_pos = pygame.Vector2(self.platforms[-1].centerx, self.platforms[-1].top)

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            x = self.np_random.integers(0, self.SCREEN_WIDTH)
            y = self.np_random.integers(0, self.SCREEN_HEIGHT)
            size = self.np_random.choice([1, 2, 3])
            self.stars.append((x, y, size))

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background()
        self._render_platforms()
        self._render_coins()
        self._render_flag()
        self._render_particles()
        self._render_player()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": (self.player_pos.x, self.player_pos.y),
            "player_vel": (self.player_vel.x, self.player_vel.y),
        }

    def _render_background(self):
        for x, y, size in self.stars:
            # Parallax effect
            screen_x = (x - self.camera_offset_x * (0.1 * size)) % self.SCREEN_WIDTH
            color_val = 50 * size
            pygame.draw.rect(self.screen, (color_val, color_val, color_val + 20), (screen_x, y, size, size))

    def _render_platforms(self):
        for p in self.platforms:
            screen_rect = p.move(-self.camera_offset_x, 0)
            if screen_rect.right < 0 or screen_rect.left > self.SCREEN_WIDTH:
                continue
            
            # Draw shadow/3D effect
            shadow_rect = screen_rect.copy()
            shadow_rect.height = 10
            shadow_rect.top = screen_rect.top
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_SHADOW, shadow_rect)
            
            # Draw main platform body
            main_rect = screen_rect.copy()
            main_rect.top = shadow_rect.bottom
            main_rect.height = screen_rect.height - shadow_rect.height
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, main_rect)
            
    def _render_coins(self):
        # Animate coin pulse
        pulse = abs(math.sin(self.steps * 0.2)) * 2
        for coin_pos in self.coins:
            screen_x = coin_pos.x - self.camera_offset_x
            if 0 < screen_x < self.SCREEN_WIDTH:
                size = int(6 + pulse)
                shadow_offset = int(size * 0.2)
                
                # Shadow
                pygame.gfxdraw.filled_circle(self.screen, int(screen_x), int(coin_pos.y + shadow_offset), size, self.COLOR_COIN_SHADOW)
                # Main coin
                pygame.gfxdraw.filled_circle(self.screen, int(screen_x), int(coin_pos.y), size, self.COLOR_COIN)
                # Shine
                pygame.gfxdraw.filled_circle(self.screen, int(screen_x - size*0.3), int(coin_pos.y - size*0.3), int(size*0.3), (255, 255, 255, 150))
                
    def _render_flag(self):
        screen_x = self.flag_pos.x - self.camera_offset_x
        screen_y = self.flag_pos.y
        if 0 < screen_x < self.SCREEN_WIDTH:
            # Pole
            pygame.draw.rect(self.screen, self.COLOR_FLAGPOLE, (screen_x, screen_y - 60, 5, 60))
            # Flag
            flag_points = [(screen_x + 5, screen_y - 60), (screen_x + 45, screen_y - 50), (screen_x + 5, screen_y - 40)]
            pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_points)

    def _render_player(self):
        h = self.PLAYER_SIZE.y * self.player_squash
        w = self.PLAYER_SIZE.x / self.player_squash
        y_offset = (self.PLAYER_SIZE.y - h)
        
        rect = pygame.Rect(
            self.player_pos.x - self.camera_offset_x,
            self.player_pos.y + y_offset,
            w, h
        )
        # Shadow
        shadow_rect = rect.copy()
        shadow_rect.height = max(4, rect.height * 0.3)
        shadow_rect.bottom = rect.bottom
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_SHADOW, shadow_rect, border_radius=4)
        # Body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=4)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, (p['lifespan'] / p['max_lifespan']) * 255)
            color = p['color'] + (int(alpha),)
            size = int(p['size'] * (1 - (p['lifespan'] / p['max_lifespan'])))
            pos = (int(p['pos'].x - self.camera_offset_x), int(p['pos'].y))
            
            if size > 0:
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (pos[0]-size, pos[1]-size))
            
    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 40))

    def _get_player_rect(self):
        return pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE.x, self.PLAYER_SIZE.y)

    def _create_particle(self, pos, color):
        return {
            'pos': pygame.Vector2(pos),
            'vel': pygame.Vector2(self.np_random.uniform(-1.5, 1.5), self.np_random.uniform(-1.5, 1.5)),
            'lifespan': self.np_random.integers(15, 30),
            'max_lifespan': 30,
            'size': self.np_random.integers(3, 7),
            'color': color
        }
    
    def _create_jump_particles(self, pos):
        for _ in range(5):
            p = self._create_particle(pos, self.COLOR_PLAYER)
            p['vel'].y = self.np_random.uniform(0.1, 0.8) # Downward burst
            self.particles.append(p)
            
    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    # It will not run in a headless environment
    try:
        del os.environ["SDL_VIDEODRIVER"]
    except KeyError:
        pass

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Arcade Hopper")
    
    terminated = False
    truncated = False
    running = True
    total_reward = 0
    
    # Disable auto-advance for human play
    env.auto_advance = False
    clock = pygame.time.Clock()

    while running:
        # --- Human Controls ---
        movement = 0 # none
        space_held = 0
        shift_held = 0
        
        # Poll for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    terminated = False
                    truncated = False
                    total_reward = 0
                if event.key == pygame.K_q:
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
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        if terminated or truncated:
            # Display game over message
            msg = "GAME OVER" if terminated else "TIME UP"
            font = pygame.font.SysFont("monospace", 48, bold=True)
            text = font.render(msg, True, (255, 0, 0))
            text_rect = text.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2 - 30))
            screen.blit(text, text_rect)
            
            font_small = pygame.font.SysFont("monospace", 24)
            text_restart = font_small.render("Press 'R' to restart", True, (255, 255, 255))
            restart_rect = text_restart.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2 + 20))
            screen.blit(text_restart, restart_rect)
        
        pygame.display.flip()
        clock.tick(30)

    env.close()