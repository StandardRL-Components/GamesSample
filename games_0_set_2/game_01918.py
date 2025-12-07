# Generated: 2025-08-27T18:42:26.567042
# Source Brief: brief_01918.md
# Brief Index: 1918

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to run, ↑ to jump. Collect coins and reach the green flag."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced pixel art platformer. Navigate procedurally generated levels, "
        "collect coins for points, and reach the end as quickly as possible. "
        "Avoid falling into the pits!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Screen
    WIDTH, HEIGHT = 640, 400
    # Colors
    COLOR_BG = (135, 206, 235)  # Sky Blue
    COLOR_PLATFORM = (139, 69, 19)  # Brown
    COLOR_COIN = (255, 215, 0)  # Gold
    COLOR_PIT = (178, 34, 34)  # Firebrick Red
    COLOR_FLAG = (34, 139, 34)  # Forest Green
    COLOR_PLAYER = (70, 130, 180)  # Steel Blue
    COLOR_PARTICLE = (255, 255, 102) # Light Yellow
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)
    # Physics
    GRAVITY = 0.4
    JUMP_STRENGTH = 9.5
    MOVE_ACCEL = 0.6
    MAX_SPEED = 5
    FRICTION = 0.85
    # Player
    PLAYER_SIZE = pygame.Vector2(24, 24)
    # Level
    LEVEL_END_X = 4000
    MAX_STEPS = 2000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_large = pygame.font.Font(pygame.font.get_default_font(), 24)
        self.font_small = pygame.font.Font(pygame.font.get_default_font(), 16)
        
        # Game state that persists across episodes
        self.level = 1

        # Initialize other state variables
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_on_ground = False
        self.coyote_frames = 0
        self.jump_buffer_frames = 0
        self.won_level = False

        self.platforms = []
        self.coins = []
        self.pits = []
        self.flag_rect = None
        self.particles = []

        self.camera_x = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.np_random = np.random.default_rng()
        
        # Initialize state variables
        # self.reset() is called by the test runner, no need to call it here.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.won_level:
            self.level += 1
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.won_level = False

        self._generate_level()

        self.player_pos = pygame.Vector2(100, 200)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_on_ground = False
        self.coyote_frames = 0
        self.jump_buffer_frames = 0
        
        self.camera_x = 0
        self.particles = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms.clear()
        self.coins.clear()
        self.pits.clear()

        # --- Difficulty scaling ---
        # Level 1: 10% pits, 50px max gap. Level 10: 55% pits, 95px max gap
        pit_chance = min(0.6, 0.1 + (self.level - 1) * 0.05) 
        max_gap = min(120, 50 + (self.level - 1) * 5)

        # Start platform
        start_platform = pygame.Rect(0, self.HEIGHT - 80, 250, 80)
        self.platforms.append(start_platform)

        current_x = start_platform.right
        last_y = start_platform.top

        while current_x < self.LEVEL_END_X:
            gap = self.np_random.integers(30, max_gap + 1)
            
            is_pit = self.np_random.random() < pit_chance
            if is_pit:
                # Create a pit
                pit_width = self.np_random.integers(80, 150)
                self.pits.append(pygame.Rect(current_x, self.HEIGHT-10, pit_width, 20))
                current_x += pit_width
                continue

            current_x += gap
            
            plat_y = np.clip(
                last_y + self.np_random.integers(-80, 80),
                self.HEIGHT / 2,
                self.HEIGHT - 40,
            )
            plat_width = self.np_random.integers(100, 300)
            plat_height = self.HEIGHT - plat_y
            
            new_platform = pygame.Rect(current_x, plat_y, plat_width, plat_height)
            self.platforms.append(new_platform)

            # Place coins
            num_coins = self.np_random.integers(0, 4)
            for i in range(num_coins):
                coin_x = new_platform.left + (new_platform.width / (num_coins + 1)) * (i + 1)
                coin_y_offset = self.np_random.integers(30, 100)
                self.coins.append(pygame.Rect(coin_x, new_platform.top - coin_y_offset, 15, 15))

            last_y = plat_y
            current_x = new_platform.right
        
        # End flag
        last_platform = self.platforms[-1]
        self.flag_rect = pygame.Rect(
            last_platform.centerx - 15, last_platform.top - 60, 30, 60
        )

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean, unused
        shift_held = action[2] == 1  # Boolean, unused
        
        reward = 0
        terminated = False
        
        # --- Player Input & Game Feel ---
        # Jump buffering
        if movement == 1: # up
            self.jump_buffer_frames = 5
        else:
            self.jump_buffer_frames = max(0, self.jump_buffer_frames - 1)

        # Coyote time
        if self.player_on_ground:
            self.coyote_frames = 7
        else:
            self.coyote_frames = max(0, self.coyote_frames - 1)

        # Jump action
        if self.jump_buffer_frames > 0 and self.coyote_frames > 0:
            self.player_vel.y = -self.JUMP_STRENGTH
            self.coyote_frames = 0
            self.jump_buffer_frames = 0
            # sfx: jump

        # Horizontal movement
        old_pos_x = self.player_pos.x
        if movement == 3:  # Left
            self.player_vel.x -= self.MOVE_ACCEL
        elif movement == 4:  # Right
            self.player_vel.x += self.MOVE_ACCEL
        
        # --- Physics Update ---
        self.player_vel.x *= self.FRICTION
        self.player_vel.x = np.clip(self.player_vel.x, -self.MAX_SPEED, self.MAX_SPEED)
        self.player_vel.y += self.GRAVITY
        self.player_vel.y = min(15, self.player_vel.y) # Terminal velocity

        self.player_pos.x += self.player_vel.x
        self.player_pos.y += self.player_vel.y

        # --- Collisions ---
        self.player_on_ground = False

        # Using float-based collision for platforms to avoid truncation errors with pygame.Rect
        px, py = self.player_pos
        pw, ph = self.PLAYER_SIZE
        for plat in self.platforms:
            # AABB collision check with float coordinates
            if (px < plat.right and px + pw > plat.left and
                py < plat.bottom and py + ph > plat.top):
                
                # Check if landing on top
                prev_bottom = py + ph - self.player_vel.y
                if self.player_vel.y > 0 and prev_bottom <= plat.top:
                    self.player_pos.y = plat.top - ph
                    self.player_vel.y = 0
                    self.player_on_ground = True
                    break # Stop checking platforms once on the ground

        # Create an integer Rect for other collisions (coins, hazards, etc.)
        player_rect = pygame.Rect(self.player_pos, self.PLAYER_SIZE)
        
        # --- Collectibles & Hazards ---
        collected_coins = []
        for coin in self.coins:
            if player_rect.colliderect(coin):
                collected_coins.append(coin)
                self.score += 1
                reward += 1
                # sfx: coin_collect
                for _ in range(10):
                    self.particles.append(Particle(coin.center, self.np_random))
        self.coins = [c for c in self.coins if c not in collected_coins]

        # Pits
        if player_rect.top > self.HEIGHT:
            terminated = True
            reward = -10
            # sfx: fall_death
        
        # Flag
        if self.flag_rect and player_rect.colliderect(self.flag_rect):
            level_bonus = 10 * self.level
            terminated = True
            self.won_level = True
            reward = 50 + level_bonus
            self.score += 50 + level_bonus
            # sfx: level_complete

        # --- Reward Shaping ---
        moved_forward = self.player_pos.x - old_pos_x
        reward += moved_forward * 0.1
        
        if abs(self.player_vel.x) < 0.1 and self.player_on_ground:
            reward -= 0.01

        # --- Update Game State ---
        self.steps += 1
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
             terminated = True # For many algos, truncated is also terminal
        
        self.game_over = terminated or truncated

        self._update_camera()
        self._update_particles()
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )
    
    def _update_camera(self):
        target_camera_x = self.player_pos.x - self.WIDTH / 3
        self.camera_x += (target_camera_x - self.camera_x) * 0.1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

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

    def _render_game(self):
        cam_x = int(self.camera_x)
        
        # Draw background elements (parallax clouds)
        for i in range(5):
            cloud_x = 100 + i * 250 - (cam_x // 4) % (250 * 5)
            cloud_y = 50 + (i % 2) * 30
            pygame.gfxdraw.filled_ellipse(self.screen, cloud_x, cloud_y, 40, 15, (255, 255, 255, 100))
            pygame.gfxdraw.filled_ellipse(self.screen, cloud_x + 25, cloud_y + 10, 50, 20, (255, 255, 255, 100))

        # Draw Pits
        for pit in self.pits:
            flicker = self.np_random.integers(0, 5)
            r, g, b = self.COLOR_PIT
            color = (max(0, r-flicker*5), g, b)
            render_rect = pit.move(-cam_x, 0)
            pygame.draw.rect(self.screen, color, render_rect)

        # Draw Platforms
        for plat in self.platforms:
            render_rect = plat.move(-cam_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, render_rect)
            pygame.draw.line(self.screen, (101, 67, 33), render_rect.topleft, render_rect.topright, 3)

        # Draw Coins
        for coin in self.coins:
            scale = abs(math.sin(self.steps * 0.2 + coin.x))
            width = int(coin.width * scale)
            render_rect = pygame.Rect(coin.centerx - width // 2 - cam_x, coin.y, width, coin.height)
            pygame.draw.ellipse(self.screen, self.COLOR_COIN, render_rect)
            pygame.gfxdraw.aaellipse(self.screen, render_rect.centerx, render_rect.centery, render_rect.width//2, render_rect.height//2, self.COLOR_COIN)

        # Draw Flag
        if self.flag_rect:
            render_rect = self.flag_rect.move(-cam_x, 0)
            pygame.draw.line(self.screen, (192, 192, 192), render_rect.bottomleft, render_rect.topleft, 4)
            wave = math.sin(self.steps * 0.1) * 5
            flag_points = [(render_rect.left, render_rect.top), (render_rect.right, render_rect.top + 10 + wave), (render_rect.left, render_rect.top + 20)]
            pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_points)

        # Draw Particles
        for p in self.particles:
            p.draw(self.screen, cam_x)

        # Draw Player
        player_render_pos = (int(self.player_pos.x - cam_x), int(self.player_pos.y))
        h_squash = max(0.1, 1.0 - abs(self.player_vel.x) * 0.05)
        v_squash = max(0.1, 1.0 + self.player_vel.y * 0.02)
        if self.player_on_ground and abs(self.player_vel.x) > 0.1:
            v_squash = 0.8; h_squash = 1.2
        w = int(self.PLAYER_SIZE.x * h_squash)
        h = int(self.PLAYER_SIZE.y * v_squash)
        player_rect = pygame.Rect(player_render_pos[0] + (self.PLAYER_SIZE.x - w) / 2, player_render_pos[1] + (self.PLAYER_SIZE.y - h), w, h)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        pygame.draw.rect(self.screen, (255,255,255), player_rect, width=2, border_radius=4)

    def _render_ui(self):
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            surface = font.render(text, True, color)
            self.screen.blit(surface, pos)

        draw_text(f"Score: {self.score}", self.font_large, self.COLOR_TEXT, (10, 10))
        draw_text(f"Level: {self.level}", self.font_large, self.COLOR_TEXT, (self.WIDTH - 120, 10))
        draw_text(f"Steps: {self.steps}/{self.MAX_STEPS}", self.font_small, self.COLOR_TEXT, (10, 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "player_pos": (self.player_pos.x, self.player_pos.y),
        }

    def close(self):
        pygame.quit()

class Particle:
    def __init__(self, pos, np_random):
        self.pos = pygame.Vector2(pos)
        self.np_random = np_random
        angle = self.np_random.random() * 2 * math.pi
        speed = self.np_random.random() * 3 + 1
        self.vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.lifespan = self.np_random.integers(20, 40)
        self.color = GameEnv.COLOR_PARTICLE
        self.radius = self.np_random.integers(3, 6)

    def update(self):
        self.pos += self.vel
        self.vel.y += 0.1 # particle gravity
        self.lifespan -= 1
        return self.lifespan > 0

    def draw(self, surface, camera_x):
        alpha = max(0, min(255, int(255 * (self.lifespan / 30))))
        # Create a temporary surface for alpha blending
        temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        color_with_alpha = self.color + (alpha,)
        pygame.gfxdraw.filled_circle(temp_surf, self.radius, self.radius, self.radius, color_with_alpha)
        pygame.gfxdraw.aacircle(temp_surf, self.radius, self.radius, self.radius, color_with_alpha)
        pos = (int(self.pos.x - camera_x - self.radius), int(self.pos.y - self.radius))
        surface.blit(temp_surf, pos)

if __name__ == "__main__":
    # This block allows you to play the game directly
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Pixel Platformer")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        # --- Render the observation to the display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Control frame rate ---
        clock.tick(60)
        
    env.close()