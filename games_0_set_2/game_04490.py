
# Generated: 2025-08-28T02:33:50.362999
# Source Brief: brief_04490.md
# Brief Index: 4490

        
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
        "Controls: ←→ to run, ↑ to jump. Reach the yellow flag at the end!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, side-scrolling platformer. Guide your robot through a "
        "procedurally generated neon world, collect speed boosts, and avoid "
        "deadly pits to reach the goal as quickly as possible."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Game parameters
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    TIME_PER_LIFE_SECONDS = 30
    MAX_LIVES = 3
    MAX_STEPS = 1500 # Roughly 50 seconds
    LEVEL_WIDTH_PIXELS = 6000

    # Colors (Neon on Dark)
    COLOR_BG = (10, 10, 30)
    COLOR_GRID = (20, 20, 60)
    COLOR_PLATFORM = (128, 128, 140)
    COLOR_PIT_HAZARD = (255, 0, 77)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_BOOST = (255, 255, 0)
    COLOR_SPEED_BOOST = (0, 255, 0)
    COLOR_GOAL = (255, 255, 0)
    COLOR_TEXT = (255, 255, 255)
    
    # Physics
    GRAVITY = 0.6
    PLAYER_JUMP_STRENGTH = -12
    PLAYER_MOVE_SPEED = 5
    PLAYER_BOOST_MULTIPLIER = 1.8
    PLAYER_BOOST_DURATION = 5 * FPS # 5 seconds

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
        self.font_icon = pygame.font.SysFont("monospace", 24, bold=True)

        self.game_objects = []
        self.platforms = []
        self.pits = []
        self.speed_boosts = []
        self.particles = []
        self.goal = None
        self.player = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.MAX_LIVES
        self.time_per_life = self.TIME_PER_LIFE_SECONDS * self.FPS
        self.timer = self.time_per_life
        
        self.camera_offset = pygame.Vector2(0, 0)
        
        self.particles.clear()
        
        self._generate_level()
        
        start_pos = (100, self.SCREEN_HEIGHT - 150)
        self.player = Player(start_pos, self)
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms.clear()
        self.pits.clear()
        self.speed_boosts.clear()
        self.game_objects.clear()

        # Start platform
        self.platforms.append(pygame.Rect(-200, self.SCREEN_HEIGHT - 100, 500, 100))

        x = 300
        current_y = self.SCREEN_HEIGHT - 100
        
        while x < self.LEVEL_WIDTH_PIXELS:
            # Difficulty scaling based on progress through the level
            progress_ratio = x / self.LEVEL_WIDTH_PIXELS
            
            # Platform gaps increase over distance
            min_gap = 50 + 50 * progress_ratio
            max_gap = 120 + 80 * progress_ratio
            gap = self.np_random.integers(min_gap, max_gap)
            
            # Pit frequency increases over distance
            pit_chance = 0.1 + 0.3 * progress_ratio
            if self.np_random.random() < pit_chance:
                pit_width = self.np_random.integers(80, 150)
                self.pits.append(pygame.Rect(x, 0, pit_width, self.SCREEN_HEIGHT * 2))
                x += pit_width
            
            x += gap
            
            plat_width = self.np_random.integers(150, 400)
            y_change = self.np_random.integers(-80, 80)
            current_y = np.clip(current_y + y_change, 200, self.SCREEN_HEIGHT - 50)
            
            new_platform = pygame.Rect(x, current_y, plat_width, self.SCREEN_HEIGHT - current_y)
            self.platforms.append(new_platform)
            
            # Add speed boosts occasionally
            if self.np_random.random() < 0.2:
                boost_pos = (x + plat_width / 2, current_y - 30)
                self.speed_boosts.append(SpeedBoost(boost_pos))

            x += plat_width
        
        # Add goal at the end
        self.goal = pygame.Rect(x + 100, current_y - 100, 20, 100)
        self.game_objects = self.platforms + self.pits + self.speed_boosts + [self.goal]

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.0
        terminated = self.game_over

        if not terminated:
            # Handle player input
            if movement == 1: self.player.jump() # Up
            if movement == 3: self.player.move(-1) # Left
            if movement == 4: self.player.move(1) # Right
            
            # Update game state
            self.player.update(self.platforms)
            
            for p in self.particles[:]:
                p.update()
                if p.lifetime <= 0:
                    self.particles.remove(p)

            for boost in self.speed_boosts[:]:
                if self.player.rect.colliderect(boost.rect):
                    # SFX: Powerup collect
                    self.player.activate_boost()
                    self.speed_boosts.remove(boost)
                    reward += 1.0
                    for _ in range(30):
                        self.particles.append(Particle(boost.pos, self.np_random, self.COLOR_SPEED_BOOST, 20, 2.5))

            # Update timer and reward for survival
            self.timer -= 1
            reward += 0.01

            # Check for termination conditions
            fell_in_pit = any(self.player.rect.colliderect(pit) for pit in self.pits)
            fell_off_world = self.player.pos.y > self.SCREEN_HEIGHT + 100
            
            if self.player.rect.colliderect(self.goal):
                # SFX: Victory fanfare
                reward += 50.0
                reward += 100.0 * (self.timer / self.time_per_life)
                self.game_over = True
            elif fell_in_pit or fell_off_world or self.timer <= 0:
                # SFX: Explosion / Life Lost
                self.lives -= 1
                reward -= 10.0
                if self.lives > 0:
                    self.player.reset()
                    self.timer = self.time_per_life
                else:
                    self.game_over = True
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            
        terminated = self.game_over
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_camera()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _update_camera(self):
        target_x = self.player.pos.x - self.SCREEN_WIDTH / 3
        target_y = self.player.pos.y - self.SCREEN_HEIGHT / 2
        
        # Smooth camera follow
        self.camera_offset.x += (target_x - self.camera_offset.x) * 0.1
        self.camera_offset.y += (target_y - self.camera_offset.y) * 0.1

    def _render_game(self):
        # Draw background grid
        cam_x_int, cam_y_int = int(self.camera_offset.x), int(self.camera_offset.y)
        grid_size = 50
        for x in range(-cam_x_int % grid_size, self.SCREEN_WIDTH, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(-cam_y_int % grid_size, self.SCREEN_HEIGHT, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw game objects relative to camera
        for platform in self.platforms:
            self._draw_rect_camera(platform, self.COLOR_PLATFORM)
        
        for pit in self.pits:
            # Animated flicker for pits
            if self.steps % 4 < 2:
                self._draw_rect_camera(pit, self.COLOR_PIT_HAZARD)

        for boost in self.speed_boosts:
            boost.draw(self.screen, self.camera_offset, self.steps)

        # Draw goal
        goal_screen_rect = self.goal.move(-self.camera_offset.x, -self.camera_offset.y)
        pygame.draw.rect(self.screen, self.COLOR_GOAL, goal_screen_rect)
        pygame.draw.polygon(self.screen, self.COLOR_GOAL, [
            (goal_screen_rect.right, goal_screen_rect.top),
            (goal_screen_rect.right + 20, goal_screen_rect.top + 10),
            (goal_screen_rect.right, goal_screen_rect.top + 20)
        ])

        # Draw particles
        for p in self.particles:
            p.draw(self.screen, self.camera_offset)

        # Draw player
        self.player.draw(self.screen, self.camera_offset)
    
    def _draw_rect_camera(self, rect, color):
        screen_rect = rect.move(-self.camera_offset.x, -self.camera_offset.y)
        pygame.draw.rect(self.screen, color, screen_rect)

    def _render_ui(self):
        # Render lives
        for i in range(self.lives):
            life_text = self.font_icon.render("R", True, self.COLOR_PLAYER)
            self.screen.blit(life_text, (20 + i * 25, 15))

        # Render timer
        time_left = max(0, self.timer / self.FPS)
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(timer_text, timer_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "time_left": max(0, self.timer / self.FPS),
        }

    def close(self):
        pygame.quit()

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

class Player:
    def __init__(self, pos, env):
        self.env = env
        self.start_pos = pygame.Vector2(pos)
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)
        self.size = pygame.Vector2(20, 30)
        self.rect = pygame.Rect(self.pos.x, self.pos.y, self.size.x, self.size.y)
        self.on_ground = False
        self.boost_timer = 0
        self.squash = 1.0

    def reset(self):
        self.pos = pygame.Vector2(self.start_pos)
        self.vel = pygame.Vector2(0, 0)
        self.boost_timer = 0
        
    def move(self, direction):
        speed = self.env.PLAYER_MOVE_SPEED
        if self.boost_timer > 0:
            speed *= self.env.PLAYER_BOOST_MULTIPLIER
        self.vel.x = direction * speed

    def jump(self):
        if self.on_ground:
            # SFX: Jump
            self.vel.y = self.env.PLAYER_JUMP_STRENGTH
            self.on_ground = False
            self.squash = 1.5 # Stretch on jump
            for _ in range(10):
                self.env.particles.append(Particle(self.rect.midbottom, self.env.np_random, (200,200,200), 15, 1.5))

    def activate_boost(self):
        self.boost_timer = self.env.PLAYER_BOOST_DURATION

    def update(self, platforms):
        # Horizontal movement and friction
        if not self.vel.x:
            self.vel.x *= 0.9
        if abs(self.vel.x) < 0.1:
            self.vel.x = 0
            
        self.pos.x += self.vel.x
        self.rect.x = int(self.pos.x)
        self.handle_collisions('horizontal', platforms)

        # Vertical movement (gravity)
        self.vel.y += self.env.GRAVITY
        if self.vel.y > 15: self.vel.y = 15 # Terminal velocity
        self.pos.y += self.vel.y
        self.rect.y = int(self.pos.y)
        self.on_ground = False
        self.handle_collisions('vertical', platforms)
        
        # Boost timer
        if self.boost_timer > 0:
            self.boost_timer -= 1
            # Add trail particles
            if self.env.steps % 2 == 0:
                self.env.particles.append(Particle(self.rect.center, self.env.np_random, self.env.COLOR_PLAYER_BOOST, 10, 1, self.vel * -0.1))

        # Squash and stretch effect
        self.squash += (1.0 - self.squash) * 0.2

    def handle_collisions(self, direction, platforms):
        colliders = [p for p in platforms if self.rect.colliderect(p)]
        for platform in colliders:
            if direction == 'horizontal':
                if self.vel.x > 0: self.rect.right = platform.left
                if self.vel.x < 0: self.rect.left = platform.right
                self.pos.x = self.rect.x
            elif direction == 'vertical':
                if self.vel.y > 0:
                    if not self.on_ground:
                        # SFX: Land
                        self.squash = 0.6 # Squash on land
                    self.rect.bottom = platform.top
                    self.on_ground = True
                    self.vel.y = 0
                if self.vel.y < 0:
                    self.rect.top = platform.bottom
                    self.vel.y = 0
                self.pos.y = self.rect.y

    def draw(self, surface, camera_offset):
        # Visuals: Squash and stretch
        h = self.size.y * self.squash
        w = self.size.x / self.squash
        
        draw_rect = pygame.Rect(
            self.rect.centerx - w/2 - camera_offset.x,
            self.rect.centery - h/2 - camera_offset.y,
            w, h
        )
        
        color = self.env.COLOR_PLAYER_BOOST if self.boost_timer > 0 else self.env.COLOR_PLAYER
        
        # Glow effect
        glow_radius = int(w * 0.8)
        for i in range(glow_radius, 0, -2):
            alpha = 60 * (1 - i / glow_radius)
            pygame.gfxdraw.filled_circle(
                surface, int(draw_rect.centerx), int(draw_rect.centery), i, (*color, alpha)
            )

        # Body
        pygame.draw.rect(surface, color, draw_rect, border_radius=3)
        
        # Eye
        eye_x = draw_rect.centerx + (3 if self.vel.x >= 0 else -3)
        pygame.draw.circle(surface, (0,0,0), (eye_x, draw_rect.centery - 3), 3)

class SpeedBoost:
    def __init__(self, pos):
        self.pos = pygame.Vector2(pos)
        self.size = 16
        self.rect = pygame.Rect(pos[0] - self.size/2, pos[1] - self.size/2, self.size, self.size)

    def draw(self, surface, camera_offset, step):
        screen_pos = self.pos - camera_offset
        
        # Pulsing effect
        pulse = (math.sin(step * 0.2) + 1) / 2
        radius = int(self.size / 2 * (1 + pulse * 0.3))
        
        # Glow
        for i in range(radius, 0, -1):
            alpha = 80 * (1 - i / radius)
            pygame.gfxdraw.filled_circle(surface, int(screen_pos.x), int(screen_pos.y), i, (*GameEnv.COLOR_SPEED_BOOST, alpha))
        
        # Core
        pygame.gfxdraw.filled_circle(surface, int(screen_pos.x), int(screen_pos.y), int(self.size/2), GameEnv.COLOR_SPEED_BOOST)
        pygame.gfxdraw.aacircle(surface, int(screen_pos.x), int(screen_pos.y), int(self.size/2), GameEnv.COLOR_SPEED_BOOST)


class Particle:
    def __init__(self, pos, rng, color, lifetime, speed_mult=1.0, initial_vel=None):
        self.pos = pygame.Vector2(pos)
        self.rng = rng
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        
        if initial_vel is None:
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * speed_mult
            self.vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
        else:
            self.vel = pygame.Vector2(initial_vel)

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1
        self.vel *= 0.98 # Damping

    def draw(self, surface, camera_offset):
        life_ratio = self.lifetime / self.max_lifetime
        radius = int(life_ratio * 4)
        if radius > 0:
            screen_pos = self.pos - camera_offset
            pygame.gfxdraw.filled_circle(
                surface, int(screen_pos.x), int(screen_pos.y), radius, (*self.color, int(255 * life_ratio))
            )