import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: Hold Space to charge a jump. Use ←→ to aim. Release Space to jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist arcade platformer. Jump between procedurally generated platforms to ascend as high as possible."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 5000
    WIN_HEIGHT = 5000

    # Colors
    COLOR_BG_TOP = (100, 149, 237)  # Cornflower Blue
    COLOR_BG_BOTTOM = (15, 23, 42)    # Dark Slate Blue
    COLOR_PLAYER = (255, 255, 0)      # Bright Yellow
    COLOR_PLAYER_GLOW = (255, 255, 150, 100)
    COLOR_PLATFORM = (255, 255, 255)  # White
    COLOR_PARTICLE = (220, 220, 220)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_CHARGE_BAR_BG = (50, 50, 50)
    COLOR_CHARGE_BAR_FG = (100, 255, 100)

    # Physics
    GRAVITY = 0.5
    PLAYER_SIZE = 20
    JUMP_VELOCITY = -12
    MAX_HORIZONTAL_SPEED = 10
    CHARGE_RATE = 0.05
    
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
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Create a static background surface for efficiency
        self.background_surf = self._create_gradient_background()

        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.platforms = None
        self.particles = None
        self.camera_y = None
        self.steps = None
        self.score = None
        self.max_height_reached = None
        self.jump_charge = None
        self.jump_direction = None
        self.was_space_held = None
        self.rng = None
        
        # Initialize rng here to be available for reset
        self.rng = np.random.default_rng()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.player_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.on_ground = True
        self.was_space_held = False
        self.jump_charge = 0.0
        self.jump_direction = 0 # -1 for left, 1 for right, 0 for up

        self.platforms = []
        self.particles = []
        
        self.steps = 0
        self.score = 0.0
        self.max_height_reached = 0
        self.camera_y = 0

        self._generate_initial_platforms()
        self._update_camera()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.0
        terminated = False
        truncated = False
        
        # --- Handle Input ---
        self._handle_input(movement, space_held)

        # --- Update Physics ---
        self._update_player_physics()

        # --- Check Collisions ---
        landed_on_platform = self._check_collisions()
        if landed_on_platform:
            reward += 1.0
            # sfx: landing sound

        # --- Update World State ---
        self._update_camera()
        self._cull_and_generate_platforms()
        self._update_particles()
        
        # --- Calculate Reward ---
        current_height = -self.player_pos.y
        if current_height > self.max_height_reached:
            reward += (current_height - self.max_height_reached) * 0.1
            self.max_height_reached = current_height
        
        self.score += reward

        # --- Check Termination Conditions ---
        if self.player_pos.y - self.camera_y > self.SCREEN_HEIGHT + self.PLAYER_SIZE:
            terminated = True
            reward -= 100.0
            self.score -= 100.0 # Apply terminal penalty to score as well
            # sfx: fall sound
        elif self.max_height_reached >= self.WIN_HEIGHT:
            terminated = True
            reward += 100.0
            self.score += 100.0
            # sfx: win sound
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            truncated = True

        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _handle_input(self, movement, space_held):
        # Handle jump charging and aiming
        if self.on_ground:
            if space_held:
                self.jump_charge = min(1.0, self.jump_charge + self.CHARGE_RATE)
                # Aim while charging
                if movement == 3: # Left
                    self.jump_direction = -1
                elif movement == 4: # Right
                    self.jump_direction = 1
                else: # Up, Down, None
                    self.jump_direction = 0
            # On release
            elif self.was_space_held and not space_held:
                self._jump()
        
        self.was_space_held = space_held

    def _jump(self):
        if self.jump_charge > 0.1: # Minimum charge to jump
            self.on_ground = False
            self.player_vel.y = self.JUMP_VELOCITY
            self.player_vel.x = self.jump_direction * self.jump_charge * self.MAX_HORIZONTAL_SPEED
            self.jump_charge = 0
            # sfx: jump sound
            # Spawn jump particles
            for _ in range(10):
                angle = self.rng.uniform(math.pi, 2 * math.pi)
                speed = self.rng.uniform(1, 3)
                vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
                self.particles.append({
                    "pos": self.player_pos.copy() + pygame.math.Vector2(0, self.PLAYER_SIZE / 2),
                    "vel": vel,
                    "radius": self.rng.uniform(2, 4),
                    "life": 20
                })

    def _update_player_physics(self):
        # Apply gravity
        if not self.on_ground:
            self.player_vel.y += self.GRAVITY

        # Update position
        self.player_pos += self.player_vel

        # Horizontal friction/drag
        if self.on_ground:
            self.player_vel.x *= 0.85
        else:
            self.player_vel.x *= 0.98

        # Screen bounds (horizontal)
        if self.player_pos.x < 0:
            self.player_pos.x = 0
            self.player_vel.x *= -0.5
        elif self.player_pos.x > self.SCREEN_WIDTH - self.PLAYER_SIZE:
            self.player_pos.x = self.SCREEN_WIDTH - self.PLAYER_SIZE
            self.player_vel.x *= -0.5
            
    def _check_collisions(self):
        landed = False
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        # Only check for landing if moving downwards
        if self.player_vel.y > 0:
            for plat in self.platforms:
                # Check if player's bottom is intersecting the top surface of the platform
                if player_rect.colliderect(plat) and player_rect.bottom <= plat.top + self.player_vel.y:
                    self.player_pos.y = plat.top - self.PLAYER_SIZE
                    self.player_vel.y = 0
                    self.on_ground = True
                    landed = True
                    self._spawn_landing_particles(pygame.math.Vector2(player_rect.midbottom))
                    break
        return landed

    def _update_camera(self):
        # Camera follows player vertically, keeping them in the middle third of the screen
        target_camera_y = self.player_pos.y - self.SCREEN_HEIGHT * 0.66
        # Smooth camera movement
        self.camera_y += (target_camera_y - self.camera_y) * 0.1

    def _cull_and_generate_platforms(self):
        # Remove platforms that are off-screen below
        self.platforms = [p for p in self.platforms if p.top < self.camera_y + self.SCREEN_HEIGHT]

        if not self.platforms:
            self._generate_initial_platforms()
            return

        last_platform = max(self.platforms, key=lambda p: -p.y)
        
        while last_platform.y > self.camera_y - self.PLAYER_SIZE:
            height_progress = self.max_height_reached / self.WIN_HEIGHT
            width = max(40, 150 - 120 * height_progress)
            
            max_jump_height = abs(self.JUMP_VELOCITY**2 / (2 * self.GRAVITY))
            
            dy = self.rng.uniform(max_jump_height * 0.3, max_jump_height * 0.85)
            new_y = last_platform.y - dy

            max_horizontal_reach = self.MAX_HORIZONTAL_SPEED * (abs(self.JUMP_VELOCITY) / self.GRAVITY) * 2
            dx_range = max_horizontal_reach * 0.7
            
            min_x = max(0, last_platform.centerx - dx_range)
            max_x = min(self.SCREEN_WIDTH - width, last_platform.centerx + dx_range)
            
            new_x = self.rng.uniform(min_x, max_x) if min_x < max_x else (self.SCREEN_WIDTH - width) / 2

            new_platform = pygame.Rect(new_x, new_y, width, 15)
            self.platforms.append(new_platform)
            last_platform = new_platform

    def _generate_initial_platforms(self):
        self.platforms.append(pygame.Rect(0, self.SCREEN_HEIGHT - 20, self.SCREEN_WIDTH, 20))
        for i in range(1, 15):
            last_platform = self.platforms[-1]
            width = 150
            new_y = last_platform.y - self.rng.uniform(50, 120)
            new_x = self.rng.uniform(0, self.SCREEN_WIDTH - width)
            self.platforms.append(pygame.Rect(new_x, new_y, width, 15))

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["radius"] -= 0.1
            p["life"] -= 1
            if p["life"] <= 0 or p["radius"] <= 0:
                self.particles.remove(p)

    def _spawn_landing_particles(self, pos):
        for _ in range(15):
            angle = self.rng.uniform(-math.pi * 0.8, -math.pi * 0.2)
            speed = self.rng.uniform(0.5, 2.5)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "radius": self.rng.uniform(1, 4),
                "life": 15
            })

    def _get_observation(self):
        self.screen.blit(self.background_surf, (0, 0))
        self._render_platforms()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_platforms(self):
        for plat in self.platforms:
            screen_y = plat.y - self.camera_y
            if -plat.height < screen_y < self.SCREEN_HEIGHT:
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, (plat.x, screen_y, plat.width, plat.height), border_radius=3)

    def _render_player(self):
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y - self.camera_y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        glow_radius = int(self.PLAYER_SIZE * 1.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (player_rect.centerx - glow_radius, player_rect.centery - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p["pos"].x), int(p["pos"].y - self.camera_y))
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE, pos, max(0, int(p["radius"])))

    def _render_ui(self):
        height_text = self.font.render(f"Height: {int(self.max_height_reached)} m", True, self.COLOR_UI_TEXT)
        self.screen.blit(height_text, (10, 10))
        
        score_text = self.small_font.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 45))

        if self.jump_charge > 0:
            bar_width = 50
            bar_height = 8
            bar_x = self.player_pos.x + (self.PLAYER_SIZE / 2) - (bar_width / 2)
            bar_y = self.player_pos.y - self.camera_y - bar_height - 5
            
            fill_width = bar_width * self.jump_charge
            
            pygame.draw.rect(self.screen, self.COLOR_CHARGE_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=2)
            pygame.draw.rect(self.screen, self.COLOR_CHARGE_BAR_FG, (bar_x, bar_y, fill_width, bar_height), border_radius=2)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": self.max_height_reached,
        }

    def _create_gradient_background(self):
        surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = [
                int(self.COLOR_BG_BOTTOM[i] * (1 - ratio) + self.COLOR_BG_TOP[i] * ratio)
                for i in range(3)
            ]
            pygame.draw.line(surf, color, (0, y), (self.SCREEN_WIDTH, y))
        return surf
        
    def close(self):
        if self.screen is not None:
            pygame.font.quit()
            pygame.quit()
            self.screen = None

if __name__ == "__main__":
    # The main loop is for human play and is not part of the Gymnasium interface
    # It will not be run by the evaluation server.
    # Un-set the headless environment variable to allow rendering
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    import sys

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("-" * 30)
    print(env.game_description)
    print(env.user_guide)
    print("-" * 30)
    
    while running:
        keys = pygame.key.get_pressed()
        
        movement = 0 
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    print("Resetting environment.")
                    total_reward = 0
                    obs, info = env.reset()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            print("Press 'R' to restart or 'Q' to quit.")
            
            wait_for_input = True
            while wait_for_input:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_input = False
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            wait_for_input = False
                            running = False
                        if event.key == pygame.K_r:
                            print("Resetting environment.")
                            total_reward = 0
                            obs, info = env.reset()
                            wait_for_input = False

        clock.tick(env.FPS)

    env.close()
    sys.exit()