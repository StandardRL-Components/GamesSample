
# Generated: 2025-08-27T22:33:38.858499
# Source Brief: brief_03162.md
# Brief Index: 3162

        
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
        "Controls: Use ←→ for air control. Press 'Up' for a small hop. "
        "Hold Space for a medium jump, or Shift for a high jump. Combine with ←→ to jump diagonally."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a hopping space explorer up a tower of procedurally generated platforms. "
        "Collect glowing fuel cells and execute risky long-jumps to maximize your score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Screen and clock
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.WIN_ALTITUDE = 4000
        self.MAX_STEPS = 3000

        # Colors
        self.COLOR_BG = (15, 10, 40)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GLOW = (100, 200, 255)
        self.COLOR_PLATFORM_OUTLINE = (255, 255, 255)
        self.COLOR_PLATFORM_FILL_1 = (60, 60, 100)
        self.COLOR_PLATFORM_FILL_2 = (100, 100, 160)
        self.COLOR_FUEL = (255, 220, 0)
        self.COLOR_FUEL_GLOW = (255, 220, 0)
        self.COLOR_TEXT = (240, 240, 240)

        # Physics and gameplay
        self.GRAVITY = 0.3
        self.JUMP_SMALL = -6.5
        self.JUMP_MEDIUM = -9.5
        self.JUMP_HIGH = -12.5
        self.AIR_CONTROL = 0.4
        self.MAX_FALL_SPEED = 10
        self.HORIZONTAL_SPEED = 4.5
        self.COYOTE_TIME_FRAMES = 5
        self.PLAYER_DIMS = pygame.Vector2(20, 30)

        # Fonts
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # State variables (initialized in reset)
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.coyote_timer = None
        self.jump_takeoff_y = None
        
        self.platforms = None
        self.fuel_cells = None
        self.stars = None
        
        self.camera_y = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.start_y = None
        self.max_altitude = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.start_y = self.SCREEN_HEIGHT - 50
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.start_y)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = True
        self.coyote_timer = 0
        self.jump_takeoff_y = self.player_pos.y
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.max_altitude = 0
        
        # World state
        self.camera_y = 0
        self._generate_initial_world()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        self._handle_input(action)
        
        prev_y = self.player_pos.y
        self._update_player_physics()
        
        # Reward for vertical movement
        height_gain = prev_y - self.player_pos.y  # Y is inverted (lower Y is higher)
        if not self.on_ground:
            reward += height_gain * 0.1
        
        # Penalty for horizontal movement
        reward -= abs(self.player_vel.x) * 0.005

        self.max_altitude = max(self.max_altitude, self.start_y - self.player_pos.y)

        landing_reward = self._handle_collisions()
        reward += landing_reward

        collect_reward = self._collect_fuel_cells()
        reward += collect_reward

        self._update_camera()
        self._cull_and_generate_objects()
        
        self.steps += 1
        terminated = self._check_termination()
        if terminated:
            if self.max_altitude >= self.WIN_ALTITUDE:
                win_bonus = 100
                reward += win_bonus
                self.score += win_bonus
            else: # Fell or timeout
                fall_penalty = -50
                reward += fall_penalty
                self.score += fall_penalty
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Horizontal air control
        if not self.on_ground:
            if movement == 3: # Left
                self.player_vel.x = max(self.player_vel.x - self.AIR_CONTROL, -self.HORIZONTAL_SPEED)
            elif movement == 4: # Right
                self.player_vel.x = min(self.player_vel.x + self.AIR_CONTROL, self.HORIZONTAL_SPEED)
        
        # Ground friction / Air drag
        if self.on_ground:
             self.player_vel.x *= 0.85
        else:
             self.player_vel.x *= 0.98

        # Jumping
        can_jump = self.on_ground or self.coyote_timer > 0
        if can_jump:
            jump_power = 0
            if shift_held: jump_power = self.JUMP_HIGH
            elif space_held: jump_power = self.JUMP_MEDIUM
            elif movement == 1: jump_power = self.JUMP_SMALL
            
            if jump_power != 0:
                self.player_vel.y = jump_power
                self.on_ground = False
                self.coyote_timer = 0
                self.jump_takeoff_y = self.player_pos.y
                # sfx: jump_sound()
                
                # Apply horizontal velocity on jump
                if movement == 3: self.player_vel.x = -self.HORIZONTAL_SPEED
                if movement == 4: self.player_vel.x = self.HORIZONTAL_SPEED

    def _update_player_physics(self):
        if self.on_ground:
            self.coyote_timer = self.COYOTE_TIME_FRAMES
        else:
            self.coyote_timer = max(0, self.coyote_timer - 1)
            self.player_vel.y += self.GRAVITY
            self.player_vel.y = min(self.player_vel.y, self.MAX_FALL_SPEED)
        
        self.player_pos += self.player_vel

        # Prevent going off horizontal sides
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_DIMS.x / 2, self.SCREEN_WIDTH - self.PLAYER_DIMS.x / 2)

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_DIMS.x / 2, self.player_pos.y - self.PLAYER_DIMS.y / 2, self.PLAYER_DIMS.x, self.PLAYER_DIMS.y)
        
        if self.player_vel.y > 0:
            for p in self.platforms:
                if player_rect.colliderect(p) and player_rect.bottom < p.centery:
                    self.on_ground = True
                    self.player_vel.y = 0
                    self.player_pos.y = p.top - self.PLAYER_DIMS.y / 2
                    # sfx: land_sound()
                    
                    # Calculate rewards on landing
                    reward = 0
                    # Penalty for landing near edge
                    edge_distance = min(abs(player_rect.centerx - p.left), abs(player_rect.centerx - p.right))
                    if edge_distance < 10:
                        reward -= 1
                    
                    # Reward for long jumps
                    jump_height = self.jump_takeoff_y - self.player_pos.y
                    if jump_height > 50: # Threshold for a long jump
                        reward += 2
                    
                    return reward
        return 0

    def _collect_fuel_cells(self):
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_DIMS.x / 2, self.player_pos.y - self.PLAYER_DIMS.y / 2, self.PLAYER_DIMS.x, self.PLAYER_DIMS.y)
        reward = 0
        
        for fuel in self.fuel_cells[:]:
            if player_rect.colliderect(fuel):
                self.fuel_cells.remove(fuel)
                reward += 5
                self.score += 5
                # sfx: collect_sound()
        return reward

    def _update_camera(self):
        target_cam_y = self.player_pos.y - self.SCREEN_HEIGHT * 0.4
        self.camera_y += (target_cam_y - self.camera_y) * 0.08

    def _cull_and_generate_objects(self):
        # Cull objects below the screen
        cull_line = self.camera_y + self.SCREEN_HEIGHT + 50
        self.platforms = [p for p in self.platforms if p.bottom > self.camera_y - 50]
        self.fuel_cells = [f for f in self.fuel_cells if f.bottom > self.camera_y - 50]
        
        # Generate new objects if needed
        if self.platforms:
            highest_platform_y = min(p.y for p in self.platforms)
            if highest_platform_y > self.camera_y - 100:
                self._generate_platforms(highest_platform_y)
                self._generate_fuel_cells()

    def _check_termination(self):
        # Fell off screen
        if self.player_pos.y > self.camera_y + self.SCREEN_HEIGHT + self.PLAYER_DIMS.y:
            self.game_over = True
            return True
        # Reached the top
        if self.max_altitude >= self.WIN_ALTITUDE:
            self.game_over = True
            return True
        # Timed out
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_platforms()
        self._render_fuel_cells()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "altitude": int(self.max_altitude),
        }
        
    def _generate_initial_world(self):
        # Platforms
        self.platforms = []
        start_platform = pygame.Rect(self.SCREEN_WIDTH/2 - 75, self.start_y + self.PLAYER_DIMS.y/2, 150, 15)
        self.platforms.append(start_platform)
        self._generate_platforms(start_platform.y)

        # Fuel Cells
        self.fuel_cells = []
        self._generate_fuel_cells()
        
        # Starfield
        self.stars = [
            (
                self.np_random.integers(0, self.SCREEN_WIDTH),
                self.np_random.integers(0, self.SCREEN_HEIGHT * 2),
                self.np_random.random() * 1.5 + 0.5, # radius
                self.np_random.random() * 0.4 + 0.1 # parallax speed
            ) for _ in range(150)
        ]

    def _generate_platforms(self, current_y):
        while len(self.platforms) < 25:
            last_platform = self.platforms[-1]
            
            difficulty = min(1.0, (self.start_y - last_platform.y) / self.WIN_ALTITUDE)
            max_gap_y = 80 + 70 * difficulty
            max_gap_x = int(self.SCREEN_WIDTH * 0.3 + self.SCREEN_WIDTH * 0.3 * difficulty)
            
            new_y = last_platform.y - self.np_random.integers(50, max_gap_y)
            
            offset_dir = 1 if last_platform.centerx < self.SCREEN_WIDTH / 2 else -1
            offset_dir = self.np_random.choice([-1, 1]) if self.np_random.random() < 0.3 else offset_dir
            new_x = last_platform.centerx + offset_dir * self.np_random.integers(int(max_gap_x*0.2), max_gap_x)
            
            width = max(60, self.np_random.integers(90, 160) * (1 - 0.6 * difficulty))
            new_x = np.clip(new_x, width/2, self.SCREEN_WIDTH - width/2)
            
            self.platforms.append(pygame.Rect(new_x - width/2, new_y, width, 15))

    def _generate_fuel_cells(self):
        for p in self.platforms[-15:]:
            if self.np_random.random() < 0.4:
                pos_x = p.centerx + self.np_random.integers(-int(p.width * 0.4), int(p.width * 0.4))
                pos_y = p.y - self.np_random.integers(30, 90)
                self.fuel_cells.append(pygame.Rect(pos_x - 5, pos_y - 5, 10, 10))

    def _render_background(self):
        for x, y, r, speed in self.stars:
            screen_x = int(x)
            screen_y = int((y - self.camera_y * speed) % (self.SCREEN_HEIGHT * 1.5))
            if 0 < screen_y < self.SCREEN_HEIGHT:
                color_val = int(100 + speed * 155)
                color = (color_val, color_val, color_val)
                pygame.gfxdraw.filled_circle(self.screen, screen_x, screen_y, int(r), color)
            
    def _render_player(self):
        screen_pos = self.player_pos - pygame.Vector2(0, self.camera_y)
        
        # Squash and stretch animation for game feel
        stretch = 1.0 - max(-1, min(1, self.player_vel.y / 15.0)) * 0.25
        squash = 1.0 + max(-1, min(1, self.player_vel.y / 15.0)) * 0.3
        if self.on_ground:
            squash = 1.25
            stretch = 0.75
        
        w, h = self.PLAYER_DIMS.x * squash, self.PLAYER_DIMS.y * stretch
        player_rect = pygame.Rect(screen_pos.x - w / 2, screen_pos.y - h / 2, w, h)
        
        # Glow effect using a surface with alpha blending
        glow_radius = int(w * 1.2)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_PLAYER_GLOW, 40), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (player_rect.centerx - glow_radius, player_rect.centery - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Player body (ellipse for a softer look)
        pygame.gfxdraw.filled_ellipse(self.screen, int(player_rect.centerx), int(player_rect.centery), int(w/2), int(h/2), self.COLOR_PLAYER)
        pygame.gfxdraw.aaellipse(self.screen, int(player_rect.centerx), int(player_rect.centery), int(w/2), int(h/2), self.COLOR_PLAYER)

    def _render_platforms(self):
        for p in self.platforms:
            screen_rect = p.copy()
            screen_rect.y -= self.camera_y
            if screen_rect.bottom < 0 or screen_rect.top > self.SCREEN_HEIGHT:
                continue

            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_FILL_1, screen_rect, border_radius=3)
            top_rect = pygame.Rect(screen_rect.x, screen_rect.y, screen_rect.width, 5)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_FILL_2, top_rect, border_top_left_radius=3, border_top_right_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, screen_rect, 2, border_radius=3)
            
    def _render_fuel_cells(self):
        for f in self.fuel_cells:
            screen_rect = f.copy()
            screen_rect.y -= self.camera_y
            if screen_rect.bottom < 0 or screen_rect.top > self.SCREEN_HEIGHT:
                continue
            
            # Pulsing glow effect
            pulse = math.sin(self.steps * 0.1) * 3 + 12
            s = pygame.Surface((pulse * 2, pulse * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.COLOR_FUEL_GLOW, 80), (pulse, pulse), int(pulse))
            self.screen.blit(s, (screen_rect.centerx - pulse, screen_rect.centery - pulse), special_flags=pygame.BLEND_RGBA_ADD)
            
            # Core element
            pygame.draw.rect(self.screen, self.COLOR_FUEL, screen_rect, border_radius=2)
            
    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        altitude_text = self.font_large.render(f"ALT: {int(self.max_altitude)}m", True, self.COLOR_TEXT)
        self.screen.blit(altitude_text, (self.SCREEN_WIDTH - altitude_text.get_width() - 10, 10))

        # Progress bar for altitude goal
        progress = min(1.0, self.max_altitude / self.WIN_ALTITUDE)
        bar_width = self.SCREEN_WIDTH - 20
        bar_height = 5
        bar_y = 40
        pygame.draw.rect(self.screen, (255, 255, 255, 50), (10, bar_y, bar_width, bar_height), 1)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (10, bar_y, bar_width * progress, bar_height))

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    
    # Use a separate display for human play
    human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Astro Hopper")
    
    obs, info = env.reset()
    done = False
    
    print(env.game_description)
    print(env.user_guide)
    
    while not done:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        # K_DOWN is not used for jumping in this scheme
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- Rendering ---
        # The env._get_observation() already renders to its internal surface.
        # We just need to blit it to the display.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESET ---")
                obs, info = env.reset()
                
        env.clock.tick(30) # Match the intended FPS

    print(f"Game Over! Final Score: {info['score']}, Max Altitude: {info['altitude']}m")
    pygame.quit()