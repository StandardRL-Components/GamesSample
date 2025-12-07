import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:40:36.439635
# Source Brief: brief_01135.md
# Brief Index: 1135
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a momentum-based platformer.
    The agent controls a character that must ascend a procedurally generated tower.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Ascend a procedurally generated tower in this momentum-based platformer. "
        "Aim your jumps carefully to reach the top."
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to aim your jump angle. Press space to jump from a platform."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 50)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_PLATFORM = (100, 100, 110)
    COLOR_PLATFORM_TOP = (255, 100, 100)
    COLOR_TEXT = (220, 220, 240)
    COLOR_INFO_TEXT = (150, 150, 170)
    
    # Screen and World
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Player Physics
    PLAYER_SIZE = 16
    GRAVITY = 0.4
    JUMP_POWER = 9.0
    MAX_FALL_SPEED = 10.0
    
    # Game Rules
    NUM_PLATFORMS = 20
    MAX_EPISODE_STEPS = 5000
    BASE_PLATFORM_V_SPACING = 90
    BASE_PLATFORM_H_SPACING = 120
    
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
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_title = pygame.font.SysFont("monospace", 24, bold=True)

        # Persistent state (survives resets)
        self.difficulty_multiplier = 1.0
        self.highest_altitude_ever = 0

        # Initialize episode-specific state variables
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.platforms = []
        self.is_grounded = False
        self.last_landed_platform_idx = -1
        self.jump_direction_idx = 0
        self.last_space_held = False
        self.camera_y = 0.0
        self.steps = 0
        self.score = 0
        self.combo = 0
        self.particles = []
        self.jump_trail = deque(maxlen=10)

        # This will be properly initialized in reset()
        # self.reset() is called by the environment wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset episode state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.combo = 0
        self.last_landed_platform_idx = 0
        self.jump_direction_idx = 2 # Default to up-right
        self.last_space_held = False
        self.particles.clear()
        self.jump_trail.clear()

        # Generate tower
        self._generate_tower()

        # Place player on the starting platform
        start_platform = self.platforms[0]
        self.player_pos = pygame.Vector2(start_platform.centerx, start_platform.top - self.PLAYER_SIZE)
        self.player_vel = pygame.Vector2(0, 0)
        self.is_grounded = True
        
        # Reset camera
        self.camera_y = self.player_pos.y - self.SCREEN_HEIGHT * 0.75

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action
        space_press = space_held and not self.last_space_held
        self.last_space_held = bool(space_held)

        self._handle_input(movement, space_press)
        reward = self._update_physics_and_collisions()
        self._update_camera()
        self._update_particles()
        
        self.steps += 1
        self.score += reward
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_EPISODE_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.player_pos.y > self.camera_y + self.SCREEN_HEIGHT + 50:
                reward += -100.0 # Fell off
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement_action, space_press):
        # Action 0 is no-op for direction
        if movement_action > 0:
            self.jump_direction_idx = movement_action

        if space_press and self.is_grounded:
            # SFX: Jump sound
            self.is_grounded = False
            
            # Map action index to a direction vector
            if self.jump_direction_idx == 1: # Up-Left
                direction = pygame.Vector2(-1, -1.5).normalize()
            elif self.jump_direction_idx == 2: # Up-Right
                direction = pygame.Vector2(1, -1.5).normalize()
            elif self.jump_direction_idx == 3: # Down-Left
                direction = pygame.Vector2(-1, 0.5).normalize()
            elif self.jump_direction_idx == 4: # Down-Right
                direction = pygame.Vector2(1, 0.5).normalize()
            else: # Should not happen, but default to up
                direction = pygame.Vector2(0, -1)

            self.player_vel = direction * self.JUMP_POWER
            self.jump_trail.clear()
            self._create_particles(self.player_pos + pygame.Vector2(self.PLAYER_SIZE/2, self.PLAYER_SIZE), 20, self.COLOR_PLAYER)

    def _update_physics_and_collisions(self):
        reward = 0.0
        
        if not self.is_grounded:
            # Apply gravity
            self.player_vel.y += self.GRAVITY
            self.player_vel.y = min(self.player_vel.y, self.MAX_FALL_SPEED)
            
            # Update position
            self.player_pos += self.player_vel
            self.jump_trail.append(self.player_pos.copy())

            # Wall bouncing
            if self.player_pos.x < 0:
                self.player_pos.x = 0
                self.player_vel.x *= -0.5
            if self.player_pos.x > self.SCREEN_WIDTH - self.PLAYER_SIZE:
                self.player_pos.x = self.SCREEN_WIDTH - self.PLAYER_SIZE
                self.player_vel.x *= -0.5

            # Check for landing
            player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)
            for i, plat in enumerate(self.platforms):
                if player_rect.colliderect(plat) and self.player_vel.y > 0:
                    # Check if player was above the platform in the previous frame
                    if player_rect.bottom - self.player_vel.y <= plat.top:
                        self.player_pos.y = plat.top - self.PLAYER_SIZE
                        self.player_vel.y = 0
                        self.player_vel.x *= 0.5 # Friction
                        self.is_grounded = True
                        # SFX: Landing puff
                        self._create_particles(player_rect.midbottom, 15, self.COLOR_PLATFORM)

                        if i > self.last_landed_platform_idx:
                            reward += 0.1  # Reward for vertical progress
                            self.combo += 1
                            if self.combo > 1:
                                reward += 1.0 # Combo reward
                        else:
                            self.combo = 0

                        self.last_landed_platform_idx = i

                        # Check for win condition
                        if i == len(self.platforms) - 1:
                            reward += 100.0 # Reached the top!
                            self.game_over = True
                            self.difficulty_multiplier *= 1.05
                            # SFX: Victory fanfare
                        
                        break
        return reward

    def _check_termination(self):
        # Game over flag is set on win or can be set here on loss/timeout
        if self.game_over:
            return True
        # Fall off screen
        if self.player_pos.y > self.camera_y + self.SCREEN_HEIGHT + 50:
            return True
        # Max steps is handled as truncation
        return False
        
    def _generate_tower(self):
        self.platforms.clear()
        
        # Start platform
        start_plat = pygame.Rect(self.SCREEN_WIDTH // 2 - 50, self.SCREEN_HEIGHT - 40, 100, 20)
        self.platforms.append(start_plat)

        last_plat_center_x = start_plat.centerx
        current_y = start_plat.top

        for _ in range(self.NUM_PLATFORMS):
            v_spacing = self.BASE_PLATFORM_V_SPACING * self.difficulty_multiplier
            h_spacing = self.BASE_PLATFORM_H_SPACING * self.difficulty_multiplier

            width = self.np_random.integers(60, 101)
            height = 20
            
            offset_x = self.np_random.uniform(-h_spacing, h_spacing)
            x = last_plat_center_x + offset_x
            
            # Clamp to screen bounds
            x = max(width // 2, min(x, self.SCREEN_WIDTH - width // 2))
            
            y = current_y - v_spacing - self.np_random.uniform(-10, 10)

            self.platforms.append(pygame.Rect(x - width // 2, y, width, height))
            current_y = y
            last_plat_center_x = x

    def _update_camera(self):
        target_cam_y = self.player_pos.y - self.SCREEN_HEIGHT * 0.6
        # Smooth camera movement
        self.camera_y += (target_cam_y - self.camera_y) * 0.08
    
    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifetime = self.np_random.integers(20, 41)
            self.particles.append([pygame.Vector2(pos), vel, lifetime, color])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[1] # pos += vel
            p[1].y += 0.1 # particle gravity
            p[2] -= 1 # lifetime
        self.particles = [p for p in self.particles if p[2] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Calculate screen offset from camera
        offset = pygame.Vector2(0, -self.camera_y)

        # Draw background grid
        grid_size = 50
        start_x = -int(offset.x) % grid_size
        start_y = -int(offset.y) % grid_size
        for x in range(start_x, self.SCREEN_WIDTH, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(start_y, self.SCREEN_HEIGHT, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw particles
        for pos, _, lifetime, color in self.particles:
            screen_pos = pos + offset
            radius = int(max(0, lifetime / 8))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), radius, color)

        # Draw jump trail
        if len(self.jump_trail) > 1:
            for i in range(len(self.jump_trail) - 1):
                p1 = self.jump_trail[i] + offset
                p2 = self.jump_trail[i+1] + offset
                alpha = int(200 * (i / len(self.jump_trail)))
                pygame.draw.line(self.screen, self.COLOR_PLAYER + (alpha,), p1, p2, 2)

        # Draw platforms
        for i, plat in enumerate(self.platforms):
            screen_rect = plat.move(offset)
            color = self.COLOR_PLATFORM_TOP if i == len(self.platforms) - 1 else self.COLOR_PLATFORM
            pygame.draw.rect(self.screen, color, screen_rect, border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.8 for c in color), screen_rect, width=2, border_radius=3)
        
        # Draw player
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        screen_player_rect = player_rect.move(offset)
        
        # Player glow
        glow_rect = screen_player_rect.inflate(self.PLAYER_SIZE, self.PLAYER_SIZE)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_rect.width // 2, glow_rect.height // 2), glow_rect.width // 2)
        self.screen.blit(glow_surf, glow_rect.topleft)

        # Player square
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, screen_player_rect, border_radius=2)

        # Draw jump angle indicator
        if self.is_grounded:
            start_pos = screen_player_rect.center
            if self.jump_direction_idx == 1: end_pos = start_pos + pygame.Vector2(-25, -25)
            elif self.jump_direction_idx == 2: end_pos = start_pos + pygame.Vector2(25, -25)
            elif self.jump_direction_idx == 3: end_pos = start_pos + pygame.Vector2(-25, 25)
            elif self.jump_direction_idx == 4: end_pos = start_pos + pygame.Vector2(25, 25)
            else: end_pos = start_pos
            
            pygame.draw.line(self.screen, self.COLOR_PLAYER, start_pos, end_pos, 2)
            pygame.draw.circle(self.screen, self.COLOR_PLAYER, end_pos, 4)

    def _render_ui(self):
        current_height = int(-self.player_pos.y)
        self.highest_altitude_ever = max(self.highest_altitude_ever, current_height)
        
        # Top left info
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        height_text = self.font_ui.render(f"HEIGHT: {current_height}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(height_text, (10, 30))

        # Top right info
        difficulty_text = self.font_ui.render(f"DIFF: {self.difficulty_multiplier:.2f}x", True, self.COLOR_INFO_TEXT)
        record_text = self.font_ui.render(f"BEST: {self.highest_altitude_ever}", True, self.COLOR_INFO_TEXT)
        self.screen.blit(difficulty_text, (self.SCREEN_WIDTH - difficulty_text.get_width() - 10, 10))
        self.screen.blit(record_text, (self.SCREEN_WIDTH - record_text.get_width() - 10, 30))

        # Combo counter
        if self.combo > 1:
            combo_text = self.font_title.render(f"{self.combo}x COMBO!", True, self.COLOR_PLAYER)
            text_rect = combo_text.get_rect(center=(self.SCREEN_WIDTH / 2, 40))
            self.screen.blit(combo_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": int(-self.player_pos.y),
            "combo": self.combo,
            "difficulty": self.difficulty_multiplier,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It is not part of the required Gymnasium interface
    
    # Unset the dummy video driver to allow for a window to be created
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Geometric Tower Ascent")
    clock = pygame.time.Clock()

    action = [0, 0, 0] # [movement, space, shift]
    total_reward = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        
        # Movement action (sets jump angle)
        move_action = 0 # no-op
        if keys[pygame.K_UP] and keys[pygame.K_LEFT]: move_action = 1
        elif keys[pygame.K_UP] and keys[pygame.K_RIGHT]: move_action = 2
        elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]: move_action = 4
        # Allow single keys
        elif keys[pygame.K_LEFT]: move_action = 1
        elif keys[pygame.K_RIGHT]: move_action = 2
        
        # Space action
        space_action = 1 if keys[pygame.K_SPACE] else 0
        
        # Shift action (unused)
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move_action, space_action, shift_action]
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render to Screen ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        if terminated or truncated:
            print(f"Episode Finished. Total Reward: {total_reward:.2f}, Info: {info}")
            print("Press 'R' to restart.")
            
        pygame.display.flip()
        clock.tick(60) # Run at 60 FPS for smooth human gameplay

    env.close()