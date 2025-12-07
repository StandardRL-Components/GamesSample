# Generated: 2025-08-27T20:57:31.942258
# Source Brief: brief_02632.md
# Brief Index: 2632

        
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
        "Controls: ↑ to jump, ←→ to move. Reach the end of all three stages before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A procedurally generated platform jumper where the player must navigate a series of gaps and obstacles to reach the end, balancing speed with safety."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 50

    # Colors
    COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
    COLOR_BG_BOTTOM = (0, 0, 139)      # Dark Blue
    COLOR_PLAYER = (0, 160, 255)
    COLOR_PLAYER_OUTLINE = (255, 255, 255)
    COLOR_PLATFORM = (128, 128, 128)
    COLOR_PLATFORM_OUTLINE = (64, 64, 64)
    COLOR_OBSTACLE = (255, 64, 64)
    COLOR_OBSTACLE_OUTLINE = (128, 0, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)
    COLOR_GOAL = (255, 215, 0) # Gold

    # Player Physics
    PLAYER_SIZE = 24
    MOVE_SPEED = 4
    GRAVITY = 0.5
    JUMP_STRENGTH = -11
    FRICTION = 0.85

    # Game Parameters
    MAX_STAGES = 3
    TIME_LIMIT_SECONDS = 120
    MAX_STEPS = TIME_LIMIT_SECONDS * TARGET_FPS

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 28)
        
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.current_platform_idx = -1
        
        self.platforms = []
        self.obstacles = []
        
        self.camera_x = 0.0
        
        self.steps = 0
        self.score = 0
        self.current_stage = 1
        self.game_over = False
        self.game_won = False
        self.terminal_reason = ""
        
        # self.reset() is called by the test harness, no need to call here.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.current_stage = 1
        self.game_over = False
        self.game_won = False
        self.terminal_reason = ""
        
        self._generate_stage(self.current_stage)
        
        self.player_pos = pygame.Vector2(100, 200)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.current_platform_idx = 0
        
        self.camera_x = 0

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        truncated = False # Gymnasium expects truncated to be returned
        
        if self.game_over:
            # Return a valid state even if the game is over
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        # --- Handle Input ---
        movement = action[0]
        self._handle_input(movement)

        # --- Update Game State ---
        self._update_player()
        self._update_obstacles()
        
        # --- Collision Detection ---
        landing_info = self._check_collisions()
        
        # --- Calculate Rewards ---
        reward += 0.01  # Survival reward

        if landing_info["landed_on_new_platform"]:
            reward += 1.0
            # sfx: new_platform_ding.wav
        if landing_info["is_risky_landing"]:
            reward -= 0.1 # Reduced penalty to avoid large negative rewards for minor mistakes
        
        # --- Check for Stage Completion ---
        if self.on_ground and self.current_platform_idx == len(self.platforms) - 1:
            if self.current_stage < self.MAX_STAGES:
                self.current_stage += 1
                self._generate_stage(self.current_stage)
                # Ensure player starts on the new platform correctly
                start_plat = self.platforms[0]
                self.player_pos.x = start_plat.centerx
                self.player_pos.y = start_plat.top - self.PLAYER_SIZE
                self.player_vel.x = 0
                self.player_vel.y = 0
                self.current_platform_idx = 0
                self.on_ground = True
                reward += 10.0
                # sfx: stage_complete.wav
            else:
                self.game_won = True
                self.game_over = True
                self.terminal_reason = "YOU WIN!"
                reward = 100.0
                terminated = True
                # sfx: victory_fanfare.wav

        # --- Check Termination Conditions ---
        if not terminated:
            if self.player_pos.y > self.SCREEN_HEIGHT + 50:
                self.game_over = True
                self.terminal_reason = "FELL"
                reward = -10.0
                terminated = True
                # sfx: fall_scream.wav
            elif landing_info["hit_obstacle"]:
                self.game_over = True
                self.terminal_reason = "HIT OBSTACLE"
                reward = -10.0
                terminated = True
                # sfx: player_hit.wav
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            if not terminated: # Don't override a win
                self.game_over = True
                self.terminal_reason = "TIME UP"
                truncated = True # Use truncated for time limit
            
        if terminated or truncated:
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement):
        # Apply friction
        self.player_vel.x *= self.FRICTION

        if movement == 1: # Up
            if self.on_ground:
                self.player_vel.y = self.JUMP_STRENGTH
                self.on_ground = False
                # sfx: jump.wav
        # movement == 2 (Down) has no effect
        elif movement == 3: # Left
            self.player_vel.x = -self.MOVE_SPEED
        elif movement == 4: # Right
            self.player_vel.x = self.MOVE_SPEED

    def _update_player(self):
        # Apply gravity
        if not self.on_ground:
            self.player_vel.y += self.GRAVITY
        
        # Clamp max fall speed
        self.player_vel.y = min(self.player_vel.y, 15)

        # Update position
        self.player_pos += self.player_vel

    def _update_obstacles(self):
        # Difficulty scaling
        obstacle_speed_multiplier = 1.0 + (self.steps / 500) * 0.05

        for obs in self.obstacles:
            obs['angle'] = (obs['angle'] + obs['rot_speed']) % 360
            obs['pos'].x += obs['vel'].x * obstacle_speed_multiplier
            
            # Bounce off platform edges
            platform = self.platforms[obs['platform_idx']]
            if obs['pos'].x < platform.left + obs['size'] / 2 or obs['pos'].x > platform.right - obs['size'] / 2:
                obs['vel'].x *= -1

    def _check_collisions(self):
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.on_ground = False
        
        landing_info = {
            "landed_on_new_platform": False,
            "is_risky_landing": False,
            "hit_obstacle": False,
        }

        # Platform collisions
        for i, plat in enumerate(self.platforms):
            if player_rect.colliderect(plat):
                # Check if player was above or on the platform and moving downwards or stationary.
                if self.player_vel.y >= 0 and (self.player_pos.y + self.PLAYER_SIZE - self.player_vel.y) <= plat.top + 1:
                    self.player_pos.y = plat.top - self.PLAYER_SIZE
                    self.player_vel.y = 0
                    self.on_ground = True
                    
                    if self.current_platform_idx != i:
                        landing_info["landed_on_new_platform"] = True
                        self.current_platform_idx = i
                    
                    # Check for risky landing
                    risky_margin = self.PLAYER_SIZE * 0.1
                    if (player_rect.left < plat.left + risky_margin) or (player_rect.right > plat.right - risky_margin):
                        landing_info["is_risky_landing"] = True
                    
                    break # Only interact with one platform at a time
                # Hitting platform from below
                elif self.player_vel.y < 0 and self.player_pos.y - self.player_vel.y >= plat.bottom:
                    self.player_pos.y = plat.bottom
                    self.player_vel.y = 0
                    # sfx: bonk.wav
                # Side collision
                elif self.player_vel.x != 0:
                    if self.player_vel.x > 0: # Moving right
                        self.player_pos.x = plat.left - self.PLAYER_SIZE
                    elif self.player_vel.x < 0: # Moving left
                        self.player_pos.x = plat.right
                    self.player_vel.x = 0
        
        # Obstacle collisions
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs['pos'].x - obs['size']/2, obs['pos'].y - obs['size']/2, obs['size'], obs['size'])
            if player_rect.colliderect(obs_rect):
                landing_info["hit_obstacle"] = True
                break

        return landing_info

    def _generate_stage(self, stage_num):
        self.platforms.clear()
        self.obstacles.clear()
        
        plat_len_mod = 1.0 - (stage_num - 1) * 0.05
        gap_mod = 1.0 + (stage_num - 1) * 0.05
        
        current_x = 0
        current_y = 350
        
        # Starting platform
        start_plat_width = 200
        self.platforms.append(pygame.Rect(current_x, current_y, start_plat_width, 100))
        current_x += start_plat_width
        
        # Procedural platforms
        num_platforms = 15
        for i in range(num_platforms):
            gap = self.np_random.integers(60, 120) * gap_mod
            width = self.np_random.integers(100, 250) * plat_len_mod
            height_diff = self.np_random.integers(-60, 60)
            
            current_x += gap
            current_y = np.clip(current_y + height_diff, 150, 360)
            
            new_plat = pygame.Rect(current_x, current_y, width, 100)
            self.platforms.append(new_plat)

            # Add obstacles with increasing probability
            if self.np_random.random() < 0.3 + 0.15 * stage_num:
                obs_size = self.np_random.integers(20, 30)
                self.obstacles.append({
                    'pos': pygame.Vector2(new_plat.centerx, new_plat.top - obs_size / 2),
                    'vel': pygame.Vector2(self.np_random.choice([-1.5, 1.5]), 0),
                    'size': obs_size,
                    'angle': 0,
                    'rot_speed': self.np_random.uniform(-5, 5),
                    'platform_idx': len(self.platforms) - 1
                })

            current_x += width
        
        # Final platform
        final_plat = pygame.Rect(current_x + 100, 250, 200, 150)
        self.platforms.append(final_plat)

    def _update_camera(self):
        target_camera_x = self.player_pos.x - self.SCREEN_WIDTH / 3
        # Smooth camera movement using linear interpolation (lerp)
        self.camera_x += (target_camera_x - self.camera_x) * 0.1

    def _get_observation(self):
        self._update_camera()
        self._render_background()
        self._render_platforms_and_obstacles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))
    
    def _render_background(self):
        # Draw a gradient
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_platforms_and_obstacles(self):
        cam_x_int = int(self.camera_x)

        # Platforms
        for i, plat in enumerate(self.platforms):
            render_rect = plat.move(-cam_x_int, 0)
            if render_rect.right < 0 or render_rect.left > self.SCREEN_WIDTH:
                continue
            
            color = self.COLOR_GOAL if i == len(self.platforms) - 1 else self.COLOR_PLATFORM
            pygame.draw.rect(self.screen, color, render_rect)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, render_rect, 3)

        # Obstacles
        for obs in self.obstacles:
            render_x = obs['pos'].x - cam_x_int
            if render_x < -obs['size'] or render_x > self.SCREEN_WIDTH + obs['size']:
                continue

            # Calculate rotated triangle points
            points = []
            for i in range(3):
                angle_rad = math.radians(obs['angle'] + i * 120)
                x = render_x + obs['size'] * 0.7 * math.cos(angle_rad)
                y = obs['pos'].y + obs['size'] * 0.7 * math.sin(angle_rad)
                points.append((int(x), int(y)))
            
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_OBSTACLE_OUTLINE)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_OBSTACLE)

    def _render_player(self):
        cam_x_int = int(self.camera_x)
        player_rect = pygame.Rect(
            int(self.player_pos.x - cam_x_int),
            int(self.player_pos.y),
            self.PLAYER_SIZE,
            self.PLAYER_SIZE
        )
        
        # Glow effect
        glow_size = self.PLAYER_SIZE + 10
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PLAYER, 60), glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, (player_rect.centerx - glow_size/2, player_rect.centery - glow_size/2))

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect, 2, border_radius=4)
        
    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            content = font.render(text, True, color)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(content, pos)

        # Score
        score_text = f"SCORE: {int(self.score)}"
        draw_text(score_text, self.font_small, self.COLOR_TEXT, (10, 10))

        # Time
        time_left = max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.TARGET_FPS))
        time_text = f"TIME: {int(time_left // 60):02}:{int(time_left % 60):02}"
        text_width = self.font_small.size(time_text)[0]
        draw_text(time_text, self.font_small, self.COLOR_TEXT, (self.SCREEN_WIDTH - text_width - 10, 10))

        # Stage
        stage_text = f"STAGE: {self.current_stage}/{self.MAX_STAGES}"
        text_width = self.font_small.size(stage_text)[0]
        draw_text(stage_text, self.font_small, self.COLOR_TEXT, (self.SCREEN_WIDTH // 2 - text_width // 2, 10))
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.terminal_reason
            text_surf = self.font_large.render(end_text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "player_x": self.player_pos.x,
            "player_y": self.player_pos.y,
        }

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # We can't use pygame.display in a headless environment, but we can simulate a game loop.
    print("--- Game Environment ---")
    print("Description:", env.game_description)
    print("Controls:", env.user_guide)
    print("Action Space:", env.action_space)
    print("Observation Space:", env.observation_space)
    
    obs, info = env.reset(seed=42)
    terminated = False
    truncated = False
    total_reward = 0
    
    # Simulate a few steps with random actions
    for i in range(200):
        # No-op action for stability check
        # action = [0, 0, 0] 
        action = env.action_space.sample() # Replace with your logic
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if (i % 50) == 0:
            print(f"Step {i}: Action={action}, Reward={reward:.2f}, Info={info}")

        if terminated or truncated:
            print(f"Episode finished after {i+1} steps. Final Score: {info['score']:.2f}")
            obs, info = env.reset(seed=42)
            total_reward = 0

    env.close()