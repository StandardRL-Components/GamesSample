import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:49:12.017945
# Source Brief: brief_03089.md
# Brief Index: 3089
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent must guide a flock of birds.

    The goal is to maintain flock cohesion for 60 seconds. The agent influences
    the flock's movement by setting a target point. Birds are lost if they
    stray too far from the flock's center (centroid).

    **Visuals:**
    - A dark, minimalist background.
    - Birds are rendered as colored circles, with color indicating speed.
    - A green semi-transparent circle shows the target cohesion area.
    - A red semi-transparent circle shows the dispersion limit where birds are lost.
    - The camera is always centered on the flock's centroid.

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `action[0]` (Movement): Not used.
    - `action[1]` (Space): If 1, sets the target to the screen center.
    - `action[2]` (Shift): If 1, sets the target to a random screen position.
      (Space action takes priority if both are active).

    **Reward Structure:**
    - +1 per step if all birds are within the cohesion radius.
    - -10 for each bird lost.
    - +100 for successfully surviving the full duration.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide a flock of birds by setting a target point, trying to keep them together. "
        "Survive for 60 seconds without letting the birds stray too far from the flock's center."
    )
    user_guide = (
        "Controls: Press Space to set the flock's target to the center. "
        "Hold Shift to set the target to a random position."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_STEPS = 60 * FPS

    INITIAL_FLOCK_SIZE = 15
    BIRD_RADIUS = 5
    BIRD_MAX_SPEED_RANGE = (1.5, 3.5)
    BIRD_MAX_FORCE = 0.1  # Steering force limit

    COHESION_RADIUS = 150  # pixels
    DISPERSION_RADIUS = 300 # pixels
    INITIAL_SPAWN_RADIUS = 100 # pixels

    # --- Colors ---
    COLOR_BG = (15, 23, 42)  # Dark Slate Blue
    COLOR_COHESION = (34, 197, 94, 100)  # Green, with alpha
    COLOR_DISPERSION = (239, 68, 68, 80)  # Red, with alpha
    COLOR_CENTROID = (255, 255, 255)
    COLOR_UI_TEXT = (226, 232, 240)  # Light Slate Gray
    
    # Speed-to-color gradient (Blue -> Yellow -> Red)
    COLOR_SPEED_MIN = pygame.Color(59, 130, 246) # Blue
    COLOR_SPEED_MAX = pygame.Color(220, 38, 38)  # Red

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
        self.font = pygame.font.SysFont("Consolas", 20, bold=True)

        self.birds = []
        self.target_pos = pygame.Vector2(0, 0)
        self.steps = 0
        self.score = 0
        self.game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.target_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        
        self.birds = []
        initial_center = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        
        for _ in range(self.INITIAL_FLOCK_SIZE):
            angle = self.np_random.uniform(0, 2 * math.pi)
            radius = self.np_random.uniform(0, self.INITIAL_SPAWN_RADIUS)
            pos = initial_center + pygame.Vector2(math.cos(angle), math.sin(angle)) * radius
            
            vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
            if vel.length() > 0:
                vel.normalize_ip()
            
            max_speed = self.np_random.uniform(*self.BIRD_MAX_SPEED_RANGE)
            
            self.birds.append({
                "pos": pos,
                "vel": vel * max_speed,
                "max_speed": max_speed
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        
        # 1. Handle Action
        _movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if space_held:
            # Action: Center target
            self.target_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        elif shift_held:
            # Action: Random target
            self.target_pos = pygame.Vector2(
                self.np_random.uniform(0, self.SCREEN_WIDTH),
                self.np_random.uniform(0, self.SCREEN_HEIGHT)
            )
        # If neither is held, target_pos persists

        # 2. Update Game Logic
        birds_lost_this_step = self._update_flock()

        # 3. Calculate Reward
        reward = self._calculate_reward(birds_lost_this_step)
        self.score += reward

        # 4. Check Termination
        terminated = self._check_termination()
        truncated = False # No truncation condition in this game
        if terminated:
            self.game_over = True
            # Final goal reward
            if self.steps >= self.MAX_STEPS and len(self.birds) > 0:
                reward += 100
                self.score += 100

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_flock(self):
        if not self.birds:
            return 0

        # Update each bird's velocity and position
        for bird in self.birds:
            # Steering behavior: steer towards target
            desired_vel = self.target_pos - bird["pos"]
            if desired_vel.length() > 0:
                desired_vel.normalize_ip()
                desired_vel *= bird["max_speed"]
            
            steer = desired_vel - bird["vel"]
            if steer.length() > self.BIRD_MAX_FORCE:
                steer.scale_to_length(self.BIRD_MAX_FORCE)
            
            bird["vel"] += steer
            if bird["vel"].length() > bird["max_speed"]:
                bird["vel"].scale_to_length(bird["max_speed"])
            
            bird["pos"] += bird["vel"]
        
        # Cull birds that are too far from the new centroid
        centroid = self._calculate_centroid()
        initial_bird_count = len(self.birds)
        
        surviving_birds = []
        for bird in self.birds:
            if bird["pos"].distance_to(centroid) < self.DISPERSION_RADIUS:
                surviving_birds.append(bird)
        
        self.birds = surviving_birds
        return initial_bird_count - len(self.birds)

    def _calculate_centroid(self):
        if not self.birds:
            return pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        return sum([b['pos'] for b in self.birds], pygame.Vector2()) / len(self.birds)

    def _calculate_reward(self, birds_lost_this_step):
        reward = 0
        # Penalty for losing birds
        if birds_lost_this_step > 0:
            reward -= 10 * birds_lost_this_step

        # Reward for maintaining cohesion
        if self.birds:
            centroid = self._calculate_centroid()
            is_cohesive = all(b['pos'].distance_to(centroid) < self.COHESION_RADIUS for b in self.birds)
            if is_cohesive:
                reward += 1
        
        return float(reward)

    def _check_termination(self):
        return len(self.birds) == 0 or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        if not self.birds and not self.game_over:
            return # Avoids division by zero if birds list is empty during reset
            
        centroid = self._calculate_centroid()
        render_offset = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2) - centroid
        
        # --- Render background elements (circles) ---
        center_screen = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        
        # Create temporary surfaces for transparency
        s1 = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s2 = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        
        # Dispersion Circle (Red)
        pygame.gfxdraw.filled_circle(s1, center_screen[0], center_screen[1], self.DISPERSION_RADIUS, self.COLOR_DISPERSION)
        # Cohesion Circle (Green)
        pygame.gfxdraw.filled_circle(s2, center_screen[0], center_screen[1], self.COHESION_RADIUS, self.COLOR_COHESION)

        self.screen.blit(s1, (0,0))
        self.screen.blit(s2, (0,0))

        # --- Render foreground elements (birds) ---
        for bird in self.birds:
            render_pos = bird["pos"] + render_offset
            
            # Interpolate color based on speed
            speed_range_width = self.BIRD_MAX_SPEED_RANGE[1] - self.BIRD_MAX_SPEED_RANGE[0]
            if speed_range_width > 0:
                speed_ratio = (bird['max_speed'] - self.BIRD_MAX_SPEED_RANGE[0]) / speed_range_width
            else:
                speed_ratio = 0.5
            speed_ratio = max(0, min(1, speed_ratio))
            bird_color = self.COLOR_SPEED_MIN.lerp(self.COLOR_SPEED_MAX, speed_ratio)

            # Glow effect
            glow_radius = int(self.BIRD_RADIUS * 1.8)
            glow_color = (*bird_color[:3], 60) # Add alpha
            pygame.gfxdraw.filled_circle(self.screen, int(render_pos.x), int(render_pos.y), glow_radius, glow_color)
            pygame.gfxdraw.aacircle(self.screen, int(render_pos.x), int(render_pos.y), glow_radius, glow_color)
            
            # Main bird circle
            pygame.gfxdraw.filled_circle(self.screen, int(render_pos.x), int(render_pos.y), self.BIRD_RADIUS, bird_color)
            pygame.gfxdraw.aacircle(self.screen, int(render_pos.x), int(render_pos.y), self.BIRD_RADIUS, bird_color)

        # Render Centroid
        pygame.gfxdraw.filled_circle(self.screen, center_screen[0], center_screen[1], 3, self.COLOR_CENTROID)
        pygame.gfxdraw.aacircle(self.screen, center_screen[0], center_screen[1], 3, self.COLOR_CENTROID)

    def _render_ui(self):
        # Time remaining
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = f"TIME: {time_left:.2f}"
        time_surf = self.font.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Birds remaining
        birds_text = f"BIRDS: {len(self.birds)}/{self.INITIAL_FLOCK_SIZE}"
        birds_surf = self.font.render(birds_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(birds_surf, (self.SCREEN_WIDTH - birds_surf.get_width() - 10, 10))
        
        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, self.SCREEN_HEIGHT - score_surf.get_height() - 10))

        if self.game_over:
            message = ""
            if len(self.birds) == 0:
                message = "FLOCK LOST"
            elif self.steps >= self.MAX_STEPS:
                message = "COHESION MAINTAINED"
            
            end_surf = self.font.render(message, True, self.COLOR_UI_TEXT)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 50))
            self.screen.blit(end_surf, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "birds_remaining": len(self.birds),
            "is_cohesive": self._is_flock_cohesive() if self.birds else False,
        }
        
    def _is_flock_cohesive(self):
        if not self.birds:
            return False
        centroid = self._calculate_centroid()
        return all(b['pos'].distance_to(centroid) < self.COHESION_RADIUS for b in self.birds)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    game_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Flock Cohesion")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)

    env.close()