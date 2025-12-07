import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:25:01.003973
# Source Brief: brief_00410.md
# Brief Index: 410
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where a morphing blob collects orbs on a dynamic terrain.

    **Gameplay:**
    - The player controls a blob that changes size and speed based on the terrain it's on.
    - **Water (Blue):** Faster speed, smaller size.
    - **Sand (Yellow):** Normal speed, normal size.
    - **Ice (Light Blue):** Slower speed, larger size.
    - The goal is to collect all 8 orbs and reach the green goal circle within the time limit.

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `actions[0]`: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - `actions[1]`: Space button (unused)
    - `actions[2]`: Shift button (unused)

    **Observation Space:** `Box(shape=(400, 640, 3), dtype=np.uint8)`
    - An RGB image of the game screen.

    **Rewards:**
    - +10 for each orb collected.
    - +100 for reaching the goal after collecting all orbs.
    - +0.1 for moving closer to the nearest orb/goal.
    - -0.01 for moving away from the nearest orb/goal.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Control a morphing blob that changes size and speed across different terrains. Collect all the orbs and reach the goal before time runs out."
    user_guide = "Controls: Use the arrow keys (↑↓←→) to move your blob around the map."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1800 # 60 seconds * 30 FPS

    # Colors
    COLOR_BG = (15, 15, 25) # Dark blue-grey
    COLOR_SAND = (60, 50, 40)
    COLOR_WATER = (40, 50, 70)
    COLOR_ICE = (60, 70, 80)
    COLOR_PLAYER = (255, 100, 100)
    COLOR_PLAYER_GLOW = (255, 100, 100, 50)
    COLOR_ORB = (255, 255, 255)
    COLOR_ORB_GLOW = (255, 255, 255, 60)
    COLOR_GOAL = (100, 255, 100)
    COLOR_GOAL_GLOW = (100, 255, 100, 60)
    COLOR_UI_TEXT = (220, 220, 220)

    # Player settings
    PLAYER_BASE_SPEED = 3.0
    PLAYER_BASE_RADIUS = 12
    PLAYER_LERP_RATE = 0.1 # Smoothness of size/speed transitions

    # Game settings
    NUM_ORBS = 8
    ORB_RADIUS = 6
    GOAL_RADIUS = 20

    # Terrain IDs
    TERRAIN_SAND = 0
    TERRAIN_WATER = 1
    TERRAIN_ICE = 2

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
        self.font = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.player_radius = 0.0
        self.player_speed = 0.0
        self.target_radius = 0.0
        self.target_speed = 0.0

        self.orbs = []
        self.goal_pos = np.zeros(2, dtype=np.int32)
        
        self.terrain_map = None
        self.terrain_surface = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_STEPS

        # Generate terrain
        self._generate_terrain()

        # Place player
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.player_radius = float(self.PLAYER_BASE_RADIUS)
        self.player_speed = float(self.PLAYER_BASE_SPEED)
        self.target_radius = self.player_radius
        self.target_speed = self.player_speed

        # Place goal and orbs
        self._place_game_elements()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        self.timer -= 1
        
        # --- 1. Calculate distance to target for reward ---
        dist_before_move = self._get_distance_to_target()

        # --- 2. Process action and update player state ---
        self._handle_input(action)
        self._update_player_physics()

        # --- 3. Check for collisions and game events ---
        reward = 0
        collected_orb = self._check_orb_collection()
        if collected_orb:
            # sfx: orb_collect.wav
            reward += 10.0
            self.score += 10

        # --- 4. Calculate movement reward ---
        dist_after_move = self._get_distance_to_target()
        if dist_after_move < dist_before_move:
            reward += 0.1
        else:
            reward -= 0.01

        # --- 5. Check for termination conditions ---
        terminated = self._check_termination()
        if terminated and not self.game_over: # Win condition
            if len(self.orbs) == 0:
                # sfx: victory.wav
                reward += 100.0
                self.score += 100
            # sfx: game_over.wav
            self.game_over = True
            
        truncated = False # This environment does not truncate based on step limit
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_terrain(self):
        self.terrain_map = np.full((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), self.TERRAIN_SAND, dtype=np.int8)
        self.terrain_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.terrain_surface.fill(self.COLOR_SAND)

        # Generate random patches of water and ice
        for _ in range(self.np_random.integers(5, 10)): # Water patches
            w, h = self.np_random.integers(50, 150), self.np_random.integers(50, 150)
            x, y = self.np_random.integers(0, self.SCREEN_WIDTH - w), self.np_random.integers(0, self.SCREEN_HEIGHT - h)
            self.terrain_map[x:x+w, y:y+h] = self.TERRAIN_WATER
            pygame.draw.rect(self.terrain_surface, self.COLOR_WATER, (x, y, w, h))

        for _ in range(self.np_random.integers(3, 8)): # Ice patches
            w, h = self.np_random.integers(40, 120), self.np_random.integers(40, 120)
            x, y = self.np_random.integers(0, self.SCREEN_WIDTH - w), self.np_random.integers(0, self.SCREEN_HEIGHT - h)
            self.terrain_map[x:x+w, y:y+h] = self.TERRAIN_ICE
            pygame.draw.rect(self.terrain_surface, self.COLOR_ICE, (x, y, w, h))

    def _place_game_elements(self):
        self.orbs = []
        occupied_areas = [pygame.Rect(self.player_pos[0] - 20, self.player_pos[1] - 20, 40, 40)]

        # Place Goal
        while True:
            pos = (self.np_random.integers(50, self.SCREEN_WIDTH - 50), self.np_random.integers(50, self.SCREEN_HEIGHT - 50))
            new_rect = pygame.Rect(pos[0] - self.GOAL_RADIUS, pos[1] - self.GOAL_RADIUS, self.GOAL_RADIUS * 2, self.GOAL_RADIUS * 2)
            if not any(new_rect.colliderect(r) for r in occupied_areas):
                self.goal_pos = np.array(pos)
                occupied_areas.append(new_rect)
                break
        
        # Place Orbs
        for _ in range(self.NUM_ORBS):
            while True:
                pos = (self.np_random.integers(20, self.SCREEN_WIDTH - 20), self.np_random.integers(20, self.SCREEN_HEIGHT - 20))
                new_rect = pygame.Rect(pos[0] - self.ORB_RADIUS, pos[1] - self.ORB_RADIUS, self.ORB_RADIUS * 2, self.ORB_RADIUS * 2)
                if not any(new_rect.colliderect(r) for r in occupied_areas):
                    self.orbs.append(np.array(pos))
                    occupied_areas.append(new_rect)
                    break

    def _handle_input(self, action):
        movement = action[0]
        dx, dy = 0, 0
        if movement == 1: dy = -1 # Up
        elif movement == 2: dy = 1 # Down
        elif movement == 3: dx = -1 # Left
        elif movement == 4: dx = 1 # Right
        
        # Update player position
        self.player_pos[0] += dx * self.player_speed
        self.player_pos[1] += dy * self.player_speed

        # Clamp to screen boundaries
        self.player_pos[0] = np.clip(self.player_pos[0], self.player_radius, self.SCREEN_WIDTH - self.player_radius)
        self.player_pos[1] = np.clip(self.player_pos[1], self.player_radius, self.SCREEN_HEIGHT - self.player_radius)

    def _update_player_physics(self):
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        px = np.clip(px, 0, self.SCREEN_WIDTH - 1)
        py = np.clip(py, 0, self.SCREEN_HEIGHT - 1)
        
        current_terrain = self.terrain_map[px, py]

        if current_terrain == self.TERRAIN_WATER:
            self.target_speed = self.PLAYER_BASE_SPEED * 2.0
            self.target_radius = self.PLAYER_BASE_RADIUS * 0.5
        elif current_terrain == self.TERRAIN_ICE:
            self.target_speed = self.PLAYER_BASE_SPEED * 0.5
            self.target_radius = self.PLAYER_BASE_RADIUS * 1.5
        else: # Sand
            self.target_speed = self.PLAYER_BASE_SPEED
            self.target_radius = self.PLAYER_BASE_RADIUS
        
        # Interpolate for smooth transitions
        self.player_speed += (self.target_speed - self.player_speed) * self.PLAYER_LERP_RATE
        self.player_radius += (self.target_radius - self.player_radius) * self.PLAYER_LERP_RATE

    def _check_orb_collection(self):
        collected_any = False
        remaining_orbs = []
        for orb_pos in self.orbs:
            dist = np.linalg.norm(self.player_pos - orb_pos)
            if dist < self.player_radius + self.ORB_RADIUS:
                collected_any = True
            else:
                remaining_orbs.append(orb_pos)
        self.orbs = remaining_orbs
        return collected_any

    def _get_distance_to_target(self):
        if not self.orbs:
            target_pos = self.goal_pos
        else:
            distances = [np.linalg.norm(self.player_pos - o) for o in self.orbs]
            target_pos = self.orbs[np.argmin(distances)]
        return np.linalg.norm(self.player_pos - target_pos)

    def _check_termination(self):
        if self.timer <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        
        # Victory condition
        if not self.orbs:
            dist_to_goal = np.linalg.norm(self.player_pos - self.goal_pos)
            if dist_to_goal < self.player_radius + self.GOAL_RADIUS:
                return True
        
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render terrain from pre-rendered surface
        self.screen.blit(self.terrain_surface, (0, 0))

        # Render Goal
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 # 0 to 1
        goal_r = int(self.GOAL_RADIUS * (1 + pulse * 0.1))
        glow_r = int(goal_r * 1.5)
        
        if len(self.orbs) == 0: # Goal is active
            pygame.gfxdraw.filled_circle(self.screen, self.goal_pos[0], self.goal_pos[1], glow_r, self.COLOR_GOAL_GLOW)
            pygame.gfxdraw.aacircle(self.screen, self.goal_pos[0], self.goal_pos[1], glow_r, self.COLOR_GOAL_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, self.goal_pos[0], self.goal_pos[1], goal_r, self.COLOR_GOAL)
        pygame.gfxdraw.aacircle(self.screen, self.goal_pos[0], self.goal_pos[1], goal_r, self.COLOR_GOAL)

        # Render Orbs
        orb_pulse = (math.sin(self.steps * 0.2) + 1) / 2
        orb_glow_r = int(self.ORB_RADIUS * (1.5 + orb_pulse * 0.5))
        for orb_pos in self.orbs:
            pygame.gfxdraw.filled_circle(self.screen, orb_pos[0], orb_pos[1], orb_glow_r, self.COLOR_ORB_GLOW)
            pygame.gfxdraw.aacircle(self.screen, orb_pos[0], orb_pos[1], orb_glow_r, self.COLOR_ORB_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, orb_pos[0], orb_pos[1], self.ORB_RADIUS, self.COLOR_ORB)
            pygame.gfxdraw.aacircle(self.screen, orb_pos[0], orb_pos[1], self.ORB_RADIUS, self.COLOR_ORB)

        # Render Player (Blob)
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        pr = int(self.player_radius)
        
        # Glow effect
        glow_radius = int(pr * 1.8)
        pygame.gfxdraw.filled_circle(self.screen, px, py, glow_radius, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, px, py, glow_radius, self.COLOR_PLAYER_GLOW)
        
        # Morphing effect
        num_blobs = 5
        for i in range(num_blobs):
            angle = (self.steps * 0.05) + (i * 2 * math.pi / num_blobs)
            offset_x = math.cos(angle) * pr * 0.2
            offset_y = math.sin(angle) * pr * 0.2
            sub_radius = int(pr * (0.6 + (math.sin(angle*2) + 1) * 0.1))
            
            # Use a slightly darker, more transparent color for the sub-blobs
            sub_blob_color = (*self.COLOR_PLAYER, 100)
            
            # This requires a surface with per-pixel alpha to draw correctly
            temp_surface = pygame.Surface((sub_radius*2, sub_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surface, sub_blob_color, (sub_radius, sub_radius), sub_radius)
            self.screen.blit(temp_surface, (px + offset_x - sub_radius, py + offset_y - sub_radius))

        # Main body
        pygame.gfxdraw.filled_circle(self.screen, px, py, pr, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, pr, self.COLOR_PLAYER)

    def _render_ui(self):
        # Orb counter
        orb_text = f"Orbs: {self.NUM_ORBS - len(self.orbs)} / {self.NUM_ORBS}"
        text_surf = self.font.render(orb_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Timer
        seconds_left = max(0, self.timer / self.FPS)
        timer_text = f"Time: {seconds_left:.1f}"
        timer_color = self.COLOR_UI_TEXT if seconds_left > 10 else (255, 50, 50)
        text_surf = self.font.render(timer_text, True, timer_color)
        text_rect = text_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(text_surf, text_rect)
        
        # Game Over message
        if self.game_over:
            message = ""
            if len(self.orbs) == 0:
                message = "GOAL REACHED!"
            elif self.timer <= 0:
                message = "TIME UP!"
            
            if message:
                end_surf = self.font.render(message, True, self.COLOR_UI_TEXT)
                end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
                self.screen.blit(end_surf, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "orbs_collected": self.NUM_ORBS - len(self.orbs),
            "timer_remaining": self.timer
        }

    def close(self):
        pygame.quit()

# Example usage:
if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Re-initialize pygame for display
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.init()
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Blob Collector")
    clock = pygame.time.Clock()

    while not done:
        # Map keyboard keys to actions
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle window closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        clock.tick(env.FPS)
        
        if done:
            print(f"Game Over! Final Info: {info}")
            # Wait a bit before closing or resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False

    env.close()