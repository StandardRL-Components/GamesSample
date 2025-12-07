import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T09:44:53.182141
# Source Brief: brief_00053.md
# Brief Index: 53
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gravity Well is a physics-based puzzle game. The player controls a cursor
    to create temporary gravity wells. The goal is to guide three colored blocks
    into their matching target zones at the bottom of the screen. Gravity wells
    have a limited duration and solidify into permanent obstacles, adding to the
    challenge. The game features multiple levels with increasing difficulty.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide colored blocks into their matching target zones by creating temporary gravity wells. "
        "Wells solidify into obstacles, so place them wisely!"
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor and press space to create a gravity well."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 5000
    MAX_LEVELS = 5

    # Colors
    COLOR_BG = (15, 23, 42)  # Slate 900
    COLOR_CURSOR = (255, 255, 255)
    COLOR_OBSTACLE = (100, 116, 139) # Slate 500
    COLOR_UI_TEXT = (226, 232, 240) # Slate 200
    BLOCK_COLORS = [
        (234, 88, 12),    # Orange 600
        (217, 70, 239),   # Fuchsia 500
        (74, 222, 128),   # Green 400
    ]
    ZONE_ALPHA = 64

    # Game Parameters
    CURSOR_SPEED = 8
    BLOCK_RADIUS = 12
    GRAVITY_WELL_STRENGTH = 1500
    GRAVITY_WELL_DURATION = 5 * FPS # 5 seconds
    GRAVITY_WELL_COST = 10
    PHYSICS_DAMPING = 0.985
    SETTLE_VELOCITY_THRESHOLD = 0.15

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.SysFont('Consolas', 24, bold=True)
            self.font_small = pygame.font.SysFont('Consolas', 18)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)
            
        # --- Game State Variables (initialized in reset) ---
        self.level = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grav_strength = 0
        self.creation_point = pygame.Vector2(0, 0)
        self.blocks = []
        self.gravity_wells = []
        self.obstacles = []
        self.last_space_held = False

        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.level = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self._setup_level()

        return self._get_observation(), self._get_info()

    def _setup_level(self):
        """Initializes the state for the current level."""
        self.creation_point = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 4)
        self.grav_strength = 55 - (self.level * 5)
        self.blocks = []
        self.gravity_wells = []
        self.obstacles = []

        num_blocks = 3
        zone_width = self.SCREEN_WIDTH / num_blocks
        for i in range(num_blocks):
            # Target Zone
            zone_rect = pygame.Rect(i * zone_width, self.SCREEN_HEIGHT - 50, zone_width, 50)
            
            # Block
            start_x = self.np_random.uniform(20, self.SCREEN_WIDTH - 20)
            start_y = self.np_random.uniform(20, 80)
            
            block = {
                "pos": pygame.Vector2(start_x, start_y),
                "vel": pygame.Vector2(0, 0),
                "color": self.BLOCK_COLORS[i],
                "target_zone": zone_rect,
                "settled": False,
                "prev_dist": 0,
            }
            block["prev_dist"] = block["pos"].distance_to(zone_rect.center)
            self.blocks.append(block)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        reward = 0.0
        
        self._handle_input(action)
        
        # --- Update Game State ---
        loss_condition_met, settled_count = self._update_game_state()
        reward += settled_count * 1.0 # +1 for each block settled this frame

        # --- Calculate Distance-Based Reward ---
        for block in self.blocks:
            if not block["settled"]:
                current_dist = block["pos"].distance_to(block["target_zone"].center)
                # Reward is proportional to the reduction in distance
                reward += (block["prev_dist"] - current_dist) * 0.01
                block["prev_dist"] = current_dist
        
        # --- Check for Termination Conditions ---
        self.steps += 1
        terminated = False
        truncated = False

        if loss_condition_met:
            self.game_over = True
            terminated = True
            reward = -10.0
            self.score -= 10

        elif all(b['settled'] for b in self.blocks):
            reward += 10.0
            self.score += 10
            self.level += 1
            if self.level > self.MAX_LEVELS:
                reward += 100.0 # Final victory bonus
                self.score += 100
                self.game_over = True
                terminated = True
            else:
                self._setup_level() # Next level

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            truncated = True # Use truncated for time limit
            if not terminated: # Don't overwrite a win/loss reward
                reward = -10.0 # Timeout is a loss
                self.score -= 10

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_action, shift_action = action[0], action[1], action[2]

        # --- Move Creation Point ---
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1 # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1 # Right
        self.creation_point += move_vec * self.CURSOR_SPEED
        self.creation_point.x = np.clip(self.creation_point.x, 0, self.SCREEN_WIDTH)
        self.creation_point.y = np.clip(self.creation_point.y, 0, self.SCREEN_HEIGHT)

        # --- Create Gravity Well ---
        space_pressed = (space_action == 1)
        if space_pressed and not self.last_space_held and self.grav_strength >= self.GRAVITY_WELL_COST:
            # SFX: Well Creation Sound
            self.grav_strength -= self.GRAVITY_WELL_COST
            self.gravity_wells.append({
                "pos": self.creation_point.copy(),
                "age": 0,
            })
        self.last_space_held = space_pressed

    def _update_game_state(self):
        """Updates all entities and checks for game-ending conditions."""
        # --- Update and Solidify Gravity Wells ---
        new_obstacles = []
        active_wells = []
        for well in self.gravity_wells:
            well["age"] += 1
            if well["age"] > self.GRAVITY_WELL_DURATION:
                # SFX: Well Solidify Sound
                obstacle_rect = pygame.Rect(
                    well["pos"].x - self.BLOCK_RADIUS,
                    well["pos"].y - self.BLOCK_RADIUS,
                    self.BLOCK_RADIUS * 2,
                    self.BLOCK_RADIUS * 2
                )
                new_obstacles.append(obstacle_rect)
            else:
                active_wells.append(well)
        self.gravity_wells = active_wells
        self.obstacles.extend(new_obstacles)

        # --- Update Blocks ---
        loss_condition_met = False
        settled_this_frame = 0
        for block in self.blocks:
            if block["settled"]:
                continue

            # Calculate forces from gravity wells
            total_force = pygame.Vector2(0, 0)
            for well in self.gravity_wells:
                vec_to_well = well["pos"] - block["pos"]
                dist_sq = vec_to_well.length_squared()
                if dist_sq > 1: # Avoid division by zero
                    force_magnitude = self.GRAVITY_WELL_STRENGTH / dist_sq
                    total_force += vec_to_well.normalize() * force_magnitude
            
            # Update velocity and position
            block["vel"] += total_force
            block["vel"] *= self.PHYSICS_DAMPING
            block["pos"] += block["vel"]

            # Check for collisions and out-of-bounds
            if not (0 < block["pos"].x < self.SCREEN_WIDTH and 0 < block["pos"].y < self.SCREEN_HEIGHT):
                loss_condition_met = True # SFX: Block Lost Sound
                break
            
            block_rect = pygame.Rect(block["pos"].x - self.BLOCK_RADIUS, block["pos"].y - self.BLOCK_RADIUS, self.BLOCK_RADIUS * 2, self.BLOCK_RADIUS * 2)
            if block_rect.collidelist(self.obstacles) != -1:
                loss_condition_met = True # SFX: Block Collision Sound
                break

            # Check for settling in target zone
            if block["target_zone"].collidepoint(block["pos"]) and block["vel"].length() < self.SETTLE_VELOCITY_THRESHOLD:
                block["settled"] = True
                block["pos"] = pygame.Vector2(block["target_zone"].center) # Snap to center
                settled_this_frame += 1 # SFX: Block Settled Sound

        return loss_condition_met, settled_this_frame

    def _get_observation(self):
        # --- Clear screen ---
        self.screen.fill(self.COLOR_BG)

        # --- Render game elements ---
        self._render_game()
        
        # --- Render UI overlay ---
        self._render_ui()
        
        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Render Target Zones ---
        for block in self.blocks:
            zone_color = block["color"] + (self.ZONE_ALPHA,)
            zone_surface = pygame.Surface(block["target_zone"].size, pygame.SRCALPHA)
            zone_surface.fill(zone_color)
            self.screen.blit(zone_surface, block["target_zone"].topleft)
    
        # --- Render Obstacles ---
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs, border_radius=3)

        # --- Render Gravity Wells ---
        for well in self.gravity_wells:
            age_ratio = well["age"] / self.GRAVITY_WELL_DURATION
            max_radius = 60
            current_radius = int(max_radius * math.sin(age_ratio * math.pi)) # Pulses in and out
            
            for i in range(5, 0, -1):
                radius = int(current_radius * (i / 5))
                alpha = int(150 * (1 - age_ratio) * (i/10 + 0.5))
                if radius > 0 and alpha > 0:
                    pygame.gfxdraw.aacircle(self.screen, int(well["pos"].x), int(well["pos"].y), radius, (173, 216, 230, alpha))

        # --- Render Blocks ---
        for block in self.blocks:
            pos_int = (int(block["pos"].x), int(block["pos"].y))
            # Draw outline
            pygame.draw.circle(self.screen, (255, 255, 255), pos_int, self.BLOCK_RADIUS + 1)
            # Draw block
            pygame.draw.circle(self.screen, block["color"], pos_int, self.BLOCK_RADIUS)

        # --- Render Creation Point Cursor ---
        x, y = int(self.creation_point.x), int(self.creation_point.y)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x - 10, y), (x + 10, y), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x, y - 10), (x, y + 10), 2)

    def _render_ui(self):
        # --- Level Text ---
        level_text = self.font_large.render(f"Level: {self.level}/{self.MAX_LEVELS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (10, 10))

        # --- Gravitational Strength Text ---
        strength_text = self.font_large.render(f"Strength: {self.grav_strength}", True, self.COLOR_UI_TEXT)
        self.screen.blit(strength_text, (self.SCREEN_WIDTH - strength_text.get_width() - 10, 10))
        
        # --- Score Text ---
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "grav_strength": self.grav_strength,
            "active_wells": len(self.gravity_wells),
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # --- Manual Play Testing ---
    # Un-comment the line below to run with display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gravity Well")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    terminated = False
    truncated = False
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("R: Reset environment")
    print("Q: Quit")
    
    while running:
        # --- Action Mapping from Keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    print("--- Resetting Environment ---")
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False
                    truncated = False

        if not terminated and not truncated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"--- Episode Finished ---")
                print(f"Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)
        
    env.close()