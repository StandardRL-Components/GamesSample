import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:52:09.140572
# Source Brief: brief_02489.md
# Brief Index: 2489
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Cultivate a fractal ecosystem by planting and harvesting resources. Evade hostile flora using "
        "camouflage and gather enough energy to ascend to the next level."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press Space to plant a seed or harvest a mature plant. "
        "Press Shift to activate camouflage and evade flora."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_FLORA = (255, 50, 100)
    COLOR_FLORA_GLOW = (255, 50, 100, 60)
    COLOR_PLANT_IMMATURE = (200, 200, 0)
    COLOR_PLANT_MATURE = (100, 255, 100)
    COLOR_CAMO_EFFECT = (170, 0, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_FRACTAL_BASE = (40, 50, 80)

    # Screen Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Game Parameters
    MAX_STEPS = 5000
    MAX_LEVELS = 10
    PLAYER_SPEED = 10
    PLAYER_RADIUS = 10
    FLORA_RADIUS = 12
    PLANT_RADIUS = 6
    
    # Resource & Action Costs/Timings
    INITIAL_RESOURCES = 20
    PLANT_COST = 10
    PLANT_GROWTH_TIME = 150 # steps
    PLANT_HARVEST_YIELD = 15
    CAMO_COST = 25
    CAMO_DURATION = 200 # steps
    LEVEL_UP_BASE_COST = 50
    LEVEL_UP_COST_MULTIPLIER = 1.5
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_level = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- State Variables ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = pygame.Vector2(0, 0)
        self.resources = 0
        self.fractal_level = 0
        self.level_up_cost = 0
        self.plants = []
        self.flora = []
        self.is_camouflaged = False
        self.camouflage_timer = 0
        self.fractal_surface = None
        
        # self.reset() is called by the environment wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.resources = self.INITIAL_RESOURCES
        self.fractal_level = 1
        self.is_camouflaged = False
        self.camouflage_timer = 0
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        """Initializes state for a new fractal level."""
        self.plants = []
        self.flora = []
        
        self.level_up_cost = int(self.LEVEL_UP_BASE_COST * (self.LEVEL_UP_COST_MULTIPLIER ** (self.fractal_level - 1)))
        
        # Place player away from edges
        self.player_pos = pygame.Vector2(
            self.np_random.integers(100, self.SCREEN_WIDTH - 100),
            self.np_random.integers(100, self.SCREEN_HEIGHT - 100)
        )
        
        # Generate fractal background for the level
        self._generate_fractal_background()
        
        # Spawn flora
        num_flora = 1 + (self.fractal_level - 1) // 3
        base_speed = 1.0 + (self.fractal_level - 1) * 0.1 # Increased speed for better challenge
        
        for _ in range(num_flora):
            self._spawn_flora(base_speed)

    def _spawn_flora(self, speed):
        """Spawns a single flora entity with a random patrol pattern."""
        start_pos = pygame.Vector2(0,0)
        # Ensure flora spawns far from the player
        while start_pos.distance_to(self.player_pos) < 200:
            start_pos.x = self.np_random.uniform(50, self.SCREEN_WIDTH - 50)
            start_pos.y = self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)

        path_type = self.np_random.choice(['horizontal', 'vertical', 'circular'])
        path_data = {}
        
        if path_type == 'horizontal':
            half_width = self.np_random.uniform(50, 150)
            path_data = {
                'min_x': max(20, start_pos.x - half_width),
                'max_x': min(self.SCREEN_WIDTH - 20, start_pos.x + half_width)
            }
        elif path_type == 'vertical':
            half_height = self.np_random.uniform(50, 150)
            path_data = {
                'min_y': max(20, start_pos.y - half_height),
                'max_y': min(self.SCREEN_HEIGHT - 20, start_pos.y + half_height)
            }
        else: # circular
            path_data = {
                'center': start_pos.copy(),
                'radius': self.np_random.uniform(40, 100),
                'angle': self.np_random.uniform(0, 2 * math.pi)
            }

        self.flora.append({
            'pos': start_pos,
            'velocity': pygame.Vector2(self.np_random.choice([-1, 1]), self.np_random.choice([-1, 1])).normalize() * speed,
            'path_type': path_type,
            'path_data': path_data
        })
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        # --- 1. Process Actions ---
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        if movement == 1: self.player_pos.y -= self.PLAYER_SPEED
        elif movement == 2: self.player_pos.y += self.PLAYER_SPEED
        elif movement == 3: self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4: self.player_pos.x += self.PLAYER_SPEED
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)
        
        # Plant / Harvest (Space)
        if space_press:
            plant_under_cursor = None
            for plant in self.plants:
                if pygame.Vector2(plant['pos']).distance_to(self.player_pos) < self.PLAYER_RADIUS + self.PLANT_RADIUS:
                    plant_under_cursor = plant
                    break
            
            if plant_under_cursor:
                if plant_under_cursor['growth'] >= self.PLANT_GROWTH_TIME:
                    # Harvest
                    self.resources += self.PLANT_HARVEST_YIELD
                    self.plants.remove(plant_under_cursor)
                    reward += 1.0 # +1 for harvesting
                    # sfx: harvest_sound
            elif self.resources >= self.PLANT_COST:
                # Plant
                self.resources -= self.PLANT_COST
                self.plants.append({'pos': self.player_pos.copy(), 'growth': 0})
                # sfx: plant_sound
        
        # Camouflage (Shift)
        if shift_press and not self.is_camouflaged and self.resources >= self.CAMO_COST:
            self.resources -= self.CAMO_COST
            self.is_camouflaged = True
            self.camouflage_timer = self.CAMO_DURATION
            # sfx: camo_activate_sound

        # --- 2. Update Game State ---
        # Plant growth
        for plant in self.plants:
            plant['growth'] = min(self.PLANT_GROWTH_TIME, plant['growth'] + 1)
            
        # Camouflage timer
        if self.is_camouflaged:
            self.camouflage_timer -= 1
            reward -= 0.1 # -0.1 penalty per step while camouflaged
            if self.camouflage_timer <= 0:
                self.is_camouflaged = False
                # sfx: camo_deactivate_sound
                
        # Flora movement
        for f in self.flora:
            if f['path_type'] == 'horizontal':
                f['pos'] += f['velocity']
                if f['pos'].x < f['path_data']['min_x'] or f['pos'].x > f['path_data']['max_x']:
                    f['velocity'].x *= -1
            elif f['path_type'] == 'vertical':
                f['pos'] += f['velocity']
                if f['pos'].y < f['path_data']['min_y'] or f['pos'].y > f['path_data']['max_y']:
                    f['velocity'].y *= -1
            elif f['path_type'] == 'circular':
                f['path_data']['angle'] += f['velocity'].length() * 0.02
                f['pos'].x = f['path_data']['center'].x + math.cos(f['path_data']['angle']) * f['path_data']['radius']
                f['pos'].y = f['path_data']['center'].y + math.sin(f['path_data']['angle']) * f['path_data']['radius']

        # --- 3. Check for Events & Termination ---
        # Level up
        if self.resources >= self.level_up_cost:
            self.fractal_level += 1
            reward += 10.0 # +10 for leveling up
            if self.fractal_level > self.MAX_LEVELS:
                self.game_over = True
                reward += 100.0 # +100 for winning
            else:
                self.resources -= self.level_up_cost
                self._setup_level()
                # sfx: level_up_sound
        
        # Termination conditions
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
            
        self.score += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _check_termination(self):
        if self.game_over: # Already terminated by winning
            return True
            
        # Detected by flora
        if not self.is_camouflaged:
            for f in self.flora:
                if f['pos'].distance_to(self.player_pos) < self.FLORA_RADIUS + self.PLAYER_RADIUS:
                    self.game_over = True
                    # sfx: game_over_sound
                    return True
        
        return False
        
    def _get_observation(self):
        # Fill background
        self.screen.fill(self.COLOR_BG)
        
        # Blit pre-rendered fractal
        if self.fractal_surface:
            self.screen.blit(self.fractal_surface, (0, 0))
        
        # Render all game elements
        self._render_game_elements()
        
        # Render UI overlay
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game_elements(self):
        # Draw plants
        for plant in self.plants:
            color = self.COLOR_PLANT_MATURE if plant['growth'] >= self.PLANT_GROWTH_TIME else self.COLOR_PLANT_IMMATURE
            pos = (int(plant['pos'].x), int(plant['pos'].y))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLANT_RADIUS, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLANT_RADIUS, color)

        # Draw flora
        for f in self.flora:
            pos = (int(f['pos'].x), int(f['pos'].y))
            self._draw_glow_circle(self.screen, pos, self.COLOR_FLORA, self.COLOR_FLORA_GLOW, self.FLORA_RADIUS)
    
        # Draw player
        player_int_pos = (int(self.player_pos.x), int(self.player_pos.y))
        self._draw_glow_circle(self.screen, player_int_pos, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, self.PLAYER_RADIUS)

        # Draw camouflage effect
        if self.is_camouflaged:
            # Pulsating effect
            pulse_alpha = 100 + 40 * math.sin(self.steps * 0.2)
            color = (*self.COLOR_CAMO_EFFECT, int(max(0, pulse_alpha)))
            radius = self.PLAYER_RADIUS + 8
            
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, color)
            self.screen.blit(temp_surf, (player_int_pos[0] - radius, player_int_pos[1] - radius))

    def _render_ui(self):
        # Resources
        res_text = self.font_main.render(f"Resources: {self.resources}/{self.level_up_cost}", True, self.COLOR_UI_TEXT)
        self.screen.blit(res_text, (10, 10))
        
        # Camouflage status
        camo_status = "READY" if self.resources >= self.CAMO_COST else "---"
        camo_color = self.COLOR_PLANT_MATURE if self.resources >= self.CAMO_COST else self.COLOR_FLORA
        if self.is_camouflaged:
            camo_status = f"ACTIVE: {self.camouflage_timer}"
            camo_color = self.COLOR_CAMO_EFFECT
        camo_text = self.font_main.render(f"Camouflage: {camo_status}", True, camo_color)
        self.screen.blit(camo_text, (10, 35))
        
        # Level
        level_text = self.font_level.render(f"Fractal Level: {self.fractal_level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 10, 10))
        
        # Score
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - level_text.get_width() - 10, 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.fractal_level,
            "resources": self.resources,
            "is_camouflaged": self.is_camouflaged
        }

    def _generate_fractal_background(self):
        """Creates a pre-rendered surface with a fractal tree pattern."""
        self.fractal_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        self.fractal_surface.fill((0,0,0,0))
        
        # Parameters change with level
        start_len = 80 + self.np_random.uniform(-10, 10)
        angle = 20 + self.fractal_level * 2 + self.np_random.uniform(-2, 2)
        depth = 5 + (self.fractal_level // 3)
        len_ratio = 0.75 + self.np_random.uniform(-0.05, 0.05)
        
        def draw_branch(start_pos, current_angle, length, width):
            if width < 1:
                return
            end_pos = start_pos + pygame.Vector2(length, 0).rotate(current_angle)
            
            # Use alpha to make deeper branches darker
            alpha = int(30 + width * 10)
            color = (*self.COLOR_FRACTAL_BASE, alpha)
            
            pygame.draw.line(self.fractal_surface, color, start_pos, end_pos, int(width))
            
            # Recursive calls
            draw_branch(end_pos, current_angle - angle, length * len_ratio, width * 0.8)
            draw_branch(end_pos, current_angle + angle, length * len_ratio, width * 0.8)

        # Draw multiple "trees" from different edges
        draw_branch(pygame.Vector2(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT), -90, start_len, depth)
        draw_branch(pygame.Vector2(0, self.SCREEN_HEIGHT/2), 0, start_len*0.8, depth-1)
        draw_branch(pygame.Vector2(self.SCREEN_WIDTH, self.SCREEN_HEIGHT/2), 180, start_len*0.8, depth-1)

    def _draw_glow_circle(self, surface, pos, main_color, glow_color, radius):
        """Draws a circle with a soft outer glow."""
        glow_radius = int(radius * 1.8)
        
        # Create a temporary surface for the glow effect
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, glow_radius, glow_color)
        
        # Blit the glow centered on the position
        surface.blit(temp_surf, (pos[0] - glow_radius, pos[1] - glow_radius))

        # Draw the main circle on top
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, main_color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, main_color)
        
    def close(self):
        pygame.quit()

# Example usage:
if __name__ == '__main__':
    # Set a non-dummy driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Manual Play Setup ---
    # This part is for human testing and is not part of the gym environment
    pygame.display.set_caption("Fractal Ecosystem Manager")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    running = True
    
    action = [0, 0, 0] # [movement, space, shift]
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("R: Reset")
    print("Q: Quit")
    
    while running:
        # --- Pygame event handling for manual play ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        
        # Reset action
        action = [0, 0, 0]
        
        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Actions
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}")

        if terminated or truncated:
            print(f"--- Episode Finished ---")
            print(f"Final Score: {info['score']:.2f}, Steps: {info['steps']}, Level: {info['level']}")
            obs, info = env.reset()
            pygame.time.wait(2000) # Pause before restarting
        
        # --- Rendering for manual play ---
        # The observation is already a rendered image, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for consistent game speed

    env.close()